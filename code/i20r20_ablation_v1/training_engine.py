"""Training loop adapted from the GPU-tested baseline runner.

Only the model factory, parameter-count/architecture metadata and protocol
lookup are generalized. Dataset access, RNG restoration, optimizer, epoch
metrics, checkpoint commit order and early stopping retain the baseline logic.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import os
import random
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from common import file_paths, metrics_from_totals, read_json, sha256_file, write_json


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@contextmanager
def logger(path):
    with Path(path).open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                yield
            except Exception:
                traceback.print_exc()
                raise


@contextmanager
def run_lock(run_dir):
    """OS-owned lock is automatically released even if the process is killed."""
    path = run_dir / "run.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another prepare/train process is using this run directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Charts(Dataset):
    def __init__(self, catalog, indices, data_dir, original_counts, normalization):
        part = catalog.iloc[indices]
        self.years = part["Year"].to_numpy(dtype=np.int16)
        self.rows = part["Row"].to_numpy(dtype=np.int32)
        self.labels = part["Label"].to_numpy(dtype=np.int64)
        self.data_dir, self.original_counts = data_dir, original_counts
        self.mean, self.std = normalization["mean"], normalization["std"]
        self.maps = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        year = int(self.years[index])
        if year not in self.maps:
            path, _ = file_paths(self.data_dir, year)
            self.maps[year] = np.memmap(path, mode="r", dtype=np.uint8,
                                       shape=(self.original_counts[year], 1, 64, 60))
        image = self.maps[year][int(self.rows[index])].astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return torch.from_numpy(image), int(self.labels[index])


def epoch(model, loader, optimizer, phase):
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss()
    confusion = torch.zeros(4, dtype=torch.int64, device="cuda")
    loss_sum = 0.0
    seen = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch_number, (images, labels) in enumerate(loader, 1):
            images = images.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite {phase} loss")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            confusion += torch.bincount(labels * 2 + logits.argmax(dim=1), minlength=4)
            seen += len(labels)
            if batch_number % 250 == 0 or batch_number == len(loader):
                elapsed = time.perf_counter() - started
                print(f"  {phase}: batch {batch_number}/{len(loader)}, "
                      f"CE={loss_sum / seen:.6f}, elapsed={elapsed:.1f}s", flush=True)
    torch.cuda.synchronize()
    metrics = metrics_from_totals(loss_sum, confusion.cpu().numpy())
    metrics["seconds"] = time.perf_counter() - started
    return metrics


def save_checkpoint(path, state):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_history(seed_dir, history):
    rows = []
    for item in history:
        row = {"epoch": item["epoch"], "improved": item["improved"]}
        for phase in ("train", "validation"):
            row.update({f"{phase}_{key}": value for key, value in item[phase].items()
                        if key != "confusion_matrix"})
        rows.append(row)
    temporary = seed_dir / "history.csv.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, seed_dir / "history.csv")


def best_payload(state, seed, manifest, normalization):
    return {"model_state": state["best_model_state"], "seed": seed,
            "run_id": manifest["run_id"], "version": manifest['protocol']["version"],
            "architecture": manifest["architecture"], "depth": manifest["depth"],
            "baseline_run_id": manifest["baseline_run_id"], "target": manifest['protocol']["target"],
            "normalization": normalization, "epoch": state["best_epoch"],
            "validation_metrics": state["best_validation"]}


def train_member(run_dir, manifest, catalog, seed, model_factory):
    seed_dir = run_dir / f"seed_{seed}"
    if (seed_dir / "completed.json").exists():
        completion = read_json(seed_dir / "completed.json")
        if completion["run_id"] != manifest["run_id"]:
            raise RuntimeError("Completed checkpoint belongs to a different experiment")
        for name, expected in completion["checkpoint_hashes"].items():
            if not (seed_dir / name).is_file() or sha256_file(seed_dir / name) != expected:
                raise RuntimeError(f"Completed checkpoint missing/changed: {seed_dir / name}")
        print(f"Seed {seed}: already completed, verified and skipped", flush=True)
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with np.load(seed_dir / "split.npz", allow_pickle=False) as split:
        train_indices, val_indices = split["train"], split["validation"]
    normalization = read_json(seed_dir / "normalization.json")
    counts = {int(year): count for year, count in manifest["original_counts"].items()}
    train_set = Charts(catalog, train_indices, Path(manifest["data_dir"]), counts, normalization)
    val_set = Charts(catalog, val_indices, Path(manifest["data_dir"]), counts, normalization)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=manifest['protocol']["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=manifest['protocol']["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)
    model = model_factory().to("cuda")
    if sum(p.numel() for p in model.parameters()) != manifest["parameters"]:
        raise RuntimeError("Unexpected model parameter count")
    optimizer = torch.optim.Adam(model.parameters(), lr=manifest['protocol']["learning_rate"],
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    last_path = seed_dir / "last.pt"
    state = {"epoch": 0, "best_epoch": 0, "bad_epochs": 0, "history": [],
             "best_validation": {"cross_entropy": float("inf")}}
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["run_id"] != manifest["run_id"] or state["seed"] != seed:
            raise RuntimeError("Resume checkpoint identity does not match")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        # last.pt is authoritative if interruption occurred between file writes.
        save_checkpoint(seed_dir / "best.pt", best_payload(state, seed, manifest, normalization))
        save_history(seed_dir, state["history"])
        print(f"Seed {seed}: resuming after complete epoch {state['epoch']}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    previous_time = float(state.get("elapsed_seconds", 0))
    member_started = time.perf_counter()
    print(f"Seed {seed}: train={len(train_set)}, validation={len(val_set)}", flush=True)
    for number in range(state["epoch"] + 1, manifest['protocol']["maximum_epochs"] + 1):
        if state["bad_epochs"] >= manifest['protocol']["patience"]:
            break
        print(f"Seed {seed}, epoch {number}/{manifest['protocol']['maximum_epochs']}", flush=True)
        train_metrics = epoch(model, train_loader, optimizer, "train")
        val_metrics = epoch(model, val_loader, None, "validation")
        improved = val_metrics["cross_entropy"] < state["best_validation"]["cross_entropy"]
        if improved:
            state["best_epoch"] = number
            state["best_validation"] = val_metrics
            state["best_model_state"] = {name: tensor.detach().cpu().clone()
                                         for name, tensor in model.state_dict().items()}
            state["bad_epochs"] = 0
        else:
            state["bad_epochs"] += 1
        state["history"].append({"epoch": number, "train": train_metrics,
                                  "validation": val_metrics, "improved": improved})
        state.update({"epoch": number, "run_id": manifest["run_id"], "seed": seed,
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                      "loader_rng": generator.get_state(),
                      "elapsed_seconds": previous_time + time.perf_counter() - member_started,
                      "peak_cuda_allocated_mib": max(float(state.get("peak_cuda_allocated_mib", 0)),
                                                       torch.cuda.max_memory_allocated() / 2**20)})
        # Commit authoritative state first; derived files can be rebuilt on resume.
        save_checkpoint(last_path, state)
        save_checkpoint(seed_dir / "best.pt", best_payload(state, seed, manifest, normalization))
        save_history(seed_dir, state["history"])
        print(f"Seed {seed} epoch {number}: train CE={train_metrics['cross_entropy']:.6f} "
              f"acc={train_metrics['accuracy']:.4f}; validation CE={val_metrics['cross_entropy']:.6f} "
              f"acc={val_metrics['accuracy']:.4f} MCC={val_metrics['mcc']:.4f}; "
              f"best={state['best_epoch']}, no-improvement={state['bad_epochs']}/2; SAVED", flush=True)

    completion = {"run_id": manifest["run_id"], "seed": seed,
                  "depth": manifest["depth"], "baseline_run_id": manifest["baseline_run_id"], "epochs_run": state["epoch"],
                  "best_epoch": state["best_epoch"], "validation_metrics": state["best_validation"],
                  "elapsed_seconds": state["elapsed_seconds"],
                  "peak_cuda_allocated_mib": state["peak_cuda_allocated_mib"],
                  "stop_reason": "early_stopping" if state["bad_epochs"] >= 2 else "maximum_epochs",
                  "finished_at": datetime.now().isoformat(),
                  "checkpoint_hashes": {name: sha256_file(seed_dir / name) for name in ("best.pt", "last.pt")}}
    write_json(seed_dir / "completed.json", completion)
    print(f"Seed {seed}: COMPLETE | best epoch {state['best_epoch']}", flush=True)
    del state, model, optimizer, train_loader, val_loader, train_set, val_set
    torch.cuda.empty_cache()
