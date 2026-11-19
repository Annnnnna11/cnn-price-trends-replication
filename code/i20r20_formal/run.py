"""Course I20/R20 preparation and five-member training entry points.

No testing or portfolio analysis is performed by either command.
Resume repeats an interrupted epoch from its beginning, not from a minibatch.
"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cnn20 import CNN20
from data_utils import (PROTOCOL, SEEDS, clean_year_frame, file_paths,
                        fit_normalization, make_split, metrics_from_totals,
                        read_json, sha256_file, write_json)


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


def code_hashes():
    directory = Path(__file__).resolve().parent
    return {name: sha256_file(directory / name)
            for name in ("run.py", "data_utils.py", "cnn20.py")}


def input_record(path):
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path)}


def verify_manifest(run_dir, manifest):
    if manifest["protocol"] != PROTOCOL or manifest["code_hashes"] != code_hashes():
        raise RuntimeError("Protocol/code changed since preparation; do not mix runs")
    data_dir = Path(manifest["data_dir"])
    for entry in manifest["inputs"]:
        path = data_dir / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing or changed input: {path}")
        # Preparation records complete SHA256 hashes. Rehash files whose mtime changed.
        if path.stat().st_mtime_ns != entry["mtime_ns"] and sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Changed input content: {path}")
    for name, expected in manifest["artifact_hashes"].items():
        if not (run_dir / name).is_file() or sha256_file(run_dir / name) != expected:
            raise RuntimeError(f"Missing or changed prepared artifact: {name}")


def prepare(args):
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if str(data_dir) != manifest["data_dir"]:
                raise RuntimeError("This run directory belongs to another data directory")
            verify_manifest(run_dir, manifest)
            print(f"Already prepared and verified: {run_dir}", flush=True)
            return
        marker = run_dir / "preparing.json"
        if marker.exists():
            previous = read_json(marker)
            if previous["code_hashes"] != code_hashes() or previous["data_dir"] != str(data_dir):
                raise RuntimeError("Incomplete preparation used different code; use a new run directory")
        elif any(path.name != "run.lock" for path in run_dir.iterdir()):
            raise RuntimeError("Run directory is not empty; choose a new --run-dir")
        write_json(marker, {"code_hashes": code_hashes(), "data_dir": str(data_dir)})
        with logger(run_dir / "prepare.log"):
            print(f"PREPARE started: {datetime.now().isoformat(timespec='seconds')}", flush=True)
            print("Reading only 1993-1999. Test images/labels are not loaded.", flush=True)
            parts, reports, inputs = [], [], []
            counts = {}
            for year in PROTOCOL["in_sample_years"]:
                image_path, label_path = file_paths(data_dir, year)
                if not image_path.is_file() or not label_path.is_file():
                    raise FileNotFoundError(f"Missing {year} image/label pair")
                frame = pd.read_feather(label_path)
                if image_path.stat().st_size != len(frame) * 64 * 60:
                    raise ValueError(f"{year}: image byte count does not match label rows")
                clean, report = clean_year_frame(frame, year)
                parts.append(clean)
                reports.append(report)
                counts[year] = len(frame)
                print(f"{year}: original={len(frame)}, retained={len(clean)}; hashing source files...", flush=True)
                inputs.extend((input_record(image_path), input_record(label_path)))
            catalog = pd.concat(parts, ignore_index=True)
            if len(catalog) < 100:
                raise ValueError("Too few valid in-sample rows")
            catalog_path = run_dir / "in_sample_catalog.feather"
            catalog.to_feather(catalog_path)
            pd.DataFrame(reports).to_csv(run_dir / "data_audit.csv", index=False)
            artifact_hashes = {"in_sample_catalog.feather": sha256_file(catalog_path)}
            seed_summaries = []
            for seed in SEEDS:
                seed_dir = run_dir / f"seed_{seed}"
                seed_dir.mkdir(exist_ok=True)
                if (seed_dir / "last.pt").exists():
                    raise RuntimeError("A checkpoint exists without a completed manifest; do not overwrite it")
                train, validation = make_split(len(catalog), seed)
                split_path = seed_dir / "split.npz"
                np.savez_compressed(split_path, train=train, validation=validation)
                print(f"Seed {seed}: train={len(train)}, validation={len(validation)}; fitting normalization...", flush=True)
                normalization = fit_normalization(catalog, train, data_dir, counts)
                normalization["train_up_fraction"] = float(catalog.iloc[train]["Label"].mean())
                normalization_path = seed_dir / "normalization.json"
                write_json(normalization_path, normalization)
                split_summary = []
                for name, indices in (("train", train), ("validation", validation)):
                    for (year, label), size in catalog.iloc[indices].groupby(["Year", "Label"]).size().items():
                        split_summary.append({"split": name, "year": int(year), "label": int(label), "samples": int(size)})
                pd.DataFrame(split_summary).to_csv(seed_dir / "split_counts.csv", index=False)
                for path in (split_path, normalization_path):
                    artifact_hashes[path.relative_to(run_dir).as_posix()] = sha256_file(path)
                seed_summaries.append({"seed": seed, "train": len(train), "validation": len(validation), **normalization})
                print(f"Seed {seed}: mean={normalization['mean']:.6f}, std={normalization['std']:.6f}", flush=True)
            manifest = {"protocol": PROTOCOL, "created_at": datetime.now().isoformat(),
                        "data_dir": str(data_dir), "inputs": inputs, "original_counts": counts,
                        "code_hashes": code_hashes(), "artifact_hashes": artifact_hashes,
                        "data_audit": reports, "seeds": seed_summaries,
                        "in_sample_rows": len(catalog)}
            identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
            manifest["run_id"] = identity
            write_json(manifest_path, manifest)
            print(f"PREPARE: PASS | retained in-sample rows: {len(catalog)}", flush=True)
            print(f"Manifest: {manifest_path}", flush=True)


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
            "run_id": manifest["run_id"], "version": PROTOCOL["version"],
            "architecture": "CNN20, 708866 parameters", "target": PROTOCOL["target"],
            "normalization": normalization, "epoch": state["best_epoch"],
            "validation_metrics": state["best_validation"]}


def train_member(run_dir, manifest, catalog, seed):
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
    train_loader = DataLoader(train_set, batch_size=PROTOCOL["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=PROTOCOL["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)
    model = CNN20().to("cuda")
    if sum(p.numel() for p in model.parameters()) != 708866:
        raise RuntimeError("Unexpected model parameter count")
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL["learning_rate"],
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
    for number in range(state["epoch"] + 1, PROTOCOL["maximum_epochs"] + 1):
        if state["bad_epochs"] >= PROTOCOL["patience"]:
            break
        print(f"Seed {seed}, epoch {number}/{PROTOCOL['maximum_epochs']}", flush=True)
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

    completion = {"run_id": manifest["run_id"], "seed": seed, "epochs_run": state["epoch"],
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


def train(args):
    run_dir = args.run_dir.resolve()
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError("Run prepare successfully before training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate cnn20 and check GPU")
    selected_seeds = list(args.seeds) if args.seeds else list(SEEDS)
    if len(selected_seeds) != len(set(selected_seeds)) or not set(selected_seeds).issubset(SEEDS):
        raise ValueError(f"Seeds must be distinct members of {SEEDS}")
    with run_lock(run_dir), logger(run_dir / "train.log"):
        manifest = read_json(run_dir / "manifest.json")
        verify_manifest(run_dir, manifest)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        environment = {"python": sys.version, "python_executable": sys.executable,
                       "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0),
                       "cuda_runtime": torch.version.cuda,
                       "packages": {name: importlib.metadata.version(name) for name in
                                    ("torch", "numpy", "pandas", "pyarrow", "scikit-learn")}}
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / f"environment_{datetime.now():%Y%m%d_%H%M%S_%f}.json", environment)
        print(f"TRAIN started: {datetime.now().isoformat(timespec='seconds')}", flush=True)
        print(f"Python: {sys.executable}\nGPU: {environment['gpu']}", flush=True)
        print("Test years 2000-2019 are not loaded or scored by this entry point.", flush=True)
        catalog = pd.read_feather(run_dir / "in_sample_catalog.feather")
        for seed in selected_seeds:
            train_member(run_dir, manifest, catalog, seed)
        completed = []
        for seed in SEEDS:
            path = run_dir / f"seed_{seed}" / "completed.json"
            if path.exists():
                item = read_json(path)
                completed.append({"seed": seed, "best_epoch": item["best_epoch"],
                                  "epochs_run": item["epochs_run"],
                                  "validation_cross_entropy": item["validation_metrics"]["cross_entropy"],
                                  "validation_accuracy": item["validation_metrics"]["accuracy"],
                                  "validation_mcc": item["validation_metrics"]["mcc"],
                                  "elapsed_seconds": item["elapsed_seconds"]})
        pd.DataFrame(completed).to_csv(run_dir / "training_summary.csv", index=False)
        print(f"TRAIN: {len(completed)}/5 members complete. Results: {run_dir}", flush=True)
        print("These are separate validation splits; do not average their validation predictions.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="prepare in-sample splits and normalization")
    preparation.add_argument("--data-dir", type=Path, required=True)
    preparation.add_argument("--run-dir", type=Path, required=True)
    training = commands.add_parser("train", help="train/resume five independent CNNs")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--seeds", type=int, nargs="+", help="optional subset of 101 102 103 104 105")
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else train)(args)
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command; training resumes at the last complete epoch.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
