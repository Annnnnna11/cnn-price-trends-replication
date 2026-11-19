"""Prepare exact paired data, check new CNNs, and train depths 2 and 4.

The baseline folder is read-only. No test labels or test metrics are read.
"""
import argparse
from datetime import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np

from common import (SEEDS, DEPTHS, architecture_spec, experiment_lock,
                    read_json, sha256_file, write_json)

HERE = Path(__file__).resolve().parent

def manifest_identity(value):
    return hashlib.sha256(json.dumps({key: item for key, item in value.items() if key != "run_id"}, sort_keys=True).encode()).hexdigest()

def code_hashes():
    return {name: sha256_file(HERE / name) for name in
            ("run.py", "common.py", "models.py", "training_engine.py", "cnn20.py", "study_plan.json")}

def check_files(directory, expected):
    for name, checksum in expected.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise RuntimeError(f"Missing or changed file: {path}")

def verify_data(data, baseline):
    for entry in baseline["inputs"]:
        path = data / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing/changed training input: {path}")
        if path.stat().st_mtime_ns != entry["mtime_ns"] and sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Changed training input: {path}")

def baseline_info(directory):
    base = read_json(directory / "manifest.json")
    plan = read_json(HERE / "study_plan.json")
    if base["run_id"] != plan["baseline_run_id"]:
        raise RuntimeError("Baseline identity does not match the predeclared study")
    if manifest_identity(base) != base["run_id"]:
        raise RuntimeError("Baseline manifest contents changed")
    if sha256_file(HERE / "cnn20.py") != base["code_hashes"]["cnn20.py"]:
        raise RuntimeError("Original CNN source checksum mismatch")
    check_files(directory, base["artifact_hashes"])
    for seed in SEEDS:
        done = read_json(directory / f"seed_{seed}/completed.json")
        if done["run_id"] != base["run_id"] or done["seed"] != seed:
            raise RuntimeError("Baseline member identity mismatch")
        check_files(directory / f"seed_{seed}", {"best.pt": done["checkpoint_hashes"]["best.pt"]})
    return base

def copy_exact(source, destination, checksum):
    if destination.exists() and sha256_file(destination) == checksum:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != checksum:
        raise RuntimeError(f"Copy verification failed: {source}")
    os.replace(temporary, destination)

def verify_root(root, check_raw=True):
    ready = read_json(root / "prepared.json")
    if ready["code_hashes"] != code_hashes():
        raise RuntimeError("Ablation code changed since preparation; do not mix experiments")
    base_dir = Path(ready["baseline_dir"])
    if sha256_file(base_dir / "manifest.json") != ready["baseline_manifest_sha256"]:
        raise RuntimeError("Baseline manifest changed")
    baseline = baseline_info(base_dir)
    if check_raw:
        verify_data(Path(ready["data_dir"]), baseline)
    for depth in DEPTHS:
        directory = root / f"depth_{depth}"
        manifest = read_json(directory / "manifest.json")
        if manifest["run_id"] != ready["variants"][str(depth)] or manifest["depth"] != depth or manifest_identity(manifest) != manifest["run_id"]:
            raise RuntimeError("Variant identity mismatch")
        check_files(directory, manifest["artifact_hashes"])
    return ready

def prepare(args):
    baseline_dir, data, root = args.baseline_dir.resolve(), args.data_dir.resolve(), args.run_root.resolve()
    if root == baseline_dir or root.is_relative_to(baseline_dir) or baseline_dir.is_relative_to(root):
        raise ValueError("Use an independent result directory, separate from the baseline")
    baseline = baseline_info(baseline_dir)
    verify_data(data, baseline)
    root.mkdir(parents=True, exist_ok=True)
    with experiment_lock(root):
        if (root / "prepared.json").exists():
            ready = verify_root(root)
            if ready["baseline_dir"] != str(baseline_dir) or ready["data_dir"] != str(data):
                raise ValueError("Existing ablation belongs to other input directories")
            print(f"Already prepared and verified: {root}", flush=True)
            return
        marker = {"baseline_dir": str(baseline_dir), "data_dir": str(data), "code_hashes": code_hashes()}
        if (root / "preparing.json").exists():
            if read_json(root / "preparing.json") != marker:
                raise RuntimeError("Incomplete preparation used different inputs or code")
        elif any(path.name != "experiment.lock" for path in root.iterdir()):
            raise ValueError("Choose an unused --run-root")
        write_json(root / "preparing.json", marker)
        variants = {}
        for depth in DEPTHS:
            directory = root / f"depth_{depth}"
            directory.mkdir(exist_ok=True)
            if (directory / "manifest.json").exists():
                existing = read_json(directory / "manifest.json")
                if existing["code_hashes"] != code_hashes():
                    raise RuntimeError("Incomplete variant used different code")
                check_files(directory, existing["artifact_hashes"])
                variants[str(depth)] = existing["run_id"]
                continue
            if any(directory.glob("seed_*/last.pt")):
                raise RuntimeError("Checkpoint exists without variant manifest; preserve files and investigate")
            artifacts = dict(baseline["artifact_hashes"])
            for name in ["data_audit.csv"] + [f"seed_{seed}/split_counts.csv" for seed in SEEDS]:
                artifacts[name] = sha256_file(baseline_dir / name)
            for name, checksum in artifacts.items():
                copy_exact(baseline_dir / name, directory / name, checksum)
            for seed in SEEDS:
                with np.load(directory / f"seed_{seed}/split.npz", allow_pickle=False) as split:
                    train, val = split["train"], split["validation"]
                    if not np.array_equal(np.sort(np.r_[train, val]), np.arange(baseline["in_sample_rows"])):
                        raise RuntimeError("Split does not cover each in-sample row exactly once")
            protocol = dict(baseline["protocol"])
            protocol.update({"version": f"i20r20_ablation_v1_depth{depth}", "architecture_depth": depth})
            spec = architecture_spec(depth)
            manifest = {"protocol": protocol, "depth": depth, "parameters": spec["parameters"],
                        "architecture": f"I20 CNN depth {depth}, {spec['parameters']} parameters",
                        "specification": spec, "baseline_run_id": baseline["run_id"],
                        "baseline_dir": str(baseline_dir), "data_dir": str(data),
                        "original_counts": baseline["original_counts"], "inputs": baseline["inputs"],
                        "artifact_hashes": artifacts, "code_hashes": code_hashes(),
                        "in_sample_rows": baseline["in_sample_rows"], "created_at": datetime.now().isoformat()}
            manifest["run_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            write_json(directory / "manifest.json", manifest)
            variants[str(depth)] = manifest["run_id"]
            print(f"Depth {depth}: {spec['parameters']:,} parameters; exact baseline splits/norms copied", flush=True)
        write_json(root / "prepared.json", {**marker, "baseline_manifest_sha256": sha256_file(baseline_dir / "manifest.json"),
                   "baseline_run_id": baseline["run_id"], "variants": variants})
        print("PREPARE: PASS | no new split or normalization fit; baseline unchanged", flush=True)

def configure(torch):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def check(args):
    import torch
    from models import DepthCNN
    from cnn20 import CNN20
    root = args.run_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the Windows cnn20 environment")
    with experiment_lock(root):
        ready = verify_root(root)
        configure(torch)
        checkpoint = torch.load(Path(ready["baseline_dir"]) / "seed_101/best.pt", map_location="cpu", weights_only=True)
        original, generalized = CNN20().to("cuda").eval(), DepthCNN(3).to("cuda").eval()
        original.load_state_dict(checkpoint["model_state"], strict=True)
        generalized.load_state_dict(checkpoint["model_state"], strict=True)
        with torch.inference_mode():
            images = torch.randn(8, 1, 64, 60, device="cuda")
            difference = float((original(images) - generalized(images)).abs().max().item())
        if difference > 1e-6:
            raise RuntimeError("Depth-3 factory does not reproduce original checkpoint output")
        print(f"Depth-3 baseline compatibility: PASS, max difference={difference:.3g}", flush=True)
        del original, generalized, checkpoint, images
        reports = []
        for depth in DEPTHS:
            torch.manual_seed(20261118)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model = DepthCNN(depth).to("cuda").train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            images = torch.randn(128, 1, 64, 60, device="cuda")
            labels = torch.arange(128, device="cuda") % 2
            output = model(images)
            if output.shape != (128, 2):
                raise RuntimeError("Unexpected output shape")
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()
            if not torch.isfinite(loss).item() or any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("Invalid smoke loss or gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise FloatingPointError("Invalid smoke parameters")
            torch.cuda.synchronize()
            report = {**architecture_spec(depth), "batch_size": 128, "seconds": time.perf_counter() - started,
                      "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            reports.append(report)
            print(f"Depth {depth}: forward/backward PASS, {report['parameters']:,} parameters, "
                  f"{report['peak_cuda_allocated_mib']:.1f} MiB allocated", flush=True)
            del optimizer, model, images, labels, output, loss
        write_json(root / "gpu_check.json", {"code_hashes": code_hashes(), "gpu": torch.cuda.get_device_name(0),
                   "baseline_output_max_difference": difference, "checks": reports,
                   "purpose": "Synthetic compatibility test only; weights discarded, no dataset/model files changed"})
        print("GPU CHECK: PASS | temporary smoke weights discarded", flush=True)

def train(args):
    import torch
    import pandas as pd
    import training_engine as engine
    from models import DepthCNN
    root = args.run_root.resolve()
    depths, seeds = args.depths or list(DEPTHS), args.seeds or list(SEEDS)
    if len(set(depths)) != len(depths) or not set(depths).issubset(DEPTHS):
        raise ValueError("Choose depth 2 and/or 4; depth 3 is already trained")
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError("Choose distinct seeds 101-105")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the Windows cnn20 environment")
    with experiment_lock(root), engine.logger(root / "train.log"):
        ready = verify_root(root)
        if not (root / "gpu_check.json").exists() or read_json(root / "gpu_check.json")["code_hashes"] != code_hashes():
            raise RuntimeError("Run check successfully with this package before training")
        configure(torch)
        environment = {"at": datetime.now().isoformat(), "python": sys.version, "python_executable": sys.executable,
                       "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(), "cuda": torch.version.cuda,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "pyarrow")}}
        write_json(root / f"environment_{datetime.now():%Y%m%d_%H%M%S_%f}.json", environment)
        print(f"GPU: {environment['gpu']} | Python: {sys.executable}", flush=True)
        print("Only baseline in-sample data are read; no test labels, predictions or test metrics.", flush=True)
        for depth in depths:
            directory = root / f"depth_{depth}"
            manifest = read_json(directory / "manifest.json")
            catalog = pd.read_feather(directory / "in_sample_catalog.feather")
            print(f"DEPTH {depth}: {manifest['parameters']:,} parameters | from-scratch independent training", flush=True)
            for seed in seeds:
                engine.train_member(directory, manifest, catalog, seed, functools.partial(DepthCNN, depth))
                save_summary(root, pd)
        save_summary(root, pd)
        completed = sum((root / f"depth_{depth}/seed_{seed}/completed.json").exists() for depth in DEPTHS for seed in SEEDS)
        print(f"ABLATION TRAIN: {completed}/10 members complete. Results: {root}", flush=True)

def save_summary(root, pd):
    rows = []
    for depth in DEPTHS:
        for seed in SEEDS:
            path = root / f"depth_{depth}/seed_{seed}/completed.json"
            if path.exists():
                item = read_json(path)
                rows.append({"depth": depth, "seed": seed, "parameters": architecture_spec(depth)["parameters"],
                             "epochs_run": item["epochs_run"], "best_epoch": item["best_epoch"],
                             "validation_cross_entropy": item["validation_metrics"]["cross_entropy"],
                             "validation_accuracy": item["validation_metrics"]["accuracy"],
                             "validation_mcc": item["validation_metrics"]["mcc"], "elapsed_seconds": item["elapsed_seconds"]})
    temporary = root / "training_summary.csv.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, root / "training_summary.csv")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--baseline-dir", type=Path, required=True)
    preparation.add_argument("--data-dir", type=Path, required=True)
    preparation.add_argument("--run-root", type=Path, required=True)
    for command in ("check", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--run-root", type=Path, required=True)
        if command == "train":
            sub.add_argument("--depths", type=int, nargs="+")
            sub.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    if args.command != "prepare" and not (args.run_root / "prepared.json").is_file():
        raise FileNotFoundError("Run prepare before check/train")
    try:
        {"prepare": prepare, "check": check, "train": train}[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted. Rerun train to resume at the last complete epoch; completed members are skipped.", flush=True)
        raise SystemExit(130)

if __name__ == "__main__":
    main()
