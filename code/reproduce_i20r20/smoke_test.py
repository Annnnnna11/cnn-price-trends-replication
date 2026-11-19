"""Small real-data CUDA test, not a full experiment or accuracy estimate.

Run from the Windows project root:
    python reproduce_i20r20/smoke_test.py --data-dir img_data/monthly_20d

Only 1993 is read. Missing/non-finite Ret_20d values are excluded before
binarization. Images remain on disk as a read-only memory map. This test
writes metadata/metrics in a new directory and saves no model checkpoint.
"""

import argparse
import importlib.metadata
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cnn20 import CNN20


class Charts(Dataset):
    def __init__(self, images, indices, returns, mean, std):
        self.images = images
        self.indices = indices
        self.returns = returns
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        # Copy just one image; never expose the read-only map as a writable tensor.
        image = self.images[row].astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        target = int(self.returns[row] > 0)
        return torch.from_numpy(image), target


def train_statistics(images, indices):
    # Compute statistics from this smoke test's training subset only.
    # Float64 sums avoid allocating the whole image dataset in float32.
    pixel_sum = 0.0
    square_sum = 0.0
    count = 0
    for start in range(0, len(indices), 128):
        pixels = images[indices[start:start + 128]].astype(np.float64) / 255.0
        pixel_sum += float(pixels.sum())
        square_sum += float(np.square(pixels).sum())
        count += pixels.size
    mean = pixel_sum / count
    variance = max(square_sum / count - mean * mean, 0.0)
    std = variance ** 0.5
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training images have invalid or zero standard deviation")
    return mean, std


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite cross-entropy loss")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                       for p in model.parameters()):
                    raise FloatingPointError("Non-finite model gradients")
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    torch.cuda.synchronize()
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "samples": total,
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/i20r20_smoke"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("batch-size must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: activate cnn20 and check GPU installation")

    seed = 20261117
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    data_dir = args.data_dir.resolve()
    prefix = "20d_month_has_vb_[20]_ma_1993"
    image_path = data_dir / f"{prefix}_images.dat"
    label_path = data_dir / f"{prefix}_labels_w_delay.feather"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"1993 data files not found in {data_dir}")
    labels = pd.read_feather(label_path)
    if image_path.stat().st_size != len(labels) * 64 * 60:
        raise ValueError("Image byte count does not match label row count")
    returns = pd.to_numeric(labels["Ret_20d"], errors="raise").to_numpy(dtype=np.float64)
    dates = pd.to_datetime(labels["Date"], errors="raise")
    valid = np.isfinite(returns) & (dates.dt.year.to_numpy() == 1993)
    indices = np.random.default_rng(seed).permutation(np.flatnonzero(valid))
    n_train, n_val = 2048, 512
    if len(indices) < n_train + n_val:
        raise ValueError("Not enough valid 1993 samples for the smoke test")
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    images = np.memmap(image_path, dtype=np.uint8, mode="r",
                       shape=(len(labels), 1, 64, 60))
    mean, std = train_statistics(images, train_indices)
    train_set = Charts(images, train_indices, returns, mean, std)
    val_set = Charts(images, val_indices, returns, mean, std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    device = torch.device("cuda:0")
    model = CNN20().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 708866:
        raise ValueError(f"Unexpected parameter count: {parameters}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Parameters: {parameters}; input: 1 x 64 x 60; classes: 2", flush=True)
    print(f"1993 labels: {len(labels)}; finite returns: {np.isfinite(returns).sum()}", flush=True)
    print(f"Smoke samples: train={n_train}, validation={n_val}; batch={args.batch_size}", flush=True)
    print(f"Train-only normalization: mean={mean:.6f}, std={std:.6f}", flush=True)
    history = []
    for epoch in (1, 2):
        train = run_epoch(model, train_loader, device, optimizer)
        validation = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(f"Epoch {epoch}/2 | train loss={train['loss']:.6f} "
              f"acc={train['accuracy']:.4f} time={train['seconds']:.2f}s | "
              f"val loss={validation['loss']:.6f} acc={validation['accuracy']:.4f} "
              f"time={validation['seconds']:.2f}s", flush=True)

    output_dir = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pandas", "pyarrow", "scikit-learn")
    }
    report = {
        "purpose": "pipeline smoke test only; not full-experiment accuracy or runtime",
        "implementation": "i20r20_smoke_v1; CNN architecture adapted from authors' code",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "data_dir": str(data_dir),
        "seed": seed,
        "parameters": parameters,
        "target": "Ret_20d > 0, after excluding non-finite returns",
        "year": 1993,
        "n_train": n_train,
        "n_validation": n_val,
        "batch_size": args.batch_size,
        "epochs": 2,
        "learning_rate": 1e-5,
        "normalization_mean": mean,
        "normalization_std": std,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "history": history,
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Peak allocated GPU memory: {report['peak_cuda_allocated_mib']:.1f} MiB", flush=True)
    print(f"Report: {report_path}", flush=True)
    print("SMOKE TEST: PASS", flush=True)


if __name__ == "__main__":
    main()
