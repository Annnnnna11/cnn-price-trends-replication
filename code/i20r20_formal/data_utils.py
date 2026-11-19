"""Data-only utilities; no Torch imports or source-data writes."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "i20r20_formal_v1"
SEEDS = (101, 102, 103, 104, 105)
PROTOCOL = {
    "version": VERSION,
    "task": "I20/R20 binary classification",
    "image_shape": [1, 64, 60],
    "target": "Ret_20d > 0, after excluding non-finite returns",
    "in_sample_years": list(range(1993, 2000)),
    "out_of_sample_years": list(range(2000, 2020)),
    "tail_exclusion": "Date >= 1999-12-01 excluded from in-sample data",
    "volatility_filter": "finite EWMA_vol > 0, consistent with nonzero-variance intent of original code",
    "train_fraction": 0.7,
    "split": "independent seeded random 70/30 split per member, following original code",
    "normalization": "one pixel mean/std fitted to each member's training images only",
    "seeds": list(SEEDS),
    "learning_rate": 1e-5,
    "batch_size": 128,
    "optimizer": "Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0",
    "loss": "unweighted CrossEntropyLoss",
    "dropout": 0.5,
    "maximum_epochs": 50,
    "patience": 2,
    "selection": "minimum validation cross-entropy; exact ties retain earliest epoch",
    "ensemble": "arithmetic mean of five positive-class probabilities, not model selection",
    "precision": "float32, no AMP or TF32",
    "workers": 0,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_paths(data_dir, year):
    prefix = f"20d_month_has_vb_[20]_ma_{year}"
    return (Path(data_dir) / f"{prefix}_images.dat",
            Path(data_dir) / f"{prefix}_labels_w_delay.feather")


def clean_year_frame(frame, year):
    required = ("Date", "StockID", "MarketCap", "Ret_20d", "EWMA_vol")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{year}: missing columns {missing}")
    dates = pd.to_datetime(frame["Date"], errors="raise")
    if dates.isna().any() or not (dates.dt.year == year).all():
        raise ValueError(f"{year}: missing dates or a date outside the file's year")
    if frame["StockID"].isna().any():
        raise ValueError(f"{year}: missing StockID")
    identifiers = frame["StockID"].astype(str)
    keys = pd.DataFrame({"Date": dates, "StockID": identifiers})
    if keys.duplicated().any():
        raise ValueError(f"{year}: duplicate Date/StockID keys")
    returns = pd.to_numeric(frame["Ret_20d"], errors="raise").to_numpy(dtype=np.float64)
    volatility = pd.to_numeric(frame["EWMA_vol"], errors="raise").to_numpy(dtype=np.float64)
    finite_return = np.isfinite(returns)
    valid_volatility = np.isfinite(volatility) & (volatility > 0)
    keep = finite_return & valid_volatility
    before_tail = int(keep.sum())
    if year == 1999:
        keep &= (dates < pd.Timestamp("1999-12-01")).to_numpy()
    rows = np.flatnonzero(keep)
    clean = pd.DataFrame({
        "Year": np.full(len(rows), year, dtype=np.int16),
        "Row": rows.astype(np.int32),
        "Date": dates.iloc[rows].to_numpy(),
        "StockID": identifiers.iloc[rows].to_numpy(),
        "MarketCap": pd.to_numeric(frame["MarketCap"], errors="raise").iloc[rows].to_numpy(),
        "Ret_20d": returns[rows],
        "Label": (returns[rows] > 0).astype(np.uint8),
    })
    report = {
        "year": year,
        "original_rows": len(frame),
        "nonfinite_return_rows": int((~finite_return).sum()),
        "invalid_volatility_rows": int((~valid_volatility).sum()),
        "removed_by_return_or_volatility": int((~(finite_return & valid_volatility)).sum()),
        "tail_rows_removed_after_other_filters": before_tail - len(rows),
        "retained_rows": len(rows),
    }
    return clean, report


def make_split(size, seed):
    if size < 4:
        raise ValueError("Too few valid samples to split")
    permutation = np.random.default_rng(seed).permutation(size)
    boundary = int(size * 0.7)
    return permutation[:boundary], permutation[boundary:]


def fit_normalization(catalog, selected_indices, data_dir, original_counts):
    # Sorted row indices give sequential disk access without materializing all images.
    pixel_sum = square_sum = 0.0
    pixel_count = 0
    subset = catalog.iloc[selected_indices].sort_values(["Year", "Row"])
    for year, part in subset.groupby("Year", sort=True):
        image_path, _ = file_paths(data_dir, int(year))
        images = np.memmap(image_path, mode="r", dtype=np.uint8,
                           shape=(original_counts[int(year)], 1, 64, 60))
        rows = part["Row"].to_numpy(dtype=np.int64)
        for start in range(0, len(rows), 256):
            pixels = images[rows[start:start + 256]].astype(np.float64) / 255.0
            pixel_sum += float(pixels.sum())
            square_sum += float(np.square(pixels).sum())
            pixel_count += pixels.size
        del images
    if not pixel_count:
        raise ValueError("No pixels in training subset")
    mean = pixel_sum / pixel_count
    std = math.sqrt(max(square_sum / pixel_count - mean * mean, 0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Training pixel standard deviation is zero or invalid")
    return {"mean": mean, "std": std, "pixel_count": pixel_count,
            "fitted_on": "this member's training images only"}


def metrics_from_totals(loss_sum, matrix):
    tn, fp, fn, tp = [int(item) for item in np.asarray(matrix).reshape(-1)]
    size = tn + fp + fn + tp
    if size == 0:
        raise ValueError("No evaluated samples")
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"samples": size, "cross_entropy": float(loss_sum) / size,
            "accuracy": (tp + tn) / size,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
            "true_up_fraction": (tp + fn) / size,
            "predicted_up_fraction": (tp + fp) / size,
            "confusion_matrix": [[tn, fp], [fn, tp]]}
