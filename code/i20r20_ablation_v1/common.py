"""Pure utilities for paired depth experiments, with no model imports."""
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import numpy as np

SEEDS = (101, 102, 103, 104, 105)
DEPTHS = (2, 4)

@contextmanager
def experiment_lock(directory):
    path = Path(directory) / "experiment.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another ablation command is using this result directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def file_paths(data_dir, year):
    prefix = f"20d_month_has_vb_[20]_ma_{year}"
    return (Path(data_dir) / f"{prefix}_images.dat", Path(data_dir) / f"{prefix}_labels_w_delay.feather")

def architecture_spec(depth):
    if depth not in (2, 3, 4):
        raise ValueError("Depth must be 2, 3, or 4")
    height, width, previous, parameters = 64, 60, 1, 0
    channels = [64 * 2**layer for layer in range(depth)]
    shapes = []
    for layer, number in enumerate(channels):
        stride = 3 if layer == 0 else 1
        dilation = 2 if layer == 0 else 1
        height = (height + 4 - dilation * 4 - 1) // stride + 1
        before_pool = height
        height = (height + 1) // 2
        parameters += number * previous * 5 * 3 + number + 2 * number
        shapes.append({"channels": number, "height_before_pool": before_pool,
                       "height_after_pool": height, "width": width})
        previous = number
    flattened = channels[-1] * height * width
    parameters += flattened * 2 + 2
    return {"depth": depth, "channels": channels, "flattened_features": flattened,
            "parameters": parameters, "block_shapes": shapes}

def metrics_from_totals(loss_sum, matrix):
    # Same metrics computation as the baseline helper.
    tn, fp, fn, tp = [int(value) for value in np.asarray(matrix).reshape(-1)]
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
