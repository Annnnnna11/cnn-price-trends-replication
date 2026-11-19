"""CPU checks only: geometry, exact paired preparation and corruption guards."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import architecture_spec, read_json, sha256_file, write_json
from run import prepare, verify_root

project = Path(__file__).resolve().parents[2]
for depth, features, parameters in [(2, 38400, 201218), (3, 46080, 708866), (4, 61440, 2707202)]:
    spec = architecture_spec(depth)
    assert (spec["flattened_features"], spec["parameters"]) == (features, parameters)
assert [architecture_spec(d)["block_shapes"][-1]["height_before_pool"] for d in (2,3,4)] == [10,5,3]
for path in (project / "i20r20_ablation_v1").glob("*.py"):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3,11))
print("Depth geometry, parameters, Grad-CAM feature sizes and Python 3.11 syntax: PASS")

base = project / "results/i20r20_main_v1"
baseline_hash = sha256_file(base / "manifest.json")
with tempfile.TemporaryDirectory() as temporary:
    args = SimpleNamespace(baseline_dir=base, data_dir=project / "img_data/monthly_20d",
                           run_root=Path(temporary) / "experiment")
    with redirect_stdout(io.StringIO()):
        prepare(args)
        before = (args.run_root / "prepared.json").read_bytes()
        prepare(args)
    assert before == (args.run_root / "prepared.json").read_bytes()
    ready = verify_root(args.run_root, check_raw=False)
    source = read_json(base / "manifest.json")
    for depth in (2,4):
        directory = args.run_root / f"depth_{depth}"
        variant = read_json(directory / "manifest.json")
        assert variant["baseline_run_id"] == source["run_id"]
        for key, value in source["protocol"].items():
            if key != "version":
                assert variant["protocol"][key] == value
        for name, checksum in source["artifact_hashes"].items():
            assert sha256_file(directory / name) == checksum
        with np.load(directory / "seed_101/split.npz", allow_pickle=False) as split:
            assert len(split["train"]) == 477059 and len(split["validation"]) == 204455
    damaged = args.run_root / "depth_2/manifest.json"
    value = read_json(damaged)
    value["protocol"]["learning_rate"] = 2e-5
    write_json(damaged, value)
    try:
        verify_root(args.run_root, check_raw=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("An edited variant protocol must not be accepted")
assert sha256_file(base / "manifest.json") == baseline_hash
print("Paired catalog/splits/normalization byte identity, unchanged hyperparameters, idempotent preparation: PASS")
print("Variant protocol corruption rejected; baseline manifest unchanged: PASS")
print("No Torch, GPU, test labels or model training were used in this check.")
