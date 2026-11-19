"""Check frozen metadata against existing artifacts in a temporary directory."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import freeze, verify_frozen
from evaluation_core import read_json, write_json

project = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    args = SimpleNamespace(train_dir=project / "results/i20r20_main_v1",
                           data_dir=project / "img_data/monthly_20d", eval_dir=directory)
    with redirect_stdout(io.StringIO()):
        freeze(args)
        first = read_json(directory / "frozen_evaluation.json")
        freeze(args)
    second = verify_frozen(directory)
    assert first == second
    assert len(second["members"]) == 5
    assert second["study_plan"]["architecture_extension"]["depths"] == [2, 3, 4]
    # Detect accidental mismatch instead of reusing a stale evaluation cache.
    second["code_hashes"]["pipeline.py"] = "invalid"
    write_json(directory / "frozen_evaluation.json", second)
    try:
        verify_frozen(directory)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Changed evaluation code must be rejected")
print("Real baseline artifact verification, idempotent freeze and changed-code guard: PASS")
print("Temporary output only. Original training files and test labels were not modified/read.")
