"""Exercise complete analysis with synthetic predictions and mocked Arrow I/O.

Uses only temporary files plus read-only baseline metadata. Never reads actual
test labels or loads a model. This test does not prove CUDA/Arrow execution.
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import freeze, analyze, year_paths
from evaluation_core import (SEEDS, YEARS, paths_for_year, prepare_labels,
                             read_json, write_json, source_record, sha256_file)

project = Path(__file__).resolve().parents[2]
original_reader = pd.read_feather
fixtures = {}
try:
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        data = directory / "data"
        data.mkdir()
        output = directory / "evaluation"
        with redirect_stdout(io.StringIO()):
            freeze(SimpleNamespace(train_dir=project / "results/i20r20_main_v1",
                                   data_dir=data, eval_dir=output))
        frozen = read_json(output / "frozen_evaluation.json")
        (output / "predictions").mkdir()
        for year in YEARS:
            image, label = paths_for_year(data, year)
            image.write_bytes(f"synthetic-image-{year}".encode())
            label.write_bytes(f"synthetic-label-{year}".encode())
            raw = pd.DataFrame({"Date": [pd.Timestamp(year, 1, 31)] * 20,
                                "StockID": np.arange(20), "MarketCap": np.arange(1, 21),
                                "Ret_20d": np.r_[np.full(10, -.1), np.full(10, .1)],
                                "Ret_month": np.arange(20) / 100 - .1,
                                "EWMA_vol": np.ones(20)})
            frame = prepare_labels(raw, year)
            for seed in SEEDS:
                frame[f"p_{seed}"] = np.linspace(.01, .99, 20).astype(np.float32)
            frame["p_ensemble"] = frame[[f"p_{seed}" for seed in SEEDS]].mean(axis=1).astype(np.float32)
            prediction, complete = year_paths(output, year)
            frame.to_csv(prediction, index=False)
            fixtures[str(prediction.resolve())] = frame
            write_json(complete, {"evaluation_id": frozen["evaluation_id"], "year": year,
                       "image_source": source_record(image), "label_source": source_record(label),
                       "prediction_sha256": sha256_file(prediction)})

        pd.read_feather = lambda path: fixtures[str(Path(path).resolve())].copy()
        with redirect_stdout(io.StringIO()):
            analyze(SimpleNamespace(eval_dir=output))
        analysis = output / "analysis"
        metrics = pd.read_csv(analysis / "classification_metrics.csv")
        main = metrics.loc[(metrics["period"] == "course_2000_2019") & (metrics["method"] == "ensemble")].iloc[0]
        assert main["samples"] == 400 and main["accuracy"] == 1 and main["roc_auc"] == 1
        common = metrics.loc[(metrics["period"] == "common_2001_2019") & (metrics["method"] == "ensemble")].iloc[0]
        assert common["samples"] == 380
        assert read_json(analysis / "analysis_complete.json")["classification_rows"] == 400
        assert read_json(analysis / "financial_status.json")["status"] == "computed"
        assert (analysis / "test_report.md").is_file()
        literature = pd.read_csv(analysis / "paper_comparison_I20R20.csv")
        assert len(literature) == 44
        paper_ew = literature.loc[(literature["period"] == "course_2000_2019") &
                                  (literature["weighting"] == "EW") & (literature["portfolio"] == "H-L")].iloc[0]
        assert paper_ew["paper_annualized_return"] == .21 and paper_ew["paper_sharpe"] == 2.16
        print("Complete analysis control flow, course/common-period sample masks and final report: PASS")
        print("Plot status:", read_json(analysis / "plot_status.json"))
finally:
    pd.read_feather = original_reader

print("Synthetic data only; Arrow reader was mocked; actual CUDA and test scoring remain unexecuted.")
