"""Frozen depth-3 baseline evaluation. Run freeze, predict, then analyze.

Only predict imports Torch. No command trains or changes the saved models.
Prediction resumes at year boundaries, retaining completed year files.
"""

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

from evaluation_core import (SEEDS, YEARS, classification_tables, calibration_table,
                             markdown_table, monthly_portfolios, portfolio_summary,
                             literature_comparison,
                             paths_for_year, prepare_labels, read_json, sha256_file,
                             source_record, unchanged_source, write_json)


HERE = Path(__file__).resolve().parent


def own_hashes():
    return {name: sha256_file(HERE / name) for name in
            ("pipeline.py", "evaluation_core.py", "cnn20.py", "study_plan.json", "paper_reference.json")}


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream

    def write(self, text):
        self.terminal.write(text)
        self.stream.write(text)

    def flush(self):
        self.terminal.flush()
        self.stream.flush()


@contextmanager
def logger(path):
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            try:
                yield
            except Exception:
                traceback.print_exc()
                raise


@contextmanager
def lock(directory):
    path = directory / "evaluation.lock"
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
            raise RuntimeError("Another evaluation command is using this output directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_frozen(directory):
    frozen = read_json(directory / "frozen_evaluation.json")
    if frozen["code_hashes"] != own_hashes() or frozen["study_plan"] != read_json(HERE / "study_plan.json"):
        raise RuntimeError("Evaluation code or study plan changed; do not mix evaluation versions")
    training = Path(frozen["training_dir"])
    if sha256_file(training / "manifest.json") != frozen["training_manifest_sha256"]:
        raise RuntimeError("Baseline training manifest changed")
    for member in frozen["members"]:
        folder = training / f"seed_{member['seed']}"
        for name, expected in member["files"].items():
            if not (folder / name).is_file() or sha256_file(folder / name) != expected:
                raise RuntimeError(f"Missing/changed baseline artifact: {folder / name}")
    return frozen


def freeze(args):
    training, data, directory = args.train_dir.resolve(), args.data_dir.resolve(), args.eval_dir.resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"Data directory missing: {data}")
    plan = read_json(HERE / "study_plan.json")
    manifest = read_json(training / "manifest.json")
    if manifest["run_id"] != plan["baseline_run_id"]:
        raise ValueError("Training run does not match the declared baseline")
    if sha256_file(HERE / "cnn20.py") != manifest["code_hashes"]["cnn20.py"]:
        raise ValueError("CNN implementation differs from the trained model implementation")
    members = []
    for seed in SEEDS:
        folder = training / f"seed_{seed}"
        done = read_json(folder / "completed.json")
        norm = read_json(folder / "normalization.json")
        if done["run_id"] != manifest["run_id"] or done["seed"] != seed:
            raise ValueError(f"Seed {seed}: completion identity mismatch")
        if sha256_file(folder / "best.pt") != done["checkpoint_hashes"]["best.pt"]:
            raise ValueError(f"Seed {seed}: best.pt checksum mismatch")
        relative = f"seed_{seed}/normalization.json"
        if sha256_file(folder / "normalization.json") != manifest["artifact_hashes"][relative]:
            raise ValueError(f"Seed {seed}: normalization checksum mismatch")
        if not np.isfinite(norm["mean"]) or not np.isfinite(norm["std"]) or norm["std"] <= 0:
            raise ValueError(f"Seed {seed}: invalid normalization")
        members.append({"seed": seed, "best_epoch": done["best_epoch"], "normalization": norm,
                        "files": {name: sha256_file(folder / name) for name in
                                  ("best.pt", "normalization.json", "completed.json")}})
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory):
        if (directory / "frozen_evaluation.json").exists():
            frozen = verify_frozen(directory)
            if frozen["training_dir"] != str(training) or frozen["data_dir"] != str(data):
                raise ValueError("Existing evaluation belongs to other input directories")
            print(f"Already frozen and verified: {directory}", flush=True)
            return
        if any(path.name != "evaluation.lock" for path in directory.iterdir()):
            raise ValueError("Choose an unused --eval-dir")
        payload = {"version": "i20r20_eval_v1", "frozen_at": datetime.now().isoformat(),
                   "training_dir": str(training), "data_dir": str(data),
                   "baseline_run_id": manifest["run_id"],
                   "training_manifest_sha256": sha256_file(training / "manifest.json"),
                   "study_plan": plan, "members": members, "code_hashes": own_hashes()}
        payload["evaluation_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        write_json(directory / "frozen_evaluation.json", payload)
        print("FREEZE: PASS | paired-depth and Grad-CAM rules recorded; no test labels read", flush=True)
        print(f"Output: {directory}", flush=True)


def year_paths(directory, year):
    return directory / "predictions" / f"{year}.feather", directory / "predictions" / f"{year}.complete.json"


def verify_year(directory, frozen, year):
    prediction, completion = year_paths(directory, year)
    if not completion.exists():
        return False
    record = read_json(completion)
    if record["evaluation_id"] != frozen["evaluation_id"] or record["year"] != year:
        raise ValueError(f"Year {year}: prediction belongs to another frozen evaluation")
    if not prediction.is_file() or sha256_file(prediction) != record["prediction_sha256"]:
        raise ValueError(f"Year {year}: prediction file missing or changed")
    image, label = paths_for_year(frozen["data_dir"], year)
    if not unchanged_source(image, record["image_source"]) or not unchanged_source(label, record["label_source"]):
        raise ValueError(f"Year {year}: source data changed")
    return True


def predict(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from cnn20 import CNN20

    directory = args.eval_dir.resolve()
    if not (directory / "frozen_evaluation.json").is_file():
        raise FileNotFoundError("Run freeze successfully before predict")
    selected = list(args.years) if args.years else list(YEARS)
    if len(set(selected)) != len(selected) or not set(selected).issubset(YEARS):
        raise ValueError("Prediction years must be distinct members of 2000-2019")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use the Windows cnn20 environment")
    with lock(directory), logger(directory / "predict.log"):
        frozen = verify_frozen(directory)
        (directory / "predictions").mkdir(exist_ok=True)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        environment = {"at": datetime.now().isoformat(), "python": sys.version,
                       "python_executable": sys.executable, "platform": platform.platform(),
                       "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "pyarrow")}}
        write_json(directory / f"runtime_{datetime.now():%Y%m%d_%H%M%S_%f}.json", environment)
        print(f"GPU: {environment['gpu']} | Python: {sys.executable}", flush=True)
        models = []
        for member in frozen["members"]:
            seed = member["seed"]
            checkpoint = torch.load(Path(frozen["training_dir"]) / f"seed_{seed}" / "best.pt",
                                     map_location="cpu", weights_only=True)
            if checkpoint["run_id"] != frozen["baseline_run_id"] or checkpoint["seed"] != seed:
                raise ValueError(f"Seed {seed}: checkpoint identity mismatch")
            if checkpoint["normalization"] != member["normalization"] or checkpoint["epoch"] != member["best_epoch"]:
                raise ValueError(f"Seed {seed}: checkpoint normalization/epoch mismatch")
            model = CNN20()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.to("cuda").eval()
            models.append((seed, model, member["normalization"]))
            print(f"Seed {seed}: best epoch {member['best_epoch']}, checkpoint loaded", flush=True)

        class Images(Dataset):
            def __init__(self, mapped, rows):
                self.mapped, self.rows = mapped, rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return torch.from_numpy(self.mapped[int(self.rows[index])].copy())

        for year in selected:
            if verify_year(directory, frozen, year):
                print(f"{year}: complete prediction verified and skipped", flush=True)
                continue
            image_path, label_path = paths_for_year(frozen["data_dir"], year)
            print(f"{year}: auditing label structure and hashing source files...", flush=True)
            labels = prepare_labels(pd.read_feather(label_path), year)
            if image_path.stat().st_size != len(labels) * 64 * 60:
                raise ValueError(f"{year}: image/label byte-count mismatch")
            image_record, label_record = source_record(image_path), source_record(label_path)
            rows = np.flatnonzero(labels["eligible_input"].to_numpy())
            if len(rows) == 0:
                raise ValueError(f"{year}: no eligible images")
            mapped = np.memmap(image_path, mode="r", dtype=np.uint8, shape=(len(labels), 1, 64, 60))
            dataset = Images(mapped, rows)
            loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
            probabilities = np.full((len(labels), len(models)), np.nan, dtype=np.float32)
            cursor = 0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                for batch_number, raw in enumerate(loader, 1):
                    pixels = raw.to("cuda", dtype=torch.float32, non_blocking=True) / 255.0
                    batch = torch.stack([torch.softmax(model((pixels - norm["mean"]) / norm["std"]), dim=1)[:, 1]
                                         for _, model, norm in models], dim=1).cpu().numpy()
                    if not np.isfinite(batch).all() or ((batch < 0) | (batch > 1)).any():
                        raise FloatingPointError("Invalid model probability")
                    probabilities[rows[cursor:cursor + len(batch)]] = batch
                    cursor += len(batch)
                    if batch_number % 100 == 0 or batch_number == len(loader):
                        elapsed = time.perf_counter() - started
                        print(f"{year}: batch {batch_number}/{len(loader)}; images={cursor}/{len(rows)}; "
                              f"elapsed={elapsed:.1f}s; all five models", flush=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if cursor != len(rows):
                raise RuntimeError("Prediction row count mismatch")
            for column, (seed, _, _) in enumerate(models):
                labels[f"p_{seed}"] = probabilities[:, column]
            ensemble = np.full(len(labels), np.nan, dtype=np.float32)
            ensemble[rows] = probabilities[rows].mean(axis=1, dtype=np.float64).astype(np.float32)
            labels["p_ensemble"] = ensemble
            if not unchanged_source(image_path, image_record) or not unchanged_source(label_path, label_record):
                raise RuntimeError("Source data changed during prediction")
            prediction_path, completion_path = year_paths(directory, year)
            temporary = prediction_path.with_name(prediction_path.name + ".tmp")
            labels.to_feather(temporary)
            os.replace(temporary, prediction_path)
            record = {"evaluation_id": frozen["evaluation_id"], "year": year,
                      "rows": len(labels), "predicted_rows": len(rows), "inference_seconds": seconds,
                      "images_per_second_all_five": len(rows) / seconds,
                      "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "image_source": image_record, "label_source": label_record,
                      "prediction_sha256": sha256_file(prediction_path),
                      "completed_at": datetime.now().isoformat()}
            write_json(completion_path, record)
            print(f"{year}: PREDICTION SAVED | {seconds/60:.2f} min | {record['peak_cuda_allocated_mib']:.1f} MiB allocated", flush=True)
            del loader, dataset, mapped
        completed = sum((year_paths(directory, year)[1]).exists() for year in YEARS)
        print(f"PREDICT: {completed}/20 years saved. No training performed.", flush=True)


def make_plots(folder, classification, calibration, monthly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = {}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "savefig.facecolor": "white"})
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for method, title in (("ensemble", "Five-model ensemble"), ("ensemble_training_prior", "Training-prior baseline"),
                          ("always_up", "Always up")):
        selection = classification.loc[(classification["method"] == method) & classification["period"].str.startswith("year_")].copy()
        selection["year"] = selection["period"].str.replace("year_", "", regex=False).astype(int)
        selection = selection.sort_values("year")
        axis.plot(selection["year"], 100 * selection["accuracy"], label=title, marker=".")
    axis.set(title="I20/R20 test classification accuracy by formation year", xlabel="Year", ylabel="Accuracy (%)")
    axis.legend(frameon=False)
    figure.savefig(folder / "test_accuracy_by_year.png", dpi=160)
    plt.close(figure)
    result["test_accuracy_by_year.png"] = "generated"

    selection = calibration.loc[(calibration["method"] == "p_ensemble") & (calibration["samples"] > 0)]
    figure, axis = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
    axis.plot(selection["mean_probability"], selection["observed_up_fraction"], marker="o", label="Ensemble")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted up probability", ylabel="Observed up fraction",
             title="Test-set reliability diagram")
    axis.legend(frameon=False)
    figure.savefig(folder / "test_calibration.png", dpi=160)
    plt.close(figure)
    result["test_calibration.png"] = "generated"

    if monthly.empty:
        result["monthly_wealth.png"] = "not generated: no supported monthly portfolio"
        return result
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, weighting in zip(axes, ("EW", "VW")):
        part = monthly.loc[monthly["weighting"] == weighting].sort_values("Date")
        months = pd.PeriodIndex(part["Date"], freq="M")
        missing = pd.period_range(months.min(), months.max(), freq="M").difference(months)
        if len(missing) or (part[["D1", "D10", "H-L"]] <= -1).any().any():
            axis.text(.5, .5, "Wealth not compounded:\ninternal gaps or return <= -100%", ha="center", va="center", transform=axis.transAxes)
            result[f"wealth_{weighting}"] = "not compounded: internal missing month or return <= -1"
        else:
            for column, title in (("D1", "Low"), ("D10", "High"), ("H-L", "H-L")):
                axis.plot(part["ReturnMonth"], np.cumprod(1 + part[column].to_numpy()), label=title)
            axis.legend(frameon=False)
            result[f"wealth_{weighting}"] = "generated; historical complete-case, gross, no trading costs"
        axis.set(title=f"{weighting}: monthly complete-case return proxy", xlabel="Return month", ylabel="Gross wealth (initial=1)")
    figure.savefig(folder / "monthly_wealth.png", dpi=160)
    plt.close(figure)
    return result


def analyze(args):
    directory = args.eval_dir.resolve()
    if not (directory / "frozen_evaluation.json").is_file():
        raise FileNotFoundError("Run freeze successfully before analyze")
    with lock(directory), logger(directory / "analysis.log"):
        frozen = verify_frozen(directory)
        missing = [year for year in YEARS if not verify_year(directory, frozen, year)]
        if missing:
            raise RuntimeError(f"Complete prediction for all 20 years before final analysis. Missing: {missing}")
        frames = [pd.read_feather(year_paths(directory, year)[0]) for year in YEARS]
        frame = pd.concat(frames, ignore_index=True)
        if frame.duplicated(["Date", "StockID"]).any():
            raise ValueError("Duplicate keys across year prediction files")
        eligible = frame["eligible_input"].to_numpy(dtype=bool)
        probabilities = frame[[f"p_{seed}" for seed in SEEDS] + ["p_ensemble"]].to_numpy()
        if not np.isfinite(probabilities[eligible]).all():
            raise ValueError("Missing probability for an input-eligible sample")
        if ((probabilities[eligible] < 0) | (probabilities[eligible] > 1)).any():
            raise ValueError("Probability outside [0,1]")
        expected = probabilities[eligible, :5].mean(axis=1, dtype=np.float64)
        if not np.allclose(expected, probabilities[eligible, 5], rtol=0, atol=1e-7):
            raise ValueError("Saved ensemble does not equal the five-model mean")
        output = directory / "analysis"
        output.mkdir(exist_ok=True)
        priors = {member["seed"]: member["normalization"]["train_up_fraction"] for member in frozen["members"]}
        classification = classification_tables(frame, priors, frozen["study_plan"]["probability_clip_epsilon"])
        classification.to_csv(output / "classification_metrics.csv", index=False)
        calibration = calibration_table(frame)
        calibration.to_csv(output / "calibration_bins.csv", index=False)
        quality = []
        for year, part in frame.groupby("Year", sort=True):
            ok = part["eligible_input"].to_numpy(dtype=bool)
            finite = np.isfinite(part["Ret_20d"].to_numpy())
            quality.append({"year": int(year), "original_rows": len(part), "input_eligible": int(ok.sum()),
                            "classification_observed": int((ok & finite).sum()),
                            "input_eligible_with_missing_Ret_20d": int((ok & ~finite).sum()),
                            "finite_Ret_20d_below_minus_one": int((finite & (part["Ret_20d"].to_numpy() < -1)).sum())})
        pd.DataFrame(quality).to_csv(output / "test_data_audit.csv", index=False)
        finance_status = {"status": "computed", "definition": frozen["study_plan"]["financial_evaluation"]}
        monthly = pd.DataFrame()
        try:
            monthly, coverage = monthly_portfolios(frame)
            coverage.to_csv(output / "monthly_coverage.csv", index=False)
            monthly.to_csv(output / "monthly_portfolio_returns.csv", index=False)
            summary = portfolio_summary(monthly)
            summary.to_csv(output / "monthly_portfolio_summary.csv", index=False)
            comparison = literature_comparison(summary, read_json(HERE / "paper_reference.json"))
            comparison.to_csv(output / "paper_comparison_I20R20.csv", index=False)
            finance_status["included_formation_dates"] = int(monthly["Date"].nunique()) if not monthly.empty else 0
            finance_status["omitted_dates"] = int((coverage["status"] != "included").sum()) if not coverage.empty else 0
            if monthly.empty:
                finance_status["status"] = "not_computed_no_supported_months"
        except ValueError as error:
            finance_status.update({"status": "not_computed", "reason": str(error)})
        write_json(output / "financial_status.json", finance_status)
        try:
            plots = make_plots(output, classification, calibration, monthly)
        except ImportError as error:
            plots = {"status": f"Plots not generated; missing plotting dependency: {error}"}
        write_json(output / "plot_status.json", plots)
        key = classification.loc[(classification["period"] == "course_2000_2019") &
                                 classification["method"].isin([f"seed_{seed}" for seed in SEEDS] + ["ensemble", "ensemble_training_prior", "always_up", "always_nonup"])]
        display = markdown_table(["Method", "N", "CE", "Accuracy", "MCC", "AUC"],
            [[row.method, int(row.samples), "n/a" if pd.isna(row.cross_entropy) else f"{row.cross_entropy:.6f}",
              f"{100*row.accuracy:.2f}%", f"{row.mcc:.4f}", "n/a" if pd.isna(row.roc_auc) else f"{row.roc_auc:.4f}"]
             for row in key.itertuples()])
        report = f"""# I20/R20 three-layer baseline test evaluation

Five best checkpoints were loaded without training or weight changes. Each used its own saved training-only normalization. Predictions were generated for input-eligible rows regardless of future-return availability. Classification scores then use observed Ret_20d labels, with threshold >0.5 and ties assigned to non-up.

## Course period: formation years 2000-2019

{display}

The table contains individual members and the arithmetic-mean probability ensemble. Detailed metrics also include the predeclared common period 2001-2019 and each formation year. Training-prior baselines use saved training fractions only. Always-up/non-up rows are fixed classification rules; their probability loss is deliberately not reported.

## Monthly financial comparison

Status: {finance_status['status']}.

Monthly portfolios use Ret_month, not Ret_20d. They sort the ensemble signal into ten within-date percentile groups using observed-return and valid-capitalization complete cases, following the historical filtering convention of the reference implementation. Missing/invalid return exclusions are in monthly_coverage.csv; this selection can bias historical performance and is not an executable-strategy guarantee.

EW and VW use the same complete-case universe. VW weights use absolute formation-date capitalization. H-L is 100% long High and 100% short Low. Annualized arithmetic mean = 12 * monthly mean; volatility = sqrt(12) * sample standard deviation; Sharpe uses a zero risk-free assumption. Results are gross, without transaction costs, borrow fees or financing adjustments. Empty/tied groups and missing months are not filled with zero. Turnover, significance tests and net returns are not estimated.

Financial statistics are in monthly_portfolio_summary.csv when supported by the observed dates. This is an I20/R20 probability signal evaluated with calendar-month holding returns; do not relabel it as an exact fixed-20-trading-day wealth backtest. Our course training years differ from the supplied paper, so paper comparisons are descriptive rather than strict numerical reproduction claims.

paper_comparison_I20R20.csv pairs the produced EW/VW tables with the supplied paper's rounded I20/R20 Table 2 and Table IA3 values. Differences are descriptive, not a controlled causal model improvement estimate. Original H-L entries are preserved rather than recomputed from rounded printed High/Low entries.

## Frozen extensions

The study plan was recorded before test scoring: depths 2/3/4, five paired seeds, identical per-seed baseline data splits and normalization, unchanged training/selection rules, fixed equal ensemble weights. Grad-CAM uses the fixed seed-101 validation set and last-block pre-pool LeakyReLU outputs, explaining both classes on the same reference samples. These extensions are planned, not executed by this evaluation entry point. Test scores must not be used to change their architecture, threshold, seed inclusion or blending weights.

## Files

- classification_metrics.csv: course, common-period and per-year classification metrics.
- test_data_audit.csv: row and label-availability counts.
- calibration_bins.csv: fixed ten-bin reliability summaries.
- monthly_coverage.csv: monthly complete-case and omission audit.
- monthly_portfolio_returns.csv and monthly_portfolio_summary.csv: gross monthly comparison, if supported.
- paper_comparison_I20R20.csv: descriptive comparison with the supplied paper's EW/VW tables.
- financial_status.json and plot_status.json: any unsupported financial/plot outputs.
- The full frozen protocol and model/input fingerprints remain in the parent evaluation directory and per-year completion records.

No training results or original input files were modified.
"""
        (output / "test_report.md").write_text(report, encoding="utf-8")
        write_json(output / "analysis_complete.json", {"evaluation_id": frozen["evaluation_id"],
                   "completed_at": datetime.now().isoformat(), "original_rows": len(frame),
                   "classification_rows": int(sum(item["classification_observed"] for item in quality)),
                   "finance_status": finance_status["status"],
                   "source_prediction_sha256": {str(year): read_json(year_paths(directory, year)[1])["prediction_sha256"] for year in YEARS}})
        print(display, flush=True)
        print(f"ANALYZE: COMPLETE | finance={finance_status['status']} | {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze", help="record the study plan and verify existing checkpoints")
    freezing.add_argument("--train-dir", type=Path, required=True)
    freezing.add_argument("--data-dir", type=Path, required=True)
    freezing.add_argument("--eval-dir", type=Path, required=True)
    inference = commands.add_parser("predict", help="resume read-only GPU prediction by year")
    inference.add_argument("--eval-dir", type=Path, required=True)
    inference.add_argument("--years", type=int, nargs="+")
    analysis = commands.add_parser("analyze", help="score complete frozen predictions; no GPU required")
    analysis.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        {"freeze": freeze, "predict": predict, "analyze": analyze}[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted. Completed prediction years are retained; rerun predict to resume.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
