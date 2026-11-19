"""Pure data/statistical functions for the frozen I20/R20 evaluation."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (101, 102, 103, 104, 105)
YEARS = tuple(range(2000, 2020))
PERIODS = (("course_2000_2019", 2000, 2019), ("common_2001_2019", 2001, 2019))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def paths_for_year(data_dir, year):
    prefix = f"20d_month_has_vb_[20]_ma_{year}"
    return (Path(data_dir) / f"{prefix}_images.dat", Path(data_dir) / f"{prefix}_labels_w_delay.feather")


def source_record(path):
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path)}


def unchanged_source(path, record):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    return path.stat().st_mtime_ns == record["mtime_ns"] or sha256_file(path) == record["sha256"]


def prepare_labels(frame, year):
    """Keep every original row, including rows with unobserved future returns."""
    required = {"Date", "StockID", "MarketCap", "Ret_20d", "Ret_month", "EWMA_vol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{year}: missing columns {sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame["Date"], errors="raise")
    if dates.isna().any() or not (dates.dt.year == year).all():
        raise ValueError(f"{year}: missing/out-of-year Date")
    if frame["StockID"].isna().any():
        raise ValueError(f"{year}: missing StockID")
    result = pd.DataFrame({"Year": np.full(len(frame), year, dtype=np.int16),
                           "Row": np.arange(len(frame), dtype=np.int32),
                           "Date": dates.to_numpy(), "StockID": frame["StockID"].astype(str).to_numpy()})
    for name in ("MarketCap", "Ret_20d", "Ret_month", "EWMA_vol"):
        result[name] = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    if result.duplicated(["Date", "StockID"]).any():
        raise ValueError(f"{year}: duplicate Date/StockID")
    result["eligible_input"] = np.isfinite(result["EWMA_vol"]) & (result["EWMA_vol"] > 0)
    return result


def probability_metrics(labels, probability, epsilon=1e-7):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != probability.shape or len(labels) == 0:
        raise ValueError("Labels and probabilities must be nonempty equal-length vectors")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(probability).all():
        raise ValueError("Invalid binary label or non-finite probability")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probability outside [0,1]")
    predicted = (probability > .5).astype(np.int64)
    tn, fp, fn, tp = np.bincount(labels * 2 + predicted, minlength=4).tolist()
    positives, negatives = tp + fn, tn + fp
    denominator = math.sqrt((tp + fp) * positives * negatives * (tn + fn))
    safe = np.clip(probability, epsilon, 1 - epsilon)
    cross_entropy = -np.mean(labels * np.log(safe) + (1 - labels) * np.log1p(-safe))
    if positives and negatives:
        ranks = pd.Series(probability).rank(method="average").to_numpy()
        auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
        balanced = .5 * (tp / positives + tn / negatives)
    else:
        auc = balanced = None
    return {"samples": len(labels), "positive_fraction": positives / len(labels),
            "cross_entropy": float(cross_entropy), "accuracy": (tp + tn) / len(labels),
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
            "balanced_accuracy": balanced, "roc_auc": auc,
            "brier": float(np.mean((probability - labels) ** 2)),
            "predicted_up_fraction": (tp + fp) / len(labels), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def classification_tables(frame, priors, epsilon=1e-7):
    valid = frame["eligible_input"].to_numpy(dtype=bool) & np.isfinite(frame["Ret_20d"].to_numpy())
    valid &= np.isfinite(frame["p_ensemble"].to_numpy())
    subset = frame.loc[valid]
    rows = []
    selections = [(name, subset["Year"].between(start, end).to_numpy()) for name, start, end in PERIODS]
    selections += [(f"year_{year}", (subset["Year"] == year).to_numpy()) for year in YEARS]
    ensemble_prior = float(np.mean(list(priors.values())))
    for period, mask in selections:
        selected = subset.loc[mask]
        if selected.empty:
            continue
        labels = (selected["Ret_20d"].to_numpy() > 0).astype(np.int64)
        for seed in SEEDS:
            rows.append({"period": period, "method": f"seed_{seed}",
                         **probability_metrics(labels, selected[f"p_{seed}"].to_numpy(), epsilon)})
            rows.append({"period": period, "method": f"training_prior_{seed}",
                         **probability_metrics(labels, np.full(len(labels), priors[seed]), epsilon)})
        rows.append({"period": period, "method": "ensemble",
                     **probability_metrics(labels, selected["p_ensemble"].to_numpy(), epsilon)})
        rows.append({"period": period, "method": "ensemble_training_prior",
                     **probability_metrics(labels, np.full(len(labels), ensemble_prior), epsilon)})
        for name, value in (("always_up", 1), ("always_nonup", 0)):
            result = probability_metrics(labels, np.full(len(labels), value), epsilon)
            # Deterministic class rules are not presented as calibrated probabilities.
            result.update({"cross_entropy": None, "brier": None, "roc_auc": None})
            rows.append({"period": period, "method": name, **result})
    return pd.DataFrame(rows)


def calibration_table(frame):
    valid = frame["eligible_input"].to_numpy(dtype=bool) & np.isfinite(frame["Ret_20d"].to_numpy())
    valid &= np.isfinite(frame["p_ensemble"].to_numpy())
    selected = frame.loc[valid]
    rows = []
    for method in [f"p_{seed}" for seed in SEEDS] + ["p_ensemble"]:
        p = selected[method].to_numpy(dtype=np.float64)
        y = (selected["Ret_20d"].to_numpy() > 0).astype(int)
        bins = np.minimum((p * 10).astype(int), 9)
        for number in range(10):
            mask = bins == number
            rows.append({"method": method, "bin_left": number / 10, "bin_right": (number + 1) / 10,
                         "samples": int(mask.sum()), "mean_probability": float(p[mask].mean()) if mask.any() else None,
                         "observed_up_fraction": float(y[mask].mean()) if mask.any() else None})
    return pd.DataFrame(rows)


def monthly_portfolios(frame):
    """Historical complete-case reference comparison, not a missing-return imputation."""
    eligible = frame.loc[frame["eligible_input"] & np.isfinite(frame["p_ensemble"])].copy()
    eligible["formation_month"] = eligible["Date"].dt.to_period("M")
    if (eligible.groupby("formation_month")["Date"].nunique() > 1).any():
        raise ValueError("Multiple formation dates in a calendar month: monthly annualization unsupported")
    rows, audits = [], []
    for date, group in eligible.groupby("Date", sort=True):
        returns = group["Ret_month"].to_numpy(dtype=np.float64)
        cap = np.abs(group["MarketCap"].to_numpy(dtype=np.float64))
        good_return = np.isfinite(returns) & (returns >= -1)
        good_cap = np.isfinite(cap) & (cap > 0)
        complete = good_return & good_cap
        audit = {"Date": date, "eligible_inputs": len(group),
                 "unobserved_or_invalid_Ret_month": int((~good_return).sum()),
                 "invalid_market_cap": int((~good_cap).sum()), "complete_cases": int(complete.sum()),
                 "status": "included"}
        if complete.sum() < 10:
            audit["status"] = "omitted_fewer_than_10_complete_cases"
            audits.append(audit)
            continue
        subset = group.loc[complete]
        p = subset["p_ensemble"].to_numpy(dtype=np.float64)
        cuts = np.percentile(p, np.linspace(0, 100, 11))
        decile = np.searchsorted(cuts[1:-1], p, side="left") + 1
        counts = np.bincount(decile, minlength=11)[1:]
        if cuts[1] == cuts[9] or (counts == 0).any():
            audit["status"] = "omitted_empty_or_tied_deciles"
            audits.append(audit)
            continue
        r = subset["Ret_month"].to_numpy(dtype=np.float64)
        c = np.abs(subset["MarketCap"].to_numpy(dtype=np.float64))
        for weighting in ("EW", "VW"):
            values = []
            for number in range(1, 11):
                member = decile == number
                weights = np.ones(int(member.sum())) if weighting == "EW" else c[member]
                values.append(float(np.dot(weights / weights.sum(), r[member])))
            record = {"Date": date, "ReturnMonth": (date.to_period("M") + 1).to_timestamp(how="end").normalize(),
                      "formation_year": date.year, "weighting": weighting,
                      **{f"D{number}": value for number, value in enumerate(values, 1)},
                      "H-L": values[-1] - values[0], "complete_case_stocks": len(subset)}
            rows.append(record)
        audits.append(audit)
    return pd.DataFrame(rows), pd.DataFrame(audits)


def portfolio_summary(monthly):
    rows = []
    if monthly.empty:
        return pd.DataFrame()
    for period, start, end in PERIODS:
        selection = monthly.loc[monthly["formation_year"].between(start, end)]
        for weighting, group in selection.groupby("weighting", sort=True):
            for name in [f"D{number}" for number in range(1, 11)] + ["H-L"]:
                values = group[name].to_numpy(dtype=np.float64)
                avg = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else None
                volatility = math.sqrt(12) * std if std is not None else None
                sharpe = math.sqrt(12) * avg / std if std is not None and std > 1e-12 else None
                months = pd.PeriodIndex(group["Date"], freq="M")
                expected = pd.period_range(months.min(), months.max(), freq="M")
                rows.append({"period": period, "weighting": weighting, "portfolio": name,
                             "observed_months": len(values), "internal_missing_months": len(expected.difference(months)),
                             "annualized_arithmetic_mean": 12 * avg, "annualized_volatility": volatility,
                             "sharpe_zero_rf_gross": sharpe})
    return pd.DataFrame(rows)


def literature_comparison(summary, reference):
    if summary.empty:
        return pd.DataFrame()
    records = []
    for weighting in ("EW", "VW"):
        for index, name in enumerate(reference["portfolios"]):
            records.append({"weighting": weighting, "portfolio": name,
                            "paper_annualized_return": reference[weighting]["annualized_return"][index],
                            "paper_sharpe": reference[weighting]["sharpe"][index],
                            "paper_source_table": reference[weighting]["table"]})
    combined = summary.merge(pd.DataFrame(records), on=["weighting", "portfolio"], how="left", validate="many_to_one")
    combined["annualized_return_minus_paper"] = combined["annualized_arithmetic_mean"] - combined["paper_annualized_return"]
    combined["sharpe_minus_paper"] = combined["sharpe_zero_rf_gross"] - combined["paper_sharpe"]
    combined["comparison_type"] = "Descriptive literature comparison; training years/label conventions differ; paper values rounded"
    return combined


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)])
