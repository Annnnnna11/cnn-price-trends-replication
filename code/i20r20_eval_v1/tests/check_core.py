"""Synthetic invariants for classification and monthly financial evaluation."""

from pathlib import Path
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_core import (SEEDS, probability_metrics, prepare_labels,
                             classification_tables, calibration_table,
                             monthly_portfolios, portfolio_summary)


result = probability_metrics(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8]))
assert result["accuracy"] == .75
assert result["roc_auc"] == .75
assert np.isclose(result["brier"], .158125)
assert np.isclose(result["mcc"], 2 / math.sqrt(12))
ties = probability_metrics([0, 0, 1, 1], [.5, .5, .5, .5])
assert ties["accuracy"] == .5 and ties["roc_auc"] == .5
assert ties["predicted_up_fraction"] == 0
assert np.isclose(ties["cross_entropy"], math.log(2))
one_class = probability_metrics([0, 0], [.2, .3])
assert one_class["roc_auc"] is None and one_class["mcc"] == 0
assert one_class["balanced_accuracy"] is None
for bad in ([.2, float("nan")], [-.1, .2], [.2, 1.1]):
    try:
        probability_metrics([0, 1], bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid probabilities must be rejected")
print("Classification scores, ties, degenerate labels and probability guards: PASS")


def fixture(month):
    raw = pd.DataFrame({"Date": [pd.Timestamp(2001, month, 1) + pd.offsets.MonthEnd(0)] * 20,
                        "StockID": np.arange(20), "MarketCap": np.arange(1, 21),
                        "Ret_20d": np.r_[np.full(10, -.1), np.full(10, .1)],
                        "Ret_month": np.arange(20) / 100 - .1 + .01 * (month - 1),
                        "EWMA_vol": np.ones(20)})
    prepared = prepare_labels(raw, 2001)
    p = np.linspace(.01, .99, 20)
    for seed in SEEDS:
        prepared[f"p_{seed}"] = p
    prepared["p_ensemble"] = p
    return prepared


frame = fixture(1)
frame.loc[0, "Ret_20d"] = np.nan
assert frame["eligible_input"].all()
classification = classification_tables(frame, {seed: .49 for seed in SEEDS})
row = classification.loc[(classification["period"] == "course_2000_2019") & (classification["method"] == "ensemble")].iloc[0]
assert row["samples"] == 19 and row["accuracy"] == 1
assert classification.loc[classification["method"] == "always_up", "cross_entropy"].isna().all()
calibration = calibration_table(frame)
assert calibration.loc[calibration["method"] == "p_ensemble", "samples"].sum() == 19
monthly, audit = monthly_portfolios(frame)
assert audit.iloc[0]["complete_cases"] == 20
assert np.isclose(monthly.loc[monthly["weighting"] == "EW", "D1"].iloc[0], -.095)
assert np.isclose(monthly.loc[monthly["weighting"] == "EW", "D10"].iloc[0], .085)
assert np.isclose(monthly.loc[monthly["weighting"] == "EW", "H-L"].iloc[0], .18)
assert np.isclose(monthly.loc[monthly["weighting"] == "VW", "D1"].iloc[0], -.28 / 3)
print("Future Ret_20d absence does not block inference/monthly comparison; EW/VW and H-L: PASS")

missing = fixture(1)
missing.loc[0, "Ret_month"] = np.nan
_, coverage = monthly_portfolios(missing)
assert coverage.iloc[0]["unobserved_or_invalid_Ret_month"] == 1
assert coverage.iloc[0]["complete_cases"] == 19
constant = fixture(1)
constant["p_ensemble"] = .5
empty, coverage = monthly_portfolios(constant)
assert empty.empty and coverage.iloc[0]["status"] == "omitted_empty_or_tied_deciles"
multiple_dates = pd.concat([fixture(1), fixture(1).assign(Date=pd.Timestamp("2001-01-30"))])
try:
    monthly_portfolios(multiple_dates)
except ValueError:
    pass
else:
    raise AssertionError("Multiple dates per month must not be annualized as monthly returns")
print("Missing-return coverage, tied-decile omission and formation-frequency guard: PASS")

monthly, _ = monthly_portfolios(pd.concat([fixture(1), fixture(2)], ignore_index=True))
summary = portfolio_summary(monthly)
low = summary.loc[(summary["period"] == "course_2000_2019") &
                  (summary["weighting"] == "EW") & (summary["portfolio"] == "D1")].iloc[0]
assert np.isclose(low["annualized_arithmetic_mean"], -1.08)
assert np.isclose(low["annualized_volatility"], np.sqrt(12) * np.std([-.095, -.085], ddof=1))
spread = summary.loc[(summary["period"] == "course_2000_2019") &
                     (summary["weighting"] == "EW") & (summary["portfolio"] == "H-L")].iloc[0]
assert pd.isna(spread["sharpe_zero_rf_gross"])
with_gap, _ = monthly_portfolios(pd.concat([fixture(1), fixture(3)], ignore_index=True))
assert (portfolio_summary(with_gap)["internal_missing_months"] == 1).all()
print("Arithmetic annualization, sample standard deviation, zero-volatility and missing-month handling: PASS")

for invalid in (fixture(1).drop(columns="StockID"), fixture(1).assign(Date=pd.Timestamp("2002-01-31"))):
    try:
        prepare_labels(invalid, 2001)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing schema/wrong year must be rejected")
print("All statistical/data checks passed. No real test labels or CUDA models were evaluated.")
