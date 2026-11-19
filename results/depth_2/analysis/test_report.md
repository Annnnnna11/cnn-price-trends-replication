# I20/R20 depth-2 test evaluation

Five best checkpoints were loaded without training or weight changes. Each used its own saved training-only normalization. Predictions were generated for input-eligible rows regardless of future-return availability. Classification scores then use observed Ret_20d labels, with threshold >0.5 and ties assigned to non-up.

## Course period: formation years 2000-2019

| Method | N | CE | Accuracy | MCC | AUC |
| --- | --- | --- | --- | --- | --- |
| seed_101 | 1490937 | 0.693464 | 52.08% | 0.0388 | 0.5274 |
| seed_102 | 1490937 | 0.691395 | 52.81% | 0.0483 | 0.5334 |
| seed_103 | 1490937 | 0.696122 | 51.81% | 0.0303 | 0.5216 |
| seed_104 | 1490937 | 0.697075 | 51.31% | 0.0303 | 0.5221 |
| seed_105 | 1490937 | 0.692474 | 52.34% | 0.0369 | 0.5269 |
| ensemble | 1490937 | 0.691202 | 52.66% | 0.0474 | 0.5336 |
| ensemble_training_prior | 1490937 | 0.693365 | 48.09% | 0.0000 | 0.5000 |
| always_up | 1490937 | n/a | 51.91% | 0.0000 | n/a |
| always_nonup | 1490937 | n/a | 48.09% | 0.0000 | n/a |

The table contains individual members and the arithmetic-mean probability ensemble. Detailed metrics also include the predeclared common period 2001-2019 and each formation year. Training-prior baselines use saved training fractions only. Always-up/non-up rows are fixed classification rules; their probability loss is deliberately not reported.

## Monthly financial comparison

Status: computed.

Monthly portfolios use Ret_month, not Ret_20d. They sort the ensemble signal into ten within-date percentile groups using observed-return and valid-capitalization complete cases, following the historical filtering convention of the reference implementation. Missing/invalid return exclusions are in monthly_coverage.csv; this selection can bias historical performance and is not an executable-strategy guarantee.

EW and VW use the same complete-case universe. VW weights use absolute formation-date capitalization. H-L is 100% long High and 100% short Low. Annualized arithmetic mean = 12 * monthly mean; volatility = sqrt(12) * sample standard deviation; Sharpe uses a zero risk-free assumption. Results are gross, without transaction costs, borrow fees or financing adjustments. Empty/tied groups and missing months are not filled with zero. Turnover, significance tests and net returns are not estimated.

Financial statistics are in monthly_portfolio_summary.csv when supported by the observed dates. This is an I20/R20 probability signal evaluated with calendar-month holding returns; do not relabel it as an exact fixed-20-trading-day wealth backtest. Our course training years differ from the supplied paper, so paper comparisons are descriptive rather than strict numerical reproduction claims.

paper_comparison_I20R20.csv pairs the produced EW/VW tables with the supplied paper's rounded I20/R20 Table 2 and Table IA3 values. Differences are descriptive, not a controlled causal model improvement estimate. Original H-L entries are preserved rather than recomputed from rounded printed High/Low entries.

## Frozen extensions

The study plan was recorded before test scoring: depths 2/3/4, five paired seeds, identical per-seed baseline data splits and normalization, unchanged training/selection rules, fixed equal ensemble weights. Grad-CAM uses the fixed seed-101 validation set and last-block pre-pool LeakyReLU outputs, explaining both classes on the same reference samples. This directory evaluates one structure; the follow-up runner separately generates the cross-depth comparison and Grad-CAM. Test scores must not be used to change their architecture, threshold, seed inclusion or blending weights.

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
