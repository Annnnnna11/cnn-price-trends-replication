"""Reconcile committed summaries and Grad-CAM arrays without loading Torch."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def review(root=REPO):
    folder = root / 'results/comparison'
    accuracy_error = mcc_error = financial_error = normalization_error = 0.0
    classification = pd.read_csv(folder / 'classification_all_members.csv')
    for row in classification.itertuples():
        tn, fp, fn, tp = [int(v) for v in (row.tn, row.fp, row.fn, row.tp)]
        check(tn + fp + fn + tp == row.samples, 'Confusion sample count mismatch')
        denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn-fp*fn)/denominator if denominator else 0.0
        accuracy_error = max(accuracy_error, abs((tn+tp)/row.samples-row.accuracy))
        mcc_error = max(mcc_error, abs(mcc-row.mcc))
    monthly = pd.read_csv(folder / 'monthly_returns_all_depths.csv')
    summary = pd.read_csv(folder / 'financial_all_deciles.csv')
    for row in summary.itertuples():
        start = 2000 if row.period == 'course_2000_2019' else 2001
        part = monthly.loc[monthly.depth.eq(row.depth) & monthly.weighting.eq(row.weighting)
                           & monthly.formation_year.between(start, 2019)]
        values = part[row.portfolio].to_numpy(dtype=np.float64)
        mu, sd = values.mean(), values.std(ddof=1)
        expected = (12*mu, math.sqrt(12)*sd, math.sqrt(12)*mu/sd)
        actual = (row.annualized_arithmetic_mean, row.annualized_volatility, row.sharpe_zero_rf_gross)
        financial_error = max(financial_error, *(abs(a-b) for a, b in zip(expected, actual)))
        check(len(values) == row.observed_months, 'Financial month count mismatch')
    train = pd.read_csv(folder / 'training_all_members.csv')
    check(len(train) == 15, 'Expected 15 trained members')
    for row in train.itertuples():
        history = pd.read_csv(folder / f'training_history/depth{row.depth}_seed{row.seed}.csv')
        best = history.loc[history.validation_cross_entropy.idxmin()]
        check(best.epoch == row.best_epoch and history.epoch.max() == row.epochs_run,
              'Best epoch/total epochs mismatch')
        check(abs(best.validation_cross_entropy-row.validation_ce) < 1e-12, 'Best CE mismatch')
    calendar = pd.read_csv(folder / 'financial_calendar_audit.csv')
    check(calendar.common_dates.eq(239).all() and calendar.excluded_from_matched.eq(0).all(),
          'Matched calendar differs from recorded comparison')
    cam = root / 'results/gradcam'
    metadata = pd.read_csv(cam / 'map_metadata.csv')
    samples = pd.read_csv(cam / 'selected_samples.csv')
    selection = read(root / 'records/followup/gradcam/selection.json')
    contract = read(root / 'records/followup/followup.json')
    completed = read(root / 'records/followup/gradcam/complete.json')
    check(completed['model_states_unchanged'], 'Saved CAM model-unchanged flag is false')
    check(len(samples) == 12 and len(metadata) == 72, 'CAM case/map count mismatch')
    check(samples.reference_group.value_counts().to_dict() == dict(TP=3,TN=3,FP=3,FN=3), 'CAM grouping mismatch')
    check(selection['sampling_seed'] == 20261118, 'CAM sampling seed mismatch')
    images = {}; zero_maps = 0
    for row in metadata.itertuples():
        checkpoint = next(m for m in contract['evaluations'][str(row.depth)]['members'] if m['seed'] == 101)
        check(row.seed == 101 and row.checkpoint_sha256 == checkpoint['files']['best.pt'], 'CAM metadata identity mismatch')
        with np.load(cam / row.map_file, allow_pickle=False) as arrays:
            image, native = arrays['input_uint8'], arrays['relu_native']
            signed, upsampled, normalized = arrays['signed_native'], arrays['relu_upsampled'], arrays['normalized']
            check(image.shape == (1,64,60) and native.shape == ({2:10,3:5,4:3}[row.depth],60), 'CAM geometry mismatch')
            check(np.array_equal(native, np.maximum(signed,0)), 'CAM ReLU mismatch')
            check(normalized.shape == (64,60) and np.isfinite(normalized).all(), 'CAM shape/finiteness mismatch')
            check((normalized >= 0).all() and (normalized <= 1).all(), 'CAM range mismatch')
            expected = upsampled/upsampled.max() if upsampled.max() > 0 else np.zeros_like(upsampled)
            normalization_error = max(normalization_error, float(np.max(np.abs(expected-normalized))))
            is_zero = not bool(normalized.any()); zero_maps += int(is_zero)
            check(is_zero == bool(row.zero_map), 'CAM zero-map flag mismatch')
            if row.CatalogIndex in images:
                check(np.array_equal(images[row.CatalogIndex], image), 'Different images in paired CAM comparison')
            else:
                images[row.CatalogIndex] = image.copy()
    for row in samples.itertuples():
        group = {(1,1):'TP',(0,0):'TN',(0,1):'FP',(1,0):'FN'}[int(row.Label),int(row.reference_probability > .5)]
        check(group == row.reference_group, 'CAM reference group inconsistent with prediction')
        part = metadata.loc[metadata.CatalogIndex.eq(row.CatalogIndex)]
        check(set(zip(part.depth,part.target_class)) == {(d,c) for d in (2,3,4) for c in (0,1)}, 'CAM pairs incomplete')
        check(abs(part.loc[part.depth.eq(3),'p_up'].iloc[0]-row.reference_probability) < 1e-5, 'CAM probability mismatch')
    check(max(accuracy_error,mcc_error,financial_error) < 1e-12 and normalization_error < 1e-6,
          'Numerical reconciliation tolerance exceeded')
    check('torch' not in sys.modules, 'Summary review unexpectedly loaded Torch')
    return {'status':'PASS', 'classification_summary_rows':len(classification),
            'max_accuracy_error':accuracy_error, 'max_mcc_error':mcc_error,
            'financial_summary_rows':len(summary), 'max_financial_error':financial_error,
            'training_histories_checked':len(train), 'gradcam_cases':len(samples),
            'gradcam_maps':len(metadata), 'zero_maps':zero_maps,
            'max_gradcam_normalization_error':normalization_error, 'torch_loaded':False,
            'scope':'Summary reconciliation and saved-array checks only; no retraining, checkpoint loading or per-image CE/AUC rescoring.'}


if __name__ == '__main__':
    result = review()
    output = REPO / 'runs/review'; output.mkdir(parents=True, exist_ok=True)
    (output/'summary_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
