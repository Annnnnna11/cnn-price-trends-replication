"""One queue: check -> depth2/4 test -> three-depth comparison -> Grad-CAM."""
import argparse
from datetime import datetime
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
import pandas as pd
from evaluation_core import read_json, write_json, sha256_file, paths_for_year
from guards import prepare, verify_contract, verify_frozen
from common import experiment_lock, architecture_spec
import engine


def check(root):
    import torch
    from cnn20 import CNN20
    from gradcam import configure, load_model, activation_cam, state_fingerprint, verify_cam_math
    contract = verify_contract(root)
    configure(torch); verify_cam_math(torch)
    frozen = {d: verify_frozen(root / f'depth_{d}') for d in (2, 3, 4)}
    base = Path(contract['baseline_dir'])
    catalog = pd.read_feather(base / 'in_sample_catalog.feather')
    with np.load(base / 'seed_101/split.npz', allow_pickle=False) as split:
        row = catalog.iloc[int(np.min(split['validation']))]
    year = int(row.Year)
    counts = read_json(base / 'manifest.json')['original_counts']
    image, _ = paths_for_year(contract['data_dir'], year)
    mapped = np.memmap(image, mode='r', dtype=np.uint8, shape=(counts[str(year)], 1, 64, 60))
    raw = torch.from_numpy(mapped[int(row.Row)].copy()).unsqueeze(0).to('cuda', dtype=torch.float32)/255.
    records = []
    for depth in (2, 3, 4):
        for seed in (101, 102, 103, 104, 105):
            model, norm = load_model(frozen[depth], seed)
            before = state_fingerprint(model)
            x = (raw-norm['mean'])/norm['std']
            with torch.inference_mode():
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite model outputs')
            difference = None
            if depth == 3 and seed == 101:
                original = CNN20().to('cuda').eval()
                original.load_state_dict(model.state_dict(), strict=True)
                with torch.inference_mode():
                    difference = float((original(x)-logits).abs().max().cpu())
                if difference != 0:
                    raise ValueError('Depth-3 factory differs from original model')
                del original
            if seed == 101:
                activation = []
                handle = model.conv_layers[depth-1][2].register_forward_hook(lambda m, args, out: activation.append(out))
                try:
                    with torch.enable_grad():
                        logits = model(x.detach().requires_grad_(True))
                        for target in (0, 1):
                            _, _, _, normalized, _ = activation_cam(logits, activation[0], target)
                            if tuple(normalized.shape) != (1, 1, 64, 60):
                                raise ValueError('Wrong Grad-CAM image size')
                finally:
                    handle.remove()
            if before != state_fingerprint(model):
                raise ValueError('Model state changed during check')
            records.append({'depth': depth, 'seed': seed, 'forward_finite': True,
                            'gradcam_checked': seed == 101, 'baseline_factory_difference': difference})
            print(f'CHECK depth {depth}, seed {seed}: PASS', flush=True)
            del model, logits
    write_json(root / 'gpu_check.json', {'followup_id': contract['followup_id'], 'at': datetime.now().isoformat(),
        'gpu': torch.cuda.get_device_name(0), 'python': sys.executable, 'cuda_runtime': torch.version.cuda,
        'packages': {name: importlib.metadata.version(name) for name in ('torch','numpy','pandas','pyarrow','matplotlib')},
        'members': records, 'synthetic_gradient_math': 'PASS', 'model_states_unchanged': True})
    print('GPU CHECK: PASS | all 15 checkpoints; 3 target layers; no training', flush=True)


def prediction(root, years=None):
    contract = verify_contract(root)
    checked = read_json(root / 'gpu_check.json')
    if checked['followup_id'] != contract['followup_id']:
        raise ValueError('GPU check belongs to another run')
    for depth in (2, 4):
        print(f'=== PREDICT DEPTH {depth} ===', flush=True)
        engine.predict(SimpleNamespace(eval_dir=root / f'depth_{depth}', years=years))


def analysis(root):
    for depth in (2, 3, 4):
        print(f'=== ANALYZE DEPTH {depth} ===', flush=True)
        engine.analyze(SimpleNamespace(eval_dir=root / f'depth_{depth}'))


def bundle(root):
    """Share analysis/attributions, not checkpoints, prediction caches or raw data."""
    contract = verify_contract(root)
    if not (root / 'comparison/complete.json').is_file() or not (root / 'gradcam/complete.json').is_file():
        raise RuntimeError('Comparison and Grad-CAM must complete before the final handoff bundle')
    for phase in ('comparison', 'gradcam'):
        if read_json(root / phase / 'complete.json')['followup_id'] != contract['followup_id']:
            raise ValueError(f'{phase} output belongs to a different run')
    from guards import check_files
    check_files(root / 'gradcam', read_json(root / 'gradcam/complete.json')['files'])
    files = [root / 'followup.json', root / 'gpu_check.json']
    files.extend(root / f'depth_{d}/frozen_evaluation.json' for d in (2, 3, 4))
    for depth in (2, 3, 4):
        files.extend(p for p in (root / f'depth_{depth}/analysis').rglob('*') if p.is_file())
    files.extend(p for p in (root / 'comparison').rglob('*') if p.is_file())
    files.extend(p for p in (root / 'gradcam').rglob('*') if p.is_file()
                 and not p.name.startswith('reference_validation.') and not p.name.endswith('.tmp'))
    for directory in (root / f'depth_{d}' for d in (2, 4)):
        files.extend(directory.glob('predictions/*.complete.json'))
    manifest = {str(p.relative_to(root)): sha256_file(p) for p in files}
    write_json(root / 'handoff_manifest.json', {'files': manifest, 'not_included':
               ['model checkpoints', 'raw data', 'yearly prediction Feather', 'full validation prediction cache'],
               'note': 'Full cache files referenced by Grad-CAM complete.json stay on Windows; selected cases and maps are included.'})
    files.append(root / 'handoff_manifest.json')
    destination = root / 'i20r20_followup_results.zip'
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f'{root.name}/{path.relative_to(root).as_posix()}')
    temporary.replace(destination)
    print(f'BUNDLE: SAVED | {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all','prepare','check','predict','analyze','compare','gradcam','bundle','status'])
    parser.add_argument('--baseline-dir', type=Path, default=Path('results/i20r20_main_v1'))
    parser.add_argument('--ablation-root', type=Path, default=Path('results/i20r20_depth_ablation_v1'))
    parser.add_argument('--baseline-eval', type=Path, default=Path('results/i20r20_baseline_eval_v1'))
    parser.add_argument('--data-dir', type=Path, default=Path('img_data/monthly_20d'))
    parser.add_argument('--output', type=Path, default=Path('results/i20r20_followup_v1'))
    parser.add_argument('--years', type=int, nargs='+', help='Only for predict; all always completes 2000-2019')
    args = parser.parse_args()
    if args.years is not None and args.command != 'predict':
        parser.error('--years is only allowed with predict')
    root = args.output.resolve()
    for source in (args.baseline_dir, args.ablation_root, args.baseline_eval, args.data_dir):
        source = source.resolve()
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            parser.error('Output must be independent of all input folders')
    root.mkdir(parents=True, exist_ok=True)
    with experiment_lock(root), engine.logger(root / 'followup.log'):
        if args.command in ('all','prepare'):
            prepare(args)
        if args.command == 'status':
            contract = verify_contract(root)
            for depth in (2, 3, 4):
                frozen = verify_frozen(root / f'depth_{depth}')
                n = sum(engine.verify_year(root / f'depth_{depth}', frozen, y) for y in range(2000, 2020))
                print(f'Depth {depth}: {n}/20 verified prediction years', flush=True)
            print('Comparison complete:', (root / 'comparison/complete.json').exists(), flush=True)
            print('Grad-CAM complete:', (root / 'gradcam/complete.json').exists(), flush=True)
            return
        if args.command in ('all','check'):
            check(root)
        if args.command in ('all','predict'):
            prediction(root, args.years)
        if args.command in ('all','analyze'):
            analysis(root)
        if args.command in ('all','compare'):
            from comparison import compare
            compare(root)
        if args.command in ('all','gradcam'):
            from gradcam import run
            run(root)
        if args.command in ('all','bundle'):
            bundle(root)
        if args.command == 'all':
            print('FOLLOWUP: ALL COMPLETE | no models were trained or modified', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Keep all files and rerun the same command; completed prediction years are verified and skipped.', flush=True)
        sys.exit(130)
