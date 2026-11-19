"""Read-only identity checks, completion gates and frozen follow-up inputs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from common import architecture_spec
from evaluation_core import (SEEDS, YEARS, read_json, write_json, sha256_file,
                             paths_for_year, unchanged_source, source_record)

HERE = Path(__file__).resolve().parent


def identity(value, field):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != field},
                                    sort_keys=True).encode()).hexdigest()


def own_hashes():
    return {p.name: sha256_file(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.json') and p.name != 'VALIDATION.json'}


def check_files(folder, expected):
    for name, checksum in expected.items():
        path = Path(folder) / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f'Missing or changed artifact: {path}')


def completion_members(folder, manifest, depth):
    members = []
    for seed in SEEDS:
        sub = folder / f'seed_{seed}'
        if not (sub / 'completed.json').is_file():
            raise RuntimeError(f'Training NOT complete: depth {depth}, seed {seed}. '
                               'Wait for TRAIN: 10/10; this package will not start training.')
        done, norm = read_json(sub / 'completed.json'), read_json(sub / 'normalization.json')
        if done['run_id'] != manifest['run_id'] or done['seed'] != seed or done.get('depth', 3) != depth:
            raise ValueError(f'Completion identity mismatch: {sub}')
        check_files(sub, {'best.pt': done['checkpoint_hashes']['best.pt']})
        if not np.isfinite(norm['mean']) or not np.isfinite(norm['std']) or norm['std'] <= 0:
            raise ValueError(f'Invalid normalization: {sub}')
        members.append({'seed': seed, 'best_epoch': done['best_epoch'], 'normalization': norm,
                        'files': {name: sha256_file(sub / name) for name in
                                  ('best.pt', 'normalization.json', 'completed.json')}})
    return members


def verify_contract(root):
    record = read_json(root / 'followup.json')
    if record['followup_id'] != identity(record, 'followup_id') or record['code_hashes'] != own_hashes():
        raise ValueError('Follow-up code or frozen configuration changed; preserve this run and investigate.')
    for folder, files in record['files'].items():
        check_files(Path(folder), files)
    data = Path(record['data_dir'])
    for entry in record['training_inputs']:
        if not unchanged_source(data / entry['name'], entry):
            raise ValueError(f"Training input changed: {entry['name']}")
    return record


def verify_frozen(directory):
    frozen = read_json(directory / 'frozen_evaluation.json')
    contract = verify_contract(directory.parent)
    if frozen != contract['evaluations'][str(frozen['depth'])]:
        raise ValueError(f'Per-depth evaluation configuration changed: {directory}')
    return frozen


def prepare(args):
    base, ablation = args.baseline_dir.resolve(), args.ablation_root.resolve()
    baseline_eval, data, root = args.baseline_eval.resolve(), args.data_dir.resolve(), args.output.resolve()
    sources = (base, ablation, baseline_eval, data)
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in sources):
        raise ValueError('Use an independent output folder, separate from data, training and baseline evaluation.')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'followup.json').exists():
        record = verify_contract(root)
        wanted = dict(baseline_dir=str(base), ablation_root=str(ablation),
                      baseline_eval=str(baseline_eval), data_dir=str(data))
        if any(record[k] != v for k, v in wanted.items()):
            raise ValueError('Existing follow-up belongs to different input paths.')
        for depth in (2, 3, 4):
            directory = root / f'depth_{depth}'
            if not (directory / 'frozen_evaluation.json').exists():
                directory.mkdir(exist_ok=True)
                write_json(directory / 'frozen_evaluation.json', record['evaluations'][str(depth)])
            verify_frozen(directory)
        print('PREPARE: previously frozen inputs verified', flush=True)
        return
    inherited = read_json(HERE / 'inherited_hashes.json')
    plan = read_json(HERE / 'study_plan.json')
    baseline = read_json(base / 'manifest.json')
    ready = read_json(ablation / 'prepared.json')
    old = read_json(baseline_eval / 'frozen_evaluation.json')
    if baseline['run_id'] != plan['baseline_run_id'] or identity(baseline, 'run_id') != baseline['run_id']:
        raise ValueError('Baseline manifest identity mismatch')
    if baseline['code_hashes']['cnn20.py'] != sha256_file(HERE / 'cnn20.py'):
        raise ValueError('Baseline CNN implementation checksum differs')
    if ready['code_hashes'] != inherited['ablation'] or old['code_hashes'] != inherited['baseline_eval']:
        raise ValueError('Earlier package version differs from the reviewed version.')
    if old['study_plan'] != plan or ready['baseline_run_id'] != baseline['run_id']:
        raise ValueError('Study plan / ablation baseline mismatch')
    for depth in (2, 4):
        for seed in SEEDS:
            if not (ablation / f'depth_{depth}/seed_{seed}/completed.json').is_file():
                raise RuntimeError(f'Training NOT complete: depth {depth}, seed {seed}. Wait for all 10 members.')
    base_sha = sha256_file(base / 'manifest.json')
    if ready['baseline_manifest_sha256'] != base_sha or old['training_manifest_sha256'] != base_sha:
        raise ValueError('Baseline manifest checksum mismatch')
    if Path(ready['baseline_dir']).resolve() != base or Path(old['training_dir']).resolve() != base:
        raise ValueError('Earlier records point to a different baseline directory')
    if Path(ready['data_dir']).resolve() != data or Path(old['data_dir']).resolve() != data:
        raise ValueError('Earlier records point to different data')
    files = {str(base): {'manifest.json': base_sha, **baseline['artifact_hashes']},
             str(ablation): {'prepared.json': sha256_file(ablation / 'prepared.json')},
             str(baseline_eval): {'frozen_evaluation.json': sha256_file(baseline_eval / 'frozen_evaluation.json')}}
    training_inputs = []
    print(f'PREPARE: training completion records found; verifying {len(baseline["inputs"])} training input files...', flush=True)
    for entry in baseline['inputs']:
        path = data / entry['name']
        # Hash once when freezing, then stat/checksum guards on later phases.
        if sha256_file(path) != entry['sha256']:
            raise ValueError(f'Training data checksum mismatch: {path}')
        training_inputs.append(source_record(path))
    evaluations = {}
    for depth in (2, 3, 4):
        folder = base if depth == 3 else ablation / f'depth_{depth}'
        manifest = read_json(folder / 'manifest.json')
        if identity(manifest, 'run_id') != manifest['run_id']:
            raise ValueError('Training manifest contents changed')
        if depth != 3:
            if (manifest['depth'] != depth or manifest['run_id'] != ready['variants'][str(depth)]
                    or manifest['baseline_run_id'] != baseline['run_id']
                    or manifest['code_hashes'] != inherited['ablation']
                    or manifest['specification'] != architecture_spec(depth)):
                raise ValueError(f'Depth {depth}: model identity mismatch')
            expected_protocol = dict(baseline['protocol'])
            expected_protocol.update(version=f'i20r20_ablation_v1_depth{depth}', architecture_depth=depth)
            if manifest['protocol'] != expected_protocol or manifest['inputs'] != baseline['inputs']:
                raise ValueError('Paired training protocol or input list changed')
            for name, checksum in baseline['artifact_hashes'].items():
                if manifest['artifact_hashes'].get(name) != checksum:
                    raise ValueError(f'Paired split, normalization or catalog differs: {name}')
        check_files(folder, manifest['artifact_hashes'])
        members = completion_members(folder, manifest, depth)
        if depth == 3 and members != old['members']:
            raise ValueError('Baseline members differ from the completed baseline evaluation')
        snapshot = {'manifest.json': sha256_file(folder / 'manifest.json'), **manifest['artifact_hashes']}
        for member in members:
            seed = member['seed']
            for name in (*member['files'], 'history.csv'):
                snapshot[f'seed_{seed}/{name}'] = sha256_file(folder / f'seed_{seed}' / name)
        files[str(folder)] = {**files.get(str(folder), {}), **snapshot}
        prediction_folder = baseline_eval / 'predictions' if depth == 3 else root / f'depth_{depth}/predictions'
        frozen = {'version': 'i20r20_followup_v1', 'depth': depth, 'run_id': manifest['run_id'],
                  'baseline_run_id': baseline['run_id'], 'training_dir': str(folder),
                  'data_dir': str(data), 'prediction_directory': str(prediction_folder),
                  'members': members, 'study_plan': plan}
        frozen['evaluation_id'] = old['evaluation_id'] if depth == 3 else identity(frozen, 'evaluation_id')
        evaluations[str(depth)] = frozen
    # Baseline predictions remain read-only and must all be complete before the queue starts.
    for year in YEARS:
        pred = baseline_eval / f'predictions/{year}.feather'
        marker = baseline_eval / f'predictions/{year}.complete.json'
        record = read_json(marker)
        if record['evaluation_id'] != old['evaluation_id'] or record['year'] != year:
            raise ValueError(f'Baseline year {year}: wrong prediction identity')
        if sha256_file(pred) != record['prediction_sha256']:
            raise ValueError(f'Baseline year {year}: prediction checksum mismatch')
        for path, key in zip(paths_for_year(data, year), ('image_source', 'label_source')):
            if not unchanged_source(path, record[key]):
                raise ValueError(f'Baseline prediction input changed: {year}')
        files[str(baseline_eval)][str(marker.relative_to(baseline_eval))] = sha256_file(marker)
    record = {'version': 'i20r20_followup_v1', 'created_at': datetime.now().isoformat(),
              'baseline_dir': str(base), 'ablation_root': str(ablation), 'baseline_eval': str(baseline_eval),
              'data_dir': str(data), 'code_hashes': own_hashes(), 'files': files,
              'training_inputs': training_inputs, 'evaluations': evaluations}
    record['followup_id'] = identity(record, 'followup_id')
    # Do not overwrite an unrelated or partially populated result directory.
    if any(p.name not in ('experiment.lock', 'followup.log') for p in root.iterdir()):
        raise ValueError('Output is not empty and has no followup.json; choose a fresh output folder.')
    # Commit the authoritative record first; missing per-depth views can be restored on resume.
    write_json(root / 'followup.json', record)
    for depth, frozen in evaluations.items():
        folder = root / f'depth_{depth}'
        folder.mkdir(exist_ok=True)
        write_json(folder / 'frozen_evaluation.json', frozen)
    print('PREPARE: PASS | 15/15 completed models verified; baseline predictions reused; no training', flush=True)
