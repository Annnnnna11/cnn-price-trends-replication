"""One isolated stage per invocation; never calls train or test prediction."""
import argparse
from pathlib import Path
import sys


def audit(root):
    from guards import verify_contract, verify_frozen
    from engine import verify_year
    from evaluation_core import write_json
    contract = verify_contract(root)
    counts, missing = {}, []
    for depth in (2, 3, 4):
        frozen = verify_frozen(root / f'depth_{depth}')
        count = 0
        for year in range(2000, 2020):
            if verify_year(root / f'depth_{depth}', frozen, year):
                count += 1
            else:
                missing.append(f'depth{depth}/{year}')
        counts[str(depth)] = count
        print(f'AUDIT depth {depth}: {count}/20 prediction years verified', flush=True)
    if missing:
        raise RuntimeError(f'Missing prediction years: {missing}. This recovery package does not rerun prediction.')
    write_json(root / 'runtime_fix/audit.json', {'followup_id': contract['followup_id'],
        'verified_years': counts, 'torch_loaded': 'torch' in sys.modules,
        'source_models_and_frozen_protocol_unchanged': True})
    if 'torch' in sys.modules:
        raise RuntimeError('CPU audit unexpectedly imported Torch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit','analyze','compare','cam_compute','cam_plot','bundle'])
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package, root = args.package.resolve(), args.output.resolve()
    sys.path.insert(0, str(package))
    from common import experiment_lock
    from guards import verify_contract
    # Fresh workers only load the libraries needed by the selected stage.
    if args.stage == 'audit':
        with experiment_lock(root): audit(root)
    elif args.stage in ('analyze', 'compare', 'bundle'):
        import run as original
        with experiment_lock(root):
            verify_contract(root)
            if args.stage == 'analyze': original.analysis(root)
            elif args.stage == 'compare':
                from comparison import compare
                compare(root)
            else:
                original.bundle(root)
                add_runtime_records(root)
    elif args.stage == 'cam_compute':
        from compute_cam import run
        with experiment_lock(root): run(root)
    else:
        from render_cam import run
        with experiment_lock(root): run(root)
    if args.stage != 'cam_compute' and 'torch' in sys.modules:
        raise RuntimeError('CPU worker unexpectedly imported Torch')
    print(f'WORKER {args.stage}: COMPLETE | torch_loaded={"torch" in sys.modules}', flush=True)


def add_runtime_records(root):
    import zipfile
    from evaluation_core import read_json, write_json, sha256_file
    destination = root / 'i20r20_followup_results.zip'
    manifest_path = root / 'handoff_manifest.json'
    manifest = read_json(manifest_path)
    records = [p for p in (root / 'runtime_fix').glob('*.json') if p.is_file()]
    for path in records:
        manifest['files'][str(path.relative_to(root))] = sha256_file(path)
    write_json(manifest_path, manifest)
    # Rebuild rather than append duplicate manifest entries to an existing ZIP.
    temp = destination.with_suffix('.zip.runtime.tmp')
    with zipfile.ZipFile(destination) as old, zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as new:
        manifest_name = f'{root.name}/handoff_manifest.json'
        for entry in old.infolist():
            if entry.filename != manifest_name:
                new.writestr(entry, old.read(entry.filename))
        for path in records:
            new.write(path, f'{root.name}/{path.relative_to(root).as_posix()}')
        new.write(manifest_path, manifest_name)
    temp.replace(destination)


if __name__ == '__main__':
    main()
