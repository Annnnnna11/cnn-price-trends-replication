"""Continue saved predictions in fresh CPU/GPU processes. Standard library only."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STAGES = ('audit', 'analyze', 'compare', 'cam_compute', 'cam_plot', 'bundle')


def child_environment():
    env = dict(os.environ)
    # Do not suppress duplicate-runtime errors; keep each job's native CPU work sequential.
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               MKL_THREADING_LAYER='SEQUENTIAL', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return env


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}


def run_job(stage, package, root, job_dir):
    command = [sys.executable, '-u', str(HERE / 'worker.py'), stage,
               '--package', str(package), '--output', str(root)]
    print(f'=== RESUME STAGE: {stage} | fresh process ===', flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', env=child_environment())
    try:
        with (job_dir / f'{stage}.log').open('a', encoding='utf-8', buffering=1) as log:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line)
        result = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=15)
        raise
    if result:
        raise RuntimeError(f'Stage {stage} exited with code {result}. '
                           f'Keep every file. Send this log: {job_dir / (stage + ".log")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('i20r20_followup_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/i20r20_followup_v1'))
    parser.add_argument('--stage', choices=STAGES, help='Optional single stage; default runs all six in order')
    args = parser.parse_args()
    package, root = args.package.resolve(), args.output.resolve()
    if not (package / 'guards.py').is_file() or not (root / 'followup.json').is_file():
        parser.error('Existing package and results/followup.json are required. Do not create a new experiment.')
    job_dir = root / 'runtime_fix'; job_dir.mkdir(exist_ok=True)
    # Reuse the existing lock helper without importing numeric libraries into this launcher.
    # This launcher uses its own lock; child jobs use the original root experiment.lock.
    lock_stream = (job_dir / 'resume.lock').open('a+b')
    if lock_stream.seek(0, 2) == 0:
        lock_stream.write(b'0'); lock_stream.flush()
    lock_stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run = {'version': 'i20r20_resume_v1', 'started_at': datetime.now().isoformat(),
               'python': sys.executable, 'package': str(package), 'output': str(root),
               'source_sha256': hashes(), 'stages': [args.stage] if args.stage else list(STAGES),
               'child_environment': {key: child_environment()[key] for key in
                                     ('MKL_THREADING_LAYER','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
               'duplicate_runtime_suppression': False}
        record = job_dir / f'launch_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        record.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding='utf-8')
        for stage in run['stages']:
            run_job(stage, package, root, job_dir)
        if not args.stage:
            print(f'RESUME: ALL COMPLETE | {root / "i20r20_followup_results.zip"}', flush=True)
    finally:
        lock_stream.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Rerun the same resume command; saved predictions and maps are retained.', flush=True)
        sys.exit(130)
    except Exception as error:
        print(f'RESUME STOPPED: {error}', flush=True)
        sys.exit(1)
