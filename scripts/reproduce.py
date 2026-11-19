"""Run a new I20/R20 experiment without altering the recorded experiment."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
PACKAGES=('i20r20_eval_v1','i20r20_ablation_v1','i20r20_followup_v1','i20r20_resume_v1')
STAGES=('plan','check-data','prepare-baseline','train-baseline','configure-extensions',
        'evaluate-baseline','train-ablation','evaluate-extensions','all')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def environment():
    env=dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',MKL_THREADING_LAYER='SEQUENTIAL',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    return env


def runtime_packages(run):
    """Bind run-specific copies to a new baseline; model/training Python stays byte-identical."""
    baseline=run/'baseline';manifest=read(baseline/'manifest.json')
    for seed in range(101,106):
        folder=baseline/f'seed_{seed}';done=read(folder/'completed.json')
        if done['seed']!=seed or done['run_id']!=manifest['run_id']:
            raise ValueError('Baseline completion identity mismatch')
        if sha(folder/'best.pt')!=done['checkpoint_hashes']['best.pt']:
            raise ValueError('Baseline checkpoint checksum mismatch')
    destination=run/'code';marker=destination/'runtime_sources.json'
    source_hashes={str(p.relative_to(REPO/'code')):sha(p) for name in PACKAGES
                   for p in (REPO/'code'/name).iterdir() if p.suffix in ('.py','.json')}
    if marker.exists():
        saved=read(marker)
        if saved['baseline_run_id']!=manifest['run_id'] or saved['source_hashes']!=source_hashes:
            raise ValueError('Runtime code belongs to another baseline/source version')
        for name,digest in saved['runtime_hashes'].items():
            if sha(destination/name)!=digest:raise ValueError(f'Changed runtime source: {name}')
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Incomplete runtime-code preparation; preserve it and choose a fresh --run-root')
    destination.mkdir(parents=True,exist_ok=True)
    plan=read(REPO/'protocols/recorded_study_plan.json')
    plan.update(baseline_run_id=manifest['run_id'],declared_on=datetime.now().date().isoformat(),
        status_at_declaration='New reproduction run: baseline complete; extension rules copied before test-stage commands.')
    for name in PACKAGES:
        target=destination/name;target.mkdir()
        for source in (REPO/'code'/name).iterdir():
            if source.is_file() and source.suffix in ('.py','.json'):
                shutil.copyfile(source,target/source.name)
        if (target/'study_plan.json').exists():write(target/'study_plan.json',plan)
    inherited={}
    for key,package,names in [
        ('ablation','i20r20_ablation_v1',('run.py','common.py','models.py','training_engine.py','cnn20.py','study_plan.json')),
        ('baseline_eval','i20r20_eval_v1',('pipeline.py','evaluation_core.py','cnn20.py','study_plan.json','paper_reference.json'))]:
        inherited[key]={name:sha(destination/package/name) for name in names}
    write(destination/'i20r20_followup_v1/inherited_hashes.json',inherited)
    runtime_hashes={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    write(marker,{'baseline_run_id':manifest['run_id'],'source_hashes':source_hashes,
        'runtime_hashes':runtime_hashes,'note':'New identities/plan fingerprints only; no training/model Python changes.'})
    return destination


def invoke(script,arguments,run,label):
    run.mkdir(parents=True,exist_ok=True);logs=run/'launcher_logs';logs.mkdir(exist_ok=True)
    command=[sys.executable,'-u',str(script),*[str(a) for a in arguments]]
    record={'stage':label,'at':datetime.now().astimezone().isoformat(),'command':command,
            'python':sys.executable,'working_directory':str(REPO)}
    write(logs/f'{label}.json',record)
    print(f'=== {label}: fresh process ===',flush=True)
    child=subprocess.Popen(command,cwd=REPO,env=environment(),stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    try:
        with (logs/f'{label}.log').open('a',encoding='utf-8',buffering=1) as log:
            for line in child.stdout:
                print(line,end='',flush=True);log.write(line)
        result=child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:child.terminate()
        child.wait(timeout=15);raise
    if result:raise RuntimeError(f'{label} stopped with exit code {result}; see {logs/f"{label}.log"}')


def execute(stage,run,data):
    baseline=run/'baseline';evaluation=run/'baseline_evaluation';ablation=run/'ablation';followup=run/'followup'
    if stage=='check-data':
        invoke(REPO/'scripts/check_data.py',['--data-dir',data,'--sha256'],run,'check_data')
    elif stage=='prepare-baseline':
        invoke(REPO/'code/i20r20_formal/run.py',['prepare','--data-dir',data,'--run-dir',baseline],run,'prepare_baseline')
    elif stage=='train-baseline':
        invoke(REPO/'code/i20r20_formal/run.py',['train','--run-dir',baseline],run,'train_baseline')
    elif stage=='configure-extensions':
        runtime_packages(run);print('Run-specific extension packages configured.',flush=True)
    elif stage=='evaluate-baseline':
        code=runtime_packages(run);script=code/'i20r20_eval_v1/pipeline.py'
        invoke(script,['freeze','--train-dir',baseline,'--data-dir',data,'--eval-dir',evaluation],run,'freeze_baseline')
        invoke(script,['predict','--eval-dir',evaluation],run,'predict_baseline')
        invoke(script,['analyze','--eval-dir',evaluation],run,'analyze_baseline')
    elif stage=='train-ablation':
        code=runtime_packages(run);script=code/'i20r20_ablation_v1/run.py'
        invoke(script,['prepare','--baseline-dir',baseline,'--data-dir',data,'--run-root',ablation],run,'prepare_ablation')
        invoke(script,['check','--run-root',ablation],run,'check_ablation')
        invoke(script,['train','--run-root',ablation],run,'train_ablation')
    elif stage=='evaluate-extensions':
        code=runtime_packages(run);script=code/'i20r20_followup_v1/run.py'
        invoke(script,['prepare','--baseline-dir',baseline,'--ablation-root',ablation,'--baseline-eval',evaluation,
                       '--data-dir',data,'--output',followup],run,'prepare_followup')
        invoke(script,['check','--output',followup],run,'check_followup')
        invoke(script,['predict','--output',followup],run,'predict_followup')
        invoke(code/'i20r20_resume_v1/resume.py',['--package',code/'i20r20_followup_v1','--output',followup],run,'finish_followup')
    else:raise ValueError(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=STAGES,nargs='?',default='plan')
    parser.add_argument('--run-root',type=Path,default=REPO/'runs/local_reproduction')
    parser.add_argument('--data-dir',type=Path,default=REPO/'data/monthly_20d')
    args=parser.parse_args();run=args.run_root.resolve();data=args.data_dir.resolve()
    protected=[REPO/'code',REPO/'results',REPO/'records',REPO/'environment',data]
    if run==REPO or any(run==p or run.is_relative_to(p) or p.is_relative_to(run) for p in protected):
        parser.error('Use a separate run directory outside recorded results, source code and data.')
    order=[s for s in STAGES if s not in ('plan','all')]
    if args.stage=='plan':
        print(json.dumps({'run_root':str(run),'data_dir':str(data),'stages':order,
            'training_members':15,'training_started':False,'recorded_results_modified':False},indent=2));return
    for stage in (order if args.stage=='all' else [args.stage]):execute(stage,run,data)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit('Interrupted; existing completed epochs/years are retained.')
