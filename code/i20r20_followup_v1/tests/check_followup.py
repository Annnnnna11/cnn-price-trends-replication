"""Synthetic control-flow tests, with Arrow explicitly replaced by pickle.

Never trains a model or reads actual test data; CUDA checks run on Windows.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guards
import engine
from comparison import assert_paired, compare
from gradcam import choose_cases
from common import architecture_spec
from evaluation_core import (read_json, write_json, sha256_file, source_record, paths_for_year,
                             prepare_labels, SEEDS, YEARS)
from run import bundle


def expect_error(action, text):
    try:
        action()
    except (ValueError, RuntimeError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f'Expected failure containing {text}')


def test_selection():
    frame = pd.DataFrame({'CatalogIndex': np.arange(15), 'Label': [1]*5+[0]*5+[0]*2+[1]*3,
                          'reference_probability': [.8]*5+[.2]*5+[.8]*2+[.5]*3})
    first, counts = choose_cases(frame)
    second, _ = choose_cases(frame.sample(frac=1, random_state=4))
    assert list(first.CatalogIndex) == list(second.CatalogIndex)
    assert counts == dict(TP=5, TN=5, FP=2, FN=3)
    assert first.reference_group.value_counts().to_dict() == dict(TP=3, TN=3, FP=2, FN=3)
    assert len(first) == 11  # no backfill; ties at .5 belong to non-up


def fixture(parent):
    package, base, ablation, prior, data, output = [parent/name for name in ('package','base','ablation','prior','data','output')]
    for folder in (package, base, ablation, prior/'predictions', data, output):
        folder.mkdir(parents=True, exist_ok=True)
    for path in guards.HERE.iterdir():
        if path.suffix in ('.py','.json') and path.name != 'VALIDATION.json':
            shutil.copyfile(path, package/path.name)
    inherited = read_json(package/'inherited_hashes.json')
    training_image, training_label = paths_for_year(data, 1993)
    training_image.write_bytes(bytes(40*3840)); training_label.write_bytes(b'synthetic training labels')
    catalog = pd.DataFrame({'Year': np.full(40,1993,dtype=np.int16), 'Row': np.arange(40,dtype=np.int32),
                            'Label': np.arange(40)%2})
    catalog.to_pickle(base/'in_sample_catalog.feather')
    artifacts = {'in_sample_catalog.feather': sha256_file(base/'in_sample_catalog.feather')}
    for seed in SEEDS:
        sub=base/f'seed_{seed}'; sub.mkdir()
        np.savez(sub/'split.npz',train=np.arange(28),validation=np.arange(28,40))
        write_json(sub/'normalization.json',dict(mean=.1,std=.3,train_up_fraction=.5))
        for name in ('split.npz','normalization.json'):
            artifacts[f'seed_{seed}/{name}']=sha256_file(sub/name)
    baseline = {'protocol': {'version':'i20r20_formal_v1'}, 'in_sample_rows':40,
                'original_counts':{'1993':40}, 'inputs':[source_record(training_image),source_record(training_label)],
                'artifact_hashes':artifacts, 'code_hashes':{'cnn20.py':sha256_file(package/'cnn20.py')}}
    baseline['run_id']=guards.identity(baseline,'run_id'); write_json(base/'manifest.json',baseline)
    plan=read_json(package/'study_plan.json'); plan['baseline_run_id']=baseline['run_id']; write_json(package/'study_plan.json',plan)
    manifests={3:baseline}
    for depth in (2,4):
        folder=ablation/f'depth_{depth}'; shutil.copytree(base,folder)
        protocol=dict(baseline['protocol'],version=f'i20r20_ablation_v1_depth{depth}',architecture_depth=depth)
        spec=architecture_spec(depth)
        manifest={**baseline,'protocol':protocol,'depth':depth,'baseline_run_id':baseline['run_id'],
                  'specification':spec,'code_hashes':inherited['ablation']}
        manifest.pop('run_id'); manifest['run_id']=guards.identity(manifest,'run_id')
        write_json(folder/'manifest.json',manifest); manifests[depth]=manifest
    for depth in (2,3,4):
        folder=base if depth==3 else ablation/f'depth_{depth}'
        for seed in SEEDS:
            sub=folder/f'seed_{seed}'
            (sub/'best.pt').write_bytes(f'synthetic {depth} {seed}'.encode())
            done={'run_id':manifests[depth]['run_id'],'seed':seed,'best_epoch':1,'epochs_run':2,
                  'validation_metrics':dict(cross_entropy=.69,accuracy=.53,mcc=.05),
                  'elapsed_seconds':12.,'peak_cuda_allocated_mib':500.,
                  'checkpoint_hashes':{'best.pt':sha256_file(sub/'best.pt')}}
            if depth!=3: done['depth']=depth
            write_json(sub/'completed.json',done)
            pd.DataFrame({'epoch':[1,2],'validation_cross_entropy':[.69,.70]}).to_csv(sub/'history.csv',index=False)
    write_json(ablation/'prepared.json',dict(baseline_dir=str(base),data_dir=str(data),
        code_hashes=inherited['ablation'],baseline_manifest_sha256=sha256_file(base/'manifest.json'),
        baseline_run_id=baseline['run_id'],variants={str(d):manifests[d]['run_id'] for d in (2,4)}))
    old={'training_dir':str(base),'data_dir':str(data),'training_manifest_sha256':sha256_file(base/'manifest.json'),
         'code_hashes':inherited['baseline_eval'],'study_plan':plan,'evaluation_id':'synthetic-baseline-eval',
         'members':guards.completion_members(base,baseline,3)}
    write_json(prior/'frozen_evaluation.json',old)
    for year in YEARS:
        image,label=paths_for_year(data,year); image.write_bytes(bytes(40*3840))
        raw=pd.DataFrame({'Date':[pd.Timestamp(year,1,31)]*40,'StockID':[str(i) for i in range(40)],
                          'MarketCap':np.arange(1,41),'Ret_20d':np.linspace(-.1,.1,40),
                          'Ret_month':np.linspace(-.1,.1,40)*(1+(year-2000)/100), 'EWMA_vol':np.ones(40)})
        raw.to_pickle(label)
        frame=prepare_labels(raw,year)
        for seed in SEEDS: frame[f'p_{seed}']=np.linspace(.01,.99,40).astype(np.float32)
        frame['p_ensemble']=frame[[f'p_{s}' for s in SEEDS]].mean(axis=1).astype(np.float32)
        pred=prior/f'predictions/{year}.feather'; frame.to_pickle(pred)
        write_json(prior/f'predictions/{year}.complete.json',dict(evaluation_id=old['evaluation_id'],year=year,
            image_source=source_record(image),label_source=source_record(label),prediction_sha256=sha256_file(pred)))
    args=SimpleNamespace(baseline_dir=base,ablation_root=ablation,baseline_eval=prior,data_dir=data,output=output)
    return args,package


test_selection()
saved_here=guards.HERE
saved_reader,saved_writer=pd.read_feather,pd.DataFrame.to_feather
try:
    pd.read_feather=pd.read_pickle
    pd.DataFrame.to_feather=pd.DataFrame.to_pickle
    with tempfile.TemporaryDirectory() as tmp:
        args, package=fixture(Path(tmp)); guards.HERE=package
        # The same gate used by `all`: no inference until every candidate completes.
        path=args.ablation_root/'depth_4/seed_105/completed.json'
        content=path.read_bytes(); path.unlink()
        expect_error(lambda: guards.prepare(args),'Training NOT complete')
        path.write_bytes(content)
        wrong=read_json(path); wrong['depth']=2; write_json(path,wrong)
        expect_error(lambda: guards.prepare(args),'Completion identity mismatch')
        path.write_bytes(content)
        with redirect_stdout(io.StringIO()):
            guards.prepare(args); guards.prepare(args)
        (args.output/'depth_2/frozen_evaluation.json').unlink()
        with redirect_stdout(io.StringIO()): guards.prepare(args)
        assert (args.output/'depth_2/frozen_evaluation.json').is_file()
        assert guards.verify_contract(args.output)['evaluations']['3']['prediction_directory']==str((args.baseline_eval/'predictions').resolve())
        for depth in (2,4):
            directory=args.output/f'depth_{depth}'; (directory/'predictions').mkdir()
            frozen=guards.verify_frozen(directory)
            for year in YEARS:
                source=args.baseline_eval/f'predictions/{year}.feather'
                frame=pd.read_pickle(source)
                frame[[f'p_{s}' for s in SEEDS]+['p_ensemble']]*=np.float32(.99 if depth==2 else .98)
                pred,marker=engine.year_paths(directory,year); frame.to_pickle(pred)
                info=read_json(args.baseline_eval/f'predictions/{year}.complete.json')
                info.update(evaluation_id=frozen['evaluation_id'],prediction_sha256=sha256_file(pred))
                write_json(marker,info)
        frame=pd.read_pickle(args.baseline_eval/'predictions/2000.feather')
        changed=frame.copy(); changed.loc[0,'Ret_20d']=9.
        expect_error(lambda:assert_paired(frame,changed,2,2000),'test rows / labels / eligibility differ')
        for depth in (2,3,4):
            with redirect_stdout(io.StringIO()): engine.analyze(SimpleNamespace(eval_dir=args.output/f'depth_{depth}'))
        with redirect_stdout(io.StringIO()): compare(args.output)
        comparison=args.output/'comparison'
        scores=pd.read_csv(comparison/'ensemble_comparison.csv')
        assert len(scores)==6 and set(scores.depth)=={2,3,4}
        assert (scores.loc[scores.depth.eq(3),'accuracy_minus_depth3']==0).all()
        assert len(pd.read_csv(comparison/'paired_seed_comparison.csv'))==30
        assert len(pd.read_csv(comparison/'training_all_members.csv'))==15
        assert len(pd.read_csv(comparison/'paired_test_audit.csv'))==20
        assert (comparison/'learning_curves.png').is_file()
        assert len(list((comparison/'training_history').glob('*.csv')))==15
        # Bundle control flow: Grad-CAM algorithm itself is checked separately on Windows CUDA.
        grad=args.output/'gradcam'; grad.mkdir()
        write_json(grad/'complete.json',{'followup_id':guards.verify_contract(args.output)['followup_id'],'files':{}})
        write_json(args.output/'gpu_check.json',{'synthetic':True})
        with redirect_stdout(io.StringIO()): bundle(args.output)
        assert (args.output/'i20r20_followup_results.zip').is_file()
        preview=Path('/private/tmp/i20r20_followup_test_preview'); preview.mkdir(exist_ok=True)
        for name in ('ensemble_metrics.png','learning_curves.png','financial_HL.png','report_zh.md'):
            shutil.copyfile(comparison/name,preview/name)
        pred,marker=engine.year_paths(args.output/'depth_2',2000)
        pred.write_bytes(pred.read_bytes()+b'changed')
        expect_error(lambda:engine.verify_year(args.output/'depth_2',guards.verify_frozen(args.output/'depth_2'),2000),'prediction file missing or changed')
        # Cached input identity must reject accidental source drift.
        best=args.ablation_root/'depth_2/seed_101/best.pt'; best.write_bytes(b'changed')
        expect_error(lambda:guards.verify_contract(args.output),'Missing or changed artifact')
        print('PASS: completion/depth/checkpoint gates, paired metadata, cache checksums, baseline cache reuse.')
        print('PASS: 3-structure analysis, paired-seed tables, finance calendars, plots, handoff bundle; synthetic data only.')
        print('PASS: deterministic TP/TN/FP/FN sampling, tie behavior, undersized group without backfill.')
        print('Arrow was mocked with pickle. No Torch/CUDA inference or actual model attributions ran locally.')
finally:
    guards.HERE=saved_here
    pd.read_feather,pd.DataFrame.to_feather=saved_reader,saved_writer
