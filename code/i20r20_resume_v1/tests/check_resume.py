"""Synthetic renderer, ZIP, subprocess isolation and compatibility checks.

No real model, test data, Windows DLL or CUDA execution is used locally.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
OLD=HERE.parent/'i20r20_followup_v1'
sys.path.insert(0,str(OLD)); sys.path.insert(1,str(HERE))
import resume
import render_cam
import compute_cam
import gradcam
import worker
from evaluation_core import read_json,write_json,sha256_file
from common import architecture_spec

for name in ('activation_cam','choose_cases','validation_predictions','load_model','state_fingerprint'):
    assert getattr(compute_cam,name) is getattr(gradcam,name)
for path in HERE.glob('*.py'):
    tree=ast.parse(path.read_text(),feature_version=(3,11))
    if path.name in ('resume.py','render_cam.py'):
        assert not any(isinstance(node,ast.Import) and any(alias.name.startswith('torch') for alias in node.names) for node in ast.walk(tree))
    if path.name=='compute_cam.py':
        assert not any(isinstance(node,ast.Import) and any(alias.name.startswith('matplotlib') for alias in node.names) for node in ast.walk(tree))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('train','train_member','predict') for node in ast.walk(tree))
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
env=resume.child_environment()
assert 'KMP_DUPLICATE_LIB_OK' not in env and env['MKL_THREADING_LAYER']=='SEQUENTIAL'
os.environ.pop('KMP_DUPLICATE_LIB_OK')

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)/'output'; out=root/'gradcam'; out.mkdir(parents=True)
    selected=pd.DataFrame([dict(CatalogIndex=7,Year=1993,Row=7,Date='1993-01-29',StockID='12345',Label=1,reference_group='TP')])
    selected.to_csv(out/'selected_samples.csv',index=False)
    write_json(out/'selection.json',{'synthetic':True})
    image=np.zeros((1,64,60),dtype=np.uint8)
    for col in range(60): image[0,45-col//3:48-col//3,col]=255
    image[0,58:64,::3]=255
    metadata=[]; files=['selected_samples.csv','selection.json']
    for depth in (2,3,4):
        height=architecture_spec(depth)['block_shapes'][-1]['height_before_pool']
        yy,xx=np.mgrid[:64,:60]
        heat=np.exp(-((yy-30)**2+(xx-35)**2)/150).astype(np.float32)
        for target in (0,1):
            heatmap=heat if target else np.zeros_like(heat)
            name=f'TP_7_depth{depth}_class{target}.npz'
            np.savez_compressed(out/name,input_uint8=image,normalized=heatmap,relu_native=np.zeros((height,60),np.float32))
            files.append(name)
            metadata.append(dict(CatalogIndex=7,depth=depth,target_class=target,seed=101,p_up=.56+depth/100,
                                 zero_map=target==0,map_file=name))
    pd.DataFrame(metadata).to_csv(out/'map_metadata.csv',index=False); files.append('map_metadata.csv')
    write_json(out/'compute_complete.json',{'followup_id':'synthetic','model_states_unchanged':True,
                                           'files':{n:sha256_file(out/n) for n in files}})
    verify=render_cam.verify_contract
    render_cam.verify_contract=lambda root:{'followup_id':'synthetic'}
    try:
        render_cam.run(root); render_cam.run(root)
        assert 'torch' not in sys.modules
        done=read_json(out/'complete.json'); assert done['samples']==1 and done['maps']==6
        preview=Path('/private/tmp/i20r20_resume_preview.png'); preview.write_bytes((out/'TP_7.png').read_bytes())
        corrupted=out/'TP_7.png'; corrupted.write_bytes(b'changed')
        try: render_cam.run(root)
        except ValueError as error: assert 'Missing or changed artifact' in str(error)
        else: raise AssertionError('Changed rendered output accepted')
    finally:
        render_cam.verify_contract=verify
    # Final runtime metadata must be inside the ZIP and its hash manifest.
    runtime=root/'runtime_fix'; runtime.mkdir()
    write_json(runtime/'audit.json',{'synthetic':True})
    write_json(root/'handoff_manifest.json',{'files':{},'not_included':[]})
    with zipfile.ZipFile(root/'i20r20_followup_results.zip','w') as z:
        z.write(root/'handoff_manifest.json',f'{root.name}/handoff_manifest.json')
        z.writestr(f'{root.name}/example.txt','retained')
    worker.add_runtime_records(root)
    with zipfile.ZipFile(root/'i20r20_followup_results.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        handoff=json.loads(z.read(f'{root.name}/handoff_manifest.json'))
        assert handoff['files']['runtime_fix/audit.json']==sha256_file(runtime/'audit.json')
        assert z.read(f'{root.name}/example.txt')==b'retained'

with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory); package=directory/'old'; package.mkdir()
    root=directory/'output'; root.mkdir(); write_json(root/'followup.json',{})
    (package/'guards.py').write_text('def verify_contract(root): return {"followup_id":"fixture"}\ndef verify_frozen(root): return {}\n')
    (package/'common.py').write_text('from contextlib import contextmanager\n@contextmanager\ndef experiment_lock(root): yield\n')
    (package/'engine.py').write_text('def verify_year(directory, frozen, year): return True\n')
    (package/'evaluation_core.py').write_text('import json,os\ndef write_json(path,value):\n value["pid"]=os.getpid()\n path.write_text(json.dumps(value))\n')
    (package/'run.py').write_text('import sys,json,os\ndef analysis(root):\n assert "torch" not in sys.modules\n (root/"analysis_fixture.json").write_text(json.dumps({"pid":os.getpid(),"mkl":os.environ.get("MKL_THREADING_LAYER")}))\n')
    for stage in ('audit','analyze'):
        command=[sys.executable,str(HERE/'resume.py'),'--package',str(package),'--output',str(root),'--stage',stage]
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=resume.child_environment())
        assert result.returncode==0,result.stdout
    audit=read_json(root/'runtime_fix/audit.json'); analysis=read_json(root/'analysis_fixture.json')
    assert audit['verified_years']=={'2':20,'3':20,'4':20}
    assert not audit['torch_loaded'] and audit['pid']!=analysis['pid']
    assert analysis['mkl']=='SEQUENTIAL'
    (package/'engine.py').write_text('def verify_year(directory, frozen, year): return year!=2007\n')
    result=subprocess.run([sys.executable,str(HERE/'resume.py'),'--package',str(package),'--output',str(root),'--stage','audit'],
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=resume.child_environment())
    assert result.returncode!=0 and 'Missing prediction years' in result.stdout

print('PASS: frozen helpers reused, no training/test prediction calls, standard-library launcher, isolated CPU stages.')
print('PASS: synthetic Grad-CAM renderer, zero maps, cache integrity, final ZIP metadata, missing-year gate.')
print('No actual CUDA/Windows DLL execution. Real environment recovery must be confirmed on Windows.')
