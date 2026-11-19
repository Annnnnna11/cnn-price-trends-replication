"""Render saved Grad-CAM arrays in a CPU-only process; no Torch imports."""
from datetime import datetime
import sys
import numpy as np
import pandas as pd
from common import architecture_spec
from guards import verify_contract, check_files
from evaluation_core import read_json, write_json, sha256_file


def run(root):
    if 'torch' in sys.modules:
        raise RuntimeError('Rendering requires a fresh process without Torch')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contract = verify_contract(root)
    output = root / 'gradcam'
    compute = read_json(output / 'compute_complete.json')
    if compute['followup_id'] != contract['followup_id'] or not compute['model_states_unchanged']:
        raise ValueError('Grad-CAM compute identity / state check mismatch')
    check_files(output, compute['files'])
    complete = output / 'complete.json'
    if complete.exists():
        done = read_json(complete)
        if done['followup_id'] != contract['followup_id']:
            raise ValueError('Existing render output belongs to a different experiment')
        check_files(output, done['files'])
        print('GRADCAM RENDER: completed output verified and skipped', flush=True)
        return
    selected = pd.read_csv(output / 'selected_samples.csv', dtype={'StockID': str})
    metadata = pd.read_csv(output / 'map_metadata.csv', dtype={'StockID': str})
    if metadata.duplicated(['CatalogIndex','depth','target_class']).any():
        raise ValueError('Duplicate map metadata keys')
    if len(metadata) != len(selected)*6 or not metadata.seed.eq(101).all():
        raise ValueError('Grad-CAM must contain six fixed-seed maps per selected input')
    files = dict(compute['files'])
    files['compute_complete.json'] = sha256_file(output / 'compute_complete.json')
    for sample in selected.itertuples():
        part = metadata.loc[metadata.CatalogIndex.eq(sample.CatalogIndex)]
        stem = f'{sample.reference_group}_{int(sample.CatalogIndex)}'
        if set(zip(part.depth,part.target_class)) != {(d,c) for d in (2,3,4) for c in (0,1)}:
            raise ValueError(f'Incomplete depth/class combinations: {stem}')
        maps, reference = {}, None
        for row in part.itertuples():
            with np.load(output / row.map_file, allow_pickle=False) as saved:
                raw = saved['input_uint8'].copy()
                heatmap = saved['normalized'].copy()
                native = saved['relu_native']
                expected_height = architecture_spec(row.depth)['block_shapes'][-1]['height_before_pool']
                if raw.shape != (1,64,60) or heatmap.shape != (64,60) or native.shape != (expected_height,60):
                    raise ValueError(f'Unexpected image/map size: {row.map_file}')
                if not np.isfinite(heatmap).all() or ((heatmap<0)|(heatmap>1)).any():
                    raise ValueError('Invalid normalized map')
            if reference is None:
                reference = raw
            elif not np.array_equal(reference, raw):
                raise ValueError('Depths/classes refer to different input images')
            maps[row.depth,row.target_class] = heatmap, row
        image = reference[0].astype(np.float32)/255.
        fig, axes = plt.subplots(2,4,figsize=(11,7),constrained_layout=True)
        for target in (0,1):
            axes[target,0].imshow(image,cmap='gray',vmin=0,vmax=1)
            axes[target,0].set_title('Input' if target==0 else f'True class: {int(sample.Label)}')
            axes[target,0].axis('off')
            for column,depth in enumerate((2,3,4),1):
                heatmap,row = maps[depth,target]
                ax = axes[target,column]
                ax.imshow(image,cmap='gray',vmin=0,vmax=1)
                im = ax.imshow(heatmap,cmap='inferno',vmin=0,vmax=1,alpha=.55)
                ax.set_title(f'Depth {depth}; target={target}\np(up)={row.p_up:.3f}' + ('; ZERO MAP' if row.zero_map else ''))
                ax.axis('off')
        fig.suptitle(f'{stem} | {sample.StockID} | {str(sample.Date)[:10]}\nTop: non-up logit; bottom: up logit; individually normalized maps')
        fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),shrink=.65,label='Within-map normalized Grad-CAM')
        name=f'{stem}.png'
        fig.savefig(output/name,dpi=180); plt.close(fig)
        files[name]=sha256_file(output/name)
        print(f'GRADCAM RENDER: {stem} | saved',flush=True)
    write_json(complete,{'followup_id':contract['followup_id'],'samples':len(selected),'maps':len(metadata),
        'finished_at':datetime.now().isoformat(),'model_states_unchanged':True,
        'execution':'i20r20_resume_v1; GPU compute and CPU rendering in separate processes', 'files':files})
    if 'torch' in sys.modules:
        raise RuntimeError('Renderer unexpectedly imported Torch')
    print(f'GRADCAM: COMPLETE | {len(selected)} inputs, {len(metadata)} maps | CPU renderer',flush=True)
