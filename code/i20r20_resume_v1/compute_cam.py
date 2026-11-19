"""Same frozen Grad-CAM protocol; compute arrays without importing Matplotlib.

Adapted from i20r20_followup_v1/gradcam.py run(), separating only rendering.
Model loading, case sampling, activation_cam and validation inference are reused
from the frozen package verbatim. No optimizer or training entry point exists.
"""
from datetime import datetime
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from evaluation_core import read_json, write_json, sha256_file, paths_for_year, unchanged_source
from common import architecture_spec
from guards import verify_contract, verify_frozen
from gradcam import (configure, verify_cam_math, validation_predictions, choose_cases,
                     load_model, state_fingerprint, activation_cam)

def run(root):
    if any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules):
        raise RuntimeError('GPU compute must run without Matplotlib loaded')
    import torch
    contract = verify_contract(root)
    configure(torch); verify_cam_math(torch)
    output = root / 'gradcam'; output.mkdir(exist_ok=True)
    complete_path = output / 'compute_complete.json'
    if complete_path.exists():
        done = read_json(complete_path)
        if done['followup_id'] != contract['followup_id']:
            raise ValueError('Grad-CAM output belongs to a different experiment')
        from guards import check_files
        check_files(output, done['files'])
        print(f'GRADCAM: completed outputs verified and skipped | {output}', flush=True)
        return
    frozen = {d: verify_frozen(root / f'depth_{d}') for d in (2, 3, 4)}
    reference = validation_predictions(root, frozen[3])
    selected, counts = choose_cases(reference)
    records = [{k: (v.item() if isinstance(v, np.generic) else str(v) if isinstance(v, pd.Timestamp) else v)
                for k, v in row.items()} for row in selected.to_dict('records')]
    selection = {'followup_id': contract['followup_id'], 'sampling_seed': 20261118,
                 'group_order': ['TP', 'TN', 'FP', 'FN'], 'group_population': counts,
                 'reference': 'depth3 seed101 validation; p>0.5; catalog-index sorted pool', 'samples': records}
    selection_path = output / 'selection.json'
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError('Frozen case selection differs; refusing to replace it')
    write_json(selection_path, selection)
    selected.to_csv(output / 'selected_samples.csv', index=False)
    base_manifest = read_json(Path(frozen[3]['training_dir']) / 'manifest.json')
    images, source_labels = {}, {}
    for row in selected.itertuples():
        year = int(row.Year)
        if year not in source_labels:
            image_path, label_path = paths_for_year(contract['data_dir'], year)
            source_labels[year] = pd.read_feather(label_path)
            n = base_manifest['original_counts'][str(year)]
            images[year] = np.memmap(image_path, mode='r', dtype=np.uint8, shape=(n, 1, 64, 60))
        raw_label = source_labels[year].iloc[int(row.Row)]
        if (str(raw_label.StockID) != str(row.StockID) or pd.Timestamp(raw_label.Date) != pd.Timestamp(row.Date)
                or int(raw_label.Ret_20d > 0) != int(row.Label)):
            raise ValueError('Selected sample metadata does not match raw label row')
    models, before = {}, {}
    for depth in (2, 3, 4):
        model, norm = load_model(frozen[depth])
        models[depth] = model, norm
        before[depth] = state_fingerprint(model)
    entries, artifact_files = [], ['selection.json', 'selected_samples.csv',
                                  'reference_validation.feather', 'reference_validation.complete.json']
    for sample in selected.itertuples():
        raw = images[int(sample.Year)][int(sample.Row)].copy()
        stem = f'{sample.reference_group}_{int(sample.CatalogIndex)}'
        for column, depth in enumerate((2, 3, 4), 1):
            model, norm = models[depth]
            captured = []
            handle = model.conv_layers[depth-1][2].register_forward_hook(lambda module, inputs, result: captured.append(result))
            try:
                with torch.enable_grad():
                    x = torch.from_numpy(raw).unsqueeze(0).to('cuda', dtype=torch.float32) / 255.
                    x = ((x - norm['mean']) / norm['std']).requires_grad_(True)
                    logits = model(x)
                    p = float(logits.detach().softmax(dim=1)[0, 1].cpu())
                    activation = captured[0]
                    expected_shape = architecture_spec(depth)['block_shapes'][-1]
                    if tuple(activation.shape[-2:]) != (expected_shape['height_before_pool'], 60):
                        raise ValueError('Unexpected Grad-CAM target layer shape')
                    if depth == 3 and abs(p - sample.reference_probability) > 1e-5:
                        raise ValueError('Reference prediction differs between batch inference and Grad-CAM')
                    for target in (0, 1):
                        signed, native, resized, normalized, peak = activation_cam(logits, activation, target)
                        heatmap = normalized.detach().cpu().numpy()[0, 0]
                        zero = bool(peak.detach().item() == 0)
                        name = f'{stem}_depth{depth}_class{target}.npz'
                        with (output / (name+'.tmp')).open('wb') as stream:
                            np.savez_compressed(stream, input_uint8=raw, signed_native=signed.detach().cpu().numpy()[0, 0],
                                relu_native=native.detach().cpu().numpy()[0, 0],
                                relu_upsampled=resized.detach().cpu().numpy()[0, 0], normalized=heatmap)
                        os.replace(output / (name+'.tmp'), output / name)
                        artifact_files.append(name)
                        entries.append({'CatalogIndex': int(sample.CatalogIndex), 'Year': int(sample.Year),
                            'Row': int(sample.Row), 'Date': str(sample.Date), 'StockID': str(sample.StockID),
                            'Label': int(sample.Label), 'reference_group': sample.reference_group,
                            'depth': depth, 'seed': 101, 'target_class': target, 'p_up': p,
                            'target_logit': float(logits[0, target].detach().cpu()),
                            'native_height': activation.shape[2], 'native_width': activation.shape[3],
                            'target_layer': f'conv_layers.{depth-1}.2', 'zero_map': zero,
                            'unnormalized_peak': float(peak.detach().cpu().item()), 'map_file': name,
                            'checkpoint_sha256': frozen[depth]['members'][0]['files']['best.pt']})
            finally:
                handle.remove()
        print(f'GRADCAM COMPUTE: {stem} | 6 raw maps saved; plotting runs in a separate process', flush=True)
    for depth, (model, _) in models.items():
        if before[depth] != state_fingerprint(model):
            raise RuntimeError('Grad-CAM changed model parameters / BatchNorm buffers')
    for entry in contract['training_inputs']:
        if not unchanged_source(Path(contract['data_dir']) / entry['name'], entry):
            raise ValueError('Training input changed during Grad-CAM')
    pd.DataFrame(entries).to_csv(output / 'map_metadata.csv', index=False)
    artifact_files.append('map_metadata.csv')
    report = f'''# Grad-CAM 说明

本流程使用固定 seed101 的二、三、四层最佳模型，不是五模型融合的解释，也没有继续训练。
样本来自三层 seed101 验证集。由三层结果定义 TP/TN/FP/FN，各组均匀抽取最多 3 张，随机种子 20261118；
不足 3 张不补齐。组内先按 CatalogIndex 排序，选择记录保存在 selection.json。没有使用测试结果挑选案例。

共选择 {len(selected)} 张输入，生成 {len(entries)} 张类别/结构热图。
TP/TN/FP/FN 的标签始终指三层参考模型；另两个结构对同一输入的判断可能不同，见 map_metadata.csv。
每张 PNG 四列依次为原图、二层、三层、四层；上行解释非上涨 logit，下行解释上涨 logit。

目标层为最后卷积块的 LeakyReLU 输出、池化之前。对指定类别的 softmax 前分数求梯度，
对空间位置平均梯度作为通道权重，加权求和后 ReLU，再双线性放大至 64×60，除以本图最大值。
原生热图尺寸依次为 10×60、5×60、3×60。色标仅表示每张图内部的相对强度，不能跨模型比较绝对贡献。
全零热图会保留并标记 ZERO MAP，而不人为制造热点。

NPZ 保留输入像素、ReLU 前原生图、ReLU 后原生图、放大图和归一化图；CSV 记录样本、概率、logit、目标层和权重指纹。
模型全程 eval，梯度仅用于解释，检查了参数与 BatchNorm 缓冲区在解释前后完全相同。
热图表示这些模型对输入区域的局部敏感性，不能证明价格形态具有因果关系或投资收益保证。

方法依据：Selvaraju et al., Grad-CAM, https://arxiv.org/abs/1610.02391 。
课程建议该扩展；本文件提供 PyTorch 实现，不声称是作者仓库中附带的 Grad-CAM 代码。
'''
    (output / 'report_zh.md').write_text(report, encoding='utf-8'); artifact_files.append('report_zh.md')
    write_json(complete_path, {'followup_id': contract['followup_id'], 'samples': len(selected), 'maps': len(entries),
                              'finished_at': datetime.now().isoformat(), 'model_states_unchanged': True, 'execution': 'GPU compute only; no Matplotlib import; i20r20_resume_v1',
                              'files': {name: sha256_file(output / name) for name in artifact_files}})
    print(f'GRADCAM COMPUTE: COMPLETE | {len(selected)} inputs, {len(entries)} maps | {output}', flush=True)
