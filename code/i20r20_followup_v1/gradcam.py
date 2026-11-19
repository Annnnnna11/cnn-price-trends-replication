"""Fixed validation-only Grad-CAM. No optimizer, fitting or test selection."""
from datetime import datetime
from pathlib import Path
import hashlib
import os
import time
import numpy as np
import pandas as pd
from evaluation_core import (read_json, write_json, sha256_file, paths_for_year,
                             unchanged_source)
from common import architecture_spec
from guards import verify_contract, verify_frozen


def choose_cases(frame, sampling_seed=20261118, count=3):
    """Uniform sampling in fixed TP,TN,FP,FN order; no backfilling."""
    if frame.CatalogIndex.duplicated().any():
        raise ValueError('Repeated validation catalog index')
    if not np.isfinite(frame.reference_probability).all():
        raise ValueError('Nonfinite validation probability')
    y, up = frame.Label.to_numpy(), frame.reference_probability.to_numpy() > .5
    masks = {'TP': (y == 1) & up, 'TN': (y == 0) & ~up,
             'FP': (y == 0) & up, 'FN': (y == 1) & ~up}
    rng = np.random.default_rng(sampling_seed)
    pieces, counts = [], {}
    for group, mask in masks.items():
        eligible = frame.loc[mask].sort_values('CatalogIndex')
        counts[group] = len(eligible)
        take = min(count, len(eligible))
        chosen = rng.choice(len(eligible), size=take, replace=False)
        selected = eligible.iloc[np.sort(chosen)].copy()
        selected['reference_group'] = group
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True), counts


def configure(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('Use the Windows cnn20 CUDA environment')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(frozen, seed=101):
    import torch
    from models import DepthCNN
    member = next(m for m in frozen['members'] if m['seed'] == seed)
    path = Path(frozen['training_dir']) / f'seed_{seed}/best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['run_id'] != frozen['run_id'] or checkpoint['seed'] != seed
            or checkpoint.get('depth', 3) != frozen['depth']
            or checkpoint['epoch'] != member['best_epoch']
            or checkpoint['normalization'] != member['normalization']):
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    model = DepthCNN(frozen['depth'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model.to('cuda').eval(), member['normalization']


def state_fingerprint(model):
    """Includes BatchNorm running buffers, so accidental training mode is caught."""
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def activation_cam(logits, activation, target):
    import torch
    gradients = torch.autograd.grad(logits[:, target].sum(), activation, retain_graph=True)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    signed = (weights * activation).sum(dim=1, keepdim=True)
    native = signed.relu()
    resized = torch.nn.functional.interpolate(native, size=(64, 60), mode='bilinear', align_corners=False)
    peak = resized.amax(dim=(2, 3), keepdim=True)
    normalized = torch.where(peak > 0, resized / peak.clamp_min(torch.finfo(resized.dtype).tiny), torch.zeros_like(resized))
    if not all(torch.isfinite(t).all().item() for t in (gradients, weights, signed, native, normalized)):
        raise FloatingPointError('Nonfinite Grad-CAM value')
    return signed, native, resized, normalized, peak


def verify_cam_math(torch):
    # Exact synthetic linear score: gradients/channel averages and ReLU have a known solution.
    activation = torch.tensor([[[[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]]], device='cuda', requires_grad=True)
    score = 2*activation[:, 0].sum((1, 2)) - activation[:, 1].sum((1, 2))
    logits = torch.stack([-score, score], dim=1)
    signed, native, _, normalized, _ = activation_cam(logits, activation, 1)
    expected = 2*activation[:, :1] - activation[:, 1:]
    torch.testing.assert_close(signed, expected)
    torch.testing.assert_close(native, expected.relu())
    assert tuple(normalized.shape) == (1, 1, 64, 60)
    assert normalized.min() >= 0 and normalized.max() <= 1
    negative_logits = torch.stack([activation.sum((1, 2, 3)), -activation.sum((1, 2, 3))], dim=1)
    _, native_zero, _, normalized_zero, peak_zero = activation_cam(negative_logits, activation, 1)
    assert not native_zero.any() and not normalized_zero.any() and not peak_zero.any()


def validation_predictions(root, frozen):
    import torch
    from torch.utils.data import Dataset, DataLoader
    output = root / 'gradcam'
    output.mkdir(exist_ok=True)
    path, marker = output / 'reference_validation.feather', output / 'reference_validation.complete.json'
    contract = read_json(root / 'followup.json')
    base = Path(frozen['training_dir'])
    if marker.exists():
        record = read_json(marker)
        if record['followup_id'] != contract['followup_id'] or sha256_file(path) != record['sha256']:
            raise ValueError('Grad-CAM validation cache identity/checksum mismatch')
        return pd.read_feather(path)
    catalog = pd.read_feather(base / 'in_sample_catalog.feather')
    with np.load(base / 'seed_101/split.npz', allow_pickle=False) as split:
        indices = np.sort(split['validation'])
    frame = catalog.iloc[indices][['Year', 'Row', 'Date', 'StockID', 'Label']].copy().reset_index(drop=True)
    frame.insert(0, 'CatalogIndex', indices)
    if not frame.Year.between(1993, 1999).all():
        raise ValueError('Grad-CAM selection must use baseline in-sample validation only')
    manifest = read_json(base / 'manifest.json')
    model, norm = load_model(frozen)

    class ValidationImages(Dataset):
        def __init__(self):
            self.maps = {}
        def __len__(self):
            return len(frame)
        def __getitem__(self, index):
            row = frame.iloc[index]
            year = int(row.Year)
            if year not in self.maps:
                image, _ = paths_for_year(frozen['data_dir'], year)
                self.maps[year] = np.memmap(image, mode='r', dtype=np.uint8,
                                          shape=(manifest['original_counts'][str(year)], 1, 64, 60))
            return torch.from_numpy(self.maps[year][int(row.Row)].copy())

    loader = DataLoader(ValidationImages(), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    probabilities = np.empty(len(frame), dtype=np.float32)
    start, cursor = time.perf_counter(), 0
    before = state_fingerprint(model)
    with torch.inference_mode():
        for number, raw in enumerate(loader, 1):
            x = (raw.to('cuda', dtype=torch.float32) / 255. - norm['mean']) / norm['std']
            p = model(x).softmax(dim=1)[:, 1].cpu().numpy()
            probabilities[cursor:cursor+len(p)] = p
            cursor += len(p)
            if number % 200 == 0 or number == len(loader):
                print(f'GRADCAM reference validation: {cursor}/{len(frame)}, {time.perf_counter()-start:.1f}s', flush=True)
    if cursor != len(frame) or not np.isfinite(probabilities).all() or before != state_fingerprint(model):
        raise RuntimeError('Invalid validation prediction or model state changed')
    frame['reference_probability'] = probabilities
    temporary = path.with_name(path.name + '.tmp')
    frame.to_feather(temporary); os.replace(temporary, path)
    write_json(marker, {'followup_id': contract['followup_id'], 'sha256': sha256_file(path),
                        'rows': len(frame), 'seconds': time.perf_counter()-start})
    del model, loader
    torch.cuda.empty_cache()
    return frame


def run(root):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contract = verify_contract(root)
    configure(torch); verify_cam_math(torch)
    output = root / 'gradcam'; output.mkdir(exist_ok=True)
    complete_path = output / 'complete.json'
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
        raw_float = raw[0].astype(np.float32) / 255.
        stem = f'{sample.reference_group}_{int(sample.CatalogIndex)}'
        fig, axes = plt.subplots(2, 4, figsize=(11, 7), constrained_layout=True)
        for target in (0, 1):
            axes[target, 0].imshow(raw_float, cmap='gray', vmin=0, vmax=1)
            axes[target, 0].set_title('Input' if target == 0 else f'True class: {int(sample.Label)}')
            axes[target, 0].axis('off')
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
                        axis = axes[target, column]
                        axis.imshow(raw_float, cmap='gray', vmin=0, vmax=1)
                        im = axis.imshow(heatmap, cmap='inferno', vmin=0, vmax=1, alpha=.55)
                        axis.set_title(f'Depth {depth}; target={target}\np(up)={p:.3f}' + ('; ZERO MAP' if zero else ''))
                        axis.axis('off')
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
        fig.suptitle(f'{stem} | {sample.StockID} | {str(sample.Date)[:10]}\nTop row: non-up logit; bottom row: up logit; individually normalized maps')
        fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=.65, label='Within-map normalized Grad-CAM')
        fig.savefig(output / f'{stem}.png', dpi=180)
        plt.close(fig); artifact_files.append(f'{stem}.png')
        print(f'GRADCAM: {stem} | 6 class/depth maps saved', flush=True)
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
                              'finished_at': datetime.now().isoformat(), 'model_states_unchanged': True,
                              'files': {name: sha256_file(output / name) for name in artifact_files}})
    print(f'GRADCAM: COMPLETE | {len(selected)} inputs, {len(entries)} maps | {output}', flush=True)
