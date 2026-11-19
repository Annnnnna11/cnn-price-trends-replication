"""Paired test comparison. Never selects or tunes an architecture."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from common import architecture_spec
from evaluation_core import (YEARS, PERIODS, SEEDS, read_json, write_json,
                             sha256_file, portfolio_summary, markdown_table)
from guards import verify_contract, verify_frozen


META = ['Year', 'Row', 'Date', 'StockID', 'MarketCap', 'Ret_20d', 'Ret_month', 'EWMA_vol', 'eligible_input']


def assert_paired(reference, candidate, depth, year):
    """Exact row-order and label identity, including missing-value positions."""
    try:
        pd.testing.assert_frame_equal(reference[META], candidate[META], check_dtype=True, check_exact=True)
    except AssertionError as error:
        raise ValueError(f'Depth {depth}, year {year}: test rows / labels / eligibility differ') from error


def compare(root):
    import engine
    verify_contract(root)
    output = root / 'comparison'
    output.mkdir(exist_ok=True)
    fingerprints, audits = {}, []
    frozen = {d: verify_frozen(root / f'depth_{d}') for d in (2, 3, 4)}
    for year in YEARS:
        frames = {}
        sources = {}
        for depth in (2, 3, 4):
            directory = root / f'depth_{depth}'
            if not engine.verify_year(directory, frozen[depth], year):
                raise RuntimeError(f'Missing prediction: depth {depth}, {year}')
            path, marker = engine.year_paths(directory, year)
            frames[depth] = pd.read_feather(path)
            sources[depth] = read_json(marker)
            fingerprints[f'{depth}/{year}'] = sha256_file(path)
        for depth in (2, 4):
            assert_paired(frames[3], frames[depth], depth, year)
            for key in ('image_source', 'label_source'):
                if sources[3][key]['sha256'] != sources[depth][key]['sha256']:
                    raise ValueError('Three structures did not use identical source files')
        ref = frames[3]
        audits.append({'year': year, 'rows': len(ref), 'same_rows_labels_inputs': True,
                       'classification_rows': int((ref.eligible_input & np.isfinite(ref.Ret_20d)).sum())})
    pd.DataFrame(audits).to_csv(output / 'paired_test_audit.csv', index=False)
    classifications, portfolios, training = [], [], []
    for depth in (2, 3, 4):
        directory = root / f'depth_{depth}/analysis'
        marker = read_json(directory / 'analysis_complete.json')
        expected = {str(y): fingerprints[f'{depth}/{y}'] for y in YEARS}
        if marker['evaluation_id'] != frozen[depth]['evaluation_id'] or marker['source_prediction_sha256'] != expected:
            raise ValueError('Analysis does not match the verified prediction files')
        metrics = pd.read_csv(directory / 'classification_metrics.csv')
        metrics.insert(0, 'depth', depth)
        classifications.append(metrics)
        status = read_json(directory / 'financial_status.json')
        if status['status'] != 'computed':
            raise RuntimeError(f'Depth {depth}: financial comparison unavailable: {status}. '
                               'Classification files are retained; investigate before comparing finance.')
        monthly = pd.read_csv(directory / 'monthly_portfolio_returns.csv', parse_dates=['Date', 'ReturnMonth'])
        monthly.insert(0, 'depth', depth)
        portfolios.append(monthly)
        folder = Path(frozen[depth]['training_dir'])
        for seed in SEEDS:
            done = read_json(folder / f'seed_{seed}/completed.json')
            val = done['validation_metrics']
            training.append({'depth': depth, 'seed': seed, 'parameters': architecture_spec(depth)['parameters'],
                             'best_epoch': done['best_epoch'], 'epochs_run': done['epochs_run'],
                             'validation_ce': val['cross_entropy'], 'validation_accuracy': val['accuracy'],
                             'validation_mcc': val['mcc'], 'elapsed_seconds': done['elapsed_seconds'],
                             'peak_cuda_allocated_mib': done['peak_cuda_allocated_mib']})
    classification = pd.concat(classifications, ignore_index=True)
    monthly = pd.concat(portfolios, ignore_index=True)
    training = pd.DataFrame(training)
    classification.to_csv(output / 'classification_all_members.csv', index=False)
    monthly.to_csv(output / 'monthly_returns_all_depths.csv', index=False)
    training.to_csv(output / 'training_all_members.csv', index=False)
    histories = output / 'training_history'
    histories.mkdir(exist_ok=True)
    training.groupby('depth', as_index=False).agg(parameters=('parameters', 'first'),
        total_epochs=('epochs_run', 'sum'), recorded_training_seconds=('elapsed_seconds', 'sum'),
        peak_cuda_allocated_mib=('peak_cuda_allocated_mib', 'max')).to_csv(output / 'training_cost_by_depth.csv', index=False)
    for depth in (2, 3, 4):
        for seed in SEEDS:
            shutil.copyfile(Path(frozen[depth]['training_dir']) / f'seed_{seed}/history.csv',
                            histories / f'depth{depth}_seed{seed}.csv')
    summary = pd.concat([portfolio_summary(part).assign(depth=depth) for depth, part in monthly.groupby('depth')], ignore_index=True)
    summary.to_csv(output / 'financial_all_deciles.csv', index=False)
    main = classification.loc[classification.method.eq('ensemble') & classification.period.isin([p[0] for p in PERIODS])].copy()
    reference = main.loc[main.depth.eq(3)].set_index('period')
    for metric in ('cross_entropy', 'accuracy', 'mcc', 'roc_auc', 'brier'):
        main[f'{metric}_minus_depth3'] = main[metric] - main.period.map(reference[metric])
    main['parameters_per_member'] = main.depth.map({d: architecture_spec(d)['parameters'] for d in (2, 3, 4)})
    main.to_csv(output / 'ensemble_comparison.csv', index=False)
    # Pair individual members by seed, since their validation splits and normalizations match across depths.
    members = classification.loc[classification.method.isin([f'seed_{s}' for s in SEEDS]) & classification.period.isin([p[0] for p in PERIODS])].copy()
    ref = members.loc[members.depth.eq(3), ['period', 'method', 'cross_entropy', 'accuracy', 'mcc', 'roc_auc']]
    members = members.merge(ref, on=['period', 'method'], suffixes=('', '_depth3'), validate='many_to_one')
    for metric in ('cross_entropy', 'accuracy', 'mcc', 'roc_auc'):
        members[f'{metric}_minus_depth3'] = members[metric] - members[f'{metric}_depth3']
    members.to_csv(output / 'paired_seed_comparison.csv', index=False)
    hl = summary.loc[summary.portfolio.eq('H-L')].copy()
    hl.to_csv(output / 'financial_HL_comparison.csv', index=False)
    # Diagnostic matched-calendar financial summary if any depth omits different months due to tied bins.
    matched = []
    calendar_audit = []
    for weighting in ('EW', 'VW'):
        by_depth = {d: monthly.loc[monthly.depth.eq(d) & monthly.weighting.eq(weighting)] for d in (2, 3, 4)}
        common = set.intersection(*(set(part.Date) for part in by_depth.values()))
        for depth, part in by_depth.items():
            calendar_audit.append({'depth': depth, 'weighting': weighting, 'supported_dates': len(part),
                                   'common_dates': len(common), 'excluded_from_matched': len(part) - len(common)})
            if common:
                matched.append(portfolio_summary(part.loc[part.Date.isin(common)]).assign(depth=depth))
    pd.DataFrame(calendar_audit).to_csv(output / 'financial_calendar_audit.csv', index=False)
    pd.concat(matched, ignore_index=True).to_csv(output / 'financial_matched_calendar.csv', index=False) if matched else None
    plots(output, classification, main, hl, monthly, training, frozen)
    display = markdown_table(['Period', 'Depth', 'CE', 'Accuracy', 'MCC', 'AUC', 'Accuracy delta vs 3 (pp)'],
        [[r.period, r.depth, f'{r.cross_entropy:.6f}', f'{r.accuracy*100:.3f}%', f'{r.mcc:.4f}',
          f'{r.roc_auc:.4f}', f'{r.accuracy_minus_depth3*100:+.3f}'] for r in main.itertuples()])
    financial_table = markdown_table(['Period', 'Depth', 'Weight', 'Months', 'Annual arithmetic return', 'Gross Sharpe'],
        [[r.period, r.depth, r.weighting, r.observed_months, f'{r.annualized_arithmetic_mean*100:.2f}%',
          f'{r.sharpe_zero_rf_gross:.3f}'] for r in hl.itertuples()])
    report = f'''# I20/R20 二、三、四层结构统一对比

三个结构使用相同测试行、标签、逐年原始文件，以及配对 seed 的训练划分和标准化。
各结构均保留五个成员，等权平均上涨概率，阈值严格 >0.5。没有重新训练或根据测试结果选择结构。

{display}

CE、Brier 越低越好；Accuracy、MCC、AUC 越高越好。差值为该结构减三层；准确率差值以百分点计。
主结果为课程要求的 2000—2019 年，2001—2019 年仅是预先约定的共同年份补充。
逐 seed 差值见 paired_seed_comparison.csv；这些 seed 的验证集不同，不能平均其验证预测。

{financial_table}

金融结果使用 Ret_month，而分类标签使用 Ret_20d。收益为历史完整观测样本上的、未扣交易成本的月度比较；
H-L 为 100% 做多 High、100% 做空 Low。收益可观测性筛选可能产生选择偏差。
financial_calendar_audit.csv 报告各结构支持月份和共同月份；financial_matched_calendar.csv 是共同月份诊断。
年化收益为月均收益乘 12，不是复合年增长率。未计算显著性检验、换手和净收益。

层数增加也改变通道数和参数量，因此不能将差异全部归因为层数。测试差异是描述性结果，不能据此重新调阈值、选 seed 或改融合权重。
训练时间为日志记录时间之和，硬件和中断会影响计时。Grad-CAM 另见 gradcam/report_zh.md，解释的是固定 seed101 单模型。

主要图片：ensemble_metrics.png、accuracy_by_year.png、calibration_comparison.png、financial_HL.png、learning_curves.png。
各结构完整的十组投资组合、论文描述性对照和数据覆盖报告位于各自 depth_N/analysis/。
'''
    (output / 'report_zh.md').write_text(report, encoding='utf-8')
    write_json(output / 'complete.json', {'followup_id': read_json(root / 'followup.json')['followup_id'],
                                        'prediction_sha256': fingerprints, 'paired_rows': True})
    print(display, flush=True)
    print(f'COMPARE: COMPLETE | {output}', flush=True)


def plots(output, classification, main, hl, monthly, training, frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {2: '#3572A5', 3: '#D98829', 4: '#568D55'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ('cross_entropy', 'accuracy', 'roc_auc'), ('Cross-entropy (lower is better)', 'Accuracy', 'ROC AUC')):
        part = main.loc[main.period.eq('course_2000_2019')].sort_values('depth')
        ax.bar(part.depth.astype(str), part[metric], color=[colors[d] for d in part.depth])
        ax.set(title=title, xlabel='Convolutional blocks')
        for x, value in enumerate(part[metric]):
            ax.text(x, value, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, part[metric].max()*1.12)
    fig.suptitle('I20/R20: five-member ensembles, formation years 2000-2019')
    fig.savefig(output / 'ensemble_metrics.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for depth in (2, 3, 4):
        part = classification.loc[classification.depth.eq(depth) & classification.method.eq('ensemble') & classification.period.str.startswith('year_')].copy()
        part['year'] = part.period.str[5:].astype(int); part = part.sort_values('year')
        ax.plot(part.year, part.accuracy*100, label=f'Depth {depth}', color=colors[depth])
    prior = classification.loc[classification.depth.eq(3) & classification.method.eq('always_up') & classification.period.str.startswith('year_')].copy()
    ax.plot(prior.period.str[5:].astype(int), prior.accuracy*100, ls='--', c='gray', label='Always up')
    ax.set(xlabel='Formation year', ylabel='Accuracy (%)', title='Annual test classification'); ax.legend()
    fig.savefig(output / 'accuracy_by_year.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], '--', c='gray')
    for depth in (2, 3, 4):
        part = pd.read_csv(output.parent / f'depth_{depth}/analysis/calibration_bins.csv')
        part = part.loc[part.method.eq('p_ensemble') & part.samples.gt(0)]
        ax.plot(part.mean_probability, part.observed_up_fraction, marker='.', label=f'Depth {depth}', c=colors[depth])
    ax.set(xlabel='Mean predicted up probability', ylabel='Observed up fraction', xlim=(0, 1), ylim=(0, 1)); ax.legend()
    fig.savefig(output / 'calibration_comparison.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, weighting in zip(axes, ('EW', 'VW')):
        part = hl.loc[hl.period.eq('course_2000_2019') & hl.weighting.eq(weighting)].sort_values('depth')
        ax.bar(part.depth.astype(str), part.sharpe_zero_rf_gross, color=[colors[d] for d in part.depth])
        ax.axhline(0, c='gray', lw=.7); ax.set(title=f'{weighting}: monthly H-L', xlabel='Convolutional blocks', ylabel='Annualized gross Sharpe (zero rf)')
    fig.savefig(output / 'financial_HL.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, depth in zip(axes, (2, 3, 4)):
        for seed in SEEDS:
            part = pd.read_csv(Path(frozen[depth]['training_dir']) / f'seed_{seed}/history.csv')
            ax.plot(part.epoch, part.validation_cross_entropy, label=f'Seed {seed}')
        ax.set(title=f'Depth {depth}: validation CE', xlabel='Epoch', ylabel='Cross-entropy'); ax.legend(fontsize=8)
    fig.savefig(output / 'learning_curves.png', dpi=170); plt.close(fig)
