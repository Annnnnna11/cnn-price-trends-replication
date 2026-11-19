"""Publication-exportable figures from the reviewed yearly CSVs."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
classification=pd.read_csv(HERE/'yearly_classification.csv')
finance=pd.read_csv(HERE/'yearly_financial.csv')
OUTPUT=HERE.parent/'runs/review/figures'
OUTPUT.mkdir(parents=True,exist_ok=True)
colors={2:'#3572A5',3:'#D98829',4:'#568D55'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.facecolor':'white'})
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
for depth in (2,3,4):
    part=classification.loc[classification.depth.eq(depth)].sort_values('year')
    axes[0].plot(part.year,part.accuracy_minus_always_up_pp,marker='.',label=f'Depth {depth}',c=colors[depth])
    axes[1].plot(part.year,part.roc_auc,marker='.',label=f'Depth {depth}',c=colors[depth])
    axes[2].plot(part.year,part.predicted_up_fraction*100,marker='.',label=f'Depth {depth}: predicted up',c=colors[depth])
reference=classification.loc[classification.depth.eq(3)].sort_values('year')
axes[2].plot(reference.year,reference.positive_fraction*100,c='#333333',lw=2,ls='--',label='Observed up share')
axes[0].axhline(0,c='gray',ls='--',lw=1)
axes[0].set(title='Accuracy gain over always-up (does not imply beating always-nonup)',ylabel='Percentage points')
axes[0].legend(ncol=3,loc='upper left')
axes[1].axhline(.5,c='gray',ls='--',lw=1)
axes[1].set(title='Within-year ROC AUC',ylabel='AUC',ylim=(.495,.58))
axes[2].set(title='Class frequencies at the fixed 0.5 threshold (not a probability calibration plot)',
            ylabel='Share (%)',xlabel='Formation year')
axes[2].legend(ncol=2,loc='lower right',fontsize=9)
axes[2].set_xticks(np.arange(2000,2020)); axes[2].tick_params(axis='x',labelrotation=45)
fig.suptitle('I20/R20: yearly diagnostics for five-member ensembles',fontsize=14)
fig.savefig(OUTPUT/'yearly_classification_diagnostics.png',dpi=180)
fig.savefig(OUTPUT/'yearly_classification_diagnostics.svg')
plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(12,7.8),sharex=True,layout='constrained')
years=np.arange(2000,2020); x=np.arange(len(years))
for axis,weighting in zip(axes,('EW','VW')):
    for depth in (2,3,4):
        part=finance.loc[finance.depth.eq(depth)&finance.weighting.eq(weighting)].set_index('year').loc[years]
        axis.bar(x+(depth-3)*.25,part.annualized_arithmetic_mean*100,width=.24,capsize=0,
                 color=colors[depth],label=f'Depth {depth}',zorder=3)
    axis.axhline(0,c='gray',lw=1)
    axis.set(title=f'{weighting}: gross High-minus-Low',ylabel='12 x monthly mean (%)')
    axis.legend(ncol=3,loc='upper right')
axes[-1].set_xticks(x,[str(y)+('*' if y==2019 else '') for y in years],rotation=45)
axes[-1].set_xlabel('Formation year; *2019 has 11 observed formation months')
fig.suptitle('Yearly financial comparison: annualized arithmetic mean, not CAGR\nHistorical complete cases; before costs; 100% long High and 100% short Low',fontsize=12)
fig.savefig(OUTPUT/'yearly_financial_comparison.png',dpi=180)
fig.savefig(OUTPUT/'yearly_financial_comparison.svg')
plt.close(fig)
print('Yearly figures saved (PNG and SVG).')
