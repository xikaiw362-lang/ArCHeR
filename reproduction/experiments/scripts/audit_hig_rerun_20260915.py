"""Independently recalculate metrics from saved predictions and verify pairing."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
ROOT=Path(r'D:\sc_dataset\experiments\metrics\HIG500_paired_rerun_20260915_v1')
sys.path.insert(0,str(ROOT/'code_snapshot'))
import run_archer_ablation_multimetric_fair as u
rows=[]; errors=[]
for case in sorted((ROOT/'runs').iterdir()):
    cells=pd.read_csv(case/'cells.csv'); target=None
    fs={}
    for variant in ['Full','NoHIG']:
        path=case/variant
        if not (path/'result.json').exists():continue
        result=json.loads((path/'result.json').read_text());target=result['target_label']
        pred=pd.read_csv(path/'predictions.csv'); genes=pd.read_csv(path/'genes.csv');fs[variant]=genes
        assert cells.cell_id.astype(str).tolist()==pred.cell_id.astype(str).tolist()
        assert np.array_equal(cells.true_label.values,pred.true_label.values)
        recomputed=u.compute_metrics(pred.true_label.values,pred.pred_cluster.values,target)
        for k,val in recomputed.items():
            if abs(float(val)-float(result[k]))>1e-10:errors.append((case.name,variant,k))
        assert np.array_equal(genes.used_final.values,(genes.HVG|genes.HIG).values if variant=='Full' else genes.HVG.values)
        assert len(genes)==genes.gene.nunique()
        rows.append(result)
    if len(fs)==2:
        for col in ['gene','HVG','HIG','view0_importance','view1_importance']:
            assert np.array_equal(fs['Full'][col].values,fs['NoHIG'][col].values),str(case)+col
assert not errors,errors
df=pd.DataFrame(rows);assert len(df)==240
keys=['dataset','setting','rare_count','rep']
assert not df.duplicated(keys+['variant']).any()
full=df[df.variant=='Full'].set_index(keys);no=df[df.variant=='NoHIG'].set_index(keys)
assert set(full.index)==set(no.index) and len(full)==120
metrics=['Rare_Precision','Rare_Recall','Rare_F1','ACC','ARI','NMI','Macro_F1']
lines=['# HIG500配对消融重跑结果','',
'本次为当前最终入口HIG500的新受控抽样实验，两个数据集、四场景、120对/240次运行。不是16数据集benchmark，不是历史图3的DPC输入扩展比较。',
'','## 配置与可比性','',
'HVG=2000，RF-HIG预算500，双视角best-rank capped union，图维数上限30，Kmax=20，SNN剪枝1/15，exact近邻，两组图阶段种子相同。Full和NoHIG都计算相同的上游伪标签/HIG用于审计，只有最终图使用的特征掩码不同。因此运行时间不作为跳过HIG训练的效率比较。',
'','## 低丰度和完整丰度','',
'|范围|版本|平均Precision|平均Recall|平均F1|TP累计|FP累计|累计Precision|','|---|---|---:|---:|---:|---:|---:|---:|']
for label,rmax in [('2–8细胞',8),('2–20细胞',20)]:
 for variant in ['Full','NoHIG']:
  x=df[(df.variant==variant)&(df.rare_count<=rmax)];tp=int(x.rare_tp.sum());fp=int(x.rare_fp.sum())
  lines.append('|%s|%s|%.4f|%.4f|%.4f|%d|%d|%.4f|'%(label,variant,x.Rare_Precision.mean(),x.Rare_Recall.mean(),x.Rare_F1.mean(),tp,fp,tp/(tp+fp) if tp+fp else 0))
lines+=['','## 四场景低丰度结果','','|数据集/场景|版本|Precision|Recall|F1|','|---|---|---:|---:|---:|']
for (ds,setting,variant),x in df[df.rare_count<=8].groupby(['dataset','setting','variant']):
 lines.append('|%s / %s|%s|%.4f|%.4f|%.4f|'%(ds,setting,variant,x.Rare_Precision.mean(),x.Rare_Recall.mean(),x.Rare_F1.mean()))
lines+=['','## 配对变化','','|范围|指标|Full减NoHIG均值|Full胜/平/负|','|---|---|---:|---|']
delta=full[metrics]-no[metrics]
for label,rmax in [('2–8',8),('2–20',20)]:
 x=delta[delta.index.get_level_values('rare_count')<=rmax]
 for m in metrics:
  v=x[m];lines.append('|%s|%s|%.5f|%d / %d / %d|'%(label,m,v.mean(),(v>1e-10).sum(),(abs(v)<=1e-10).sum(),(v< -1e-10).sum()))
lines+=['','## 验收','',
'240份预测重新计算的全部指标与运行结果一致。120对cell-ID、HVG和上游HIG重要性一致，NoHIG最终掩码确为仅HVG。无缺失配对，原始记录不覆盖。',
'','累计TP/FP来自重复抽样，并非独立生物学重复。不进行基于120个独立数据集假设的显著性推断。Hungarian一对一匹配后对多余簇采用多数类映射，仍可能将低纯度簇映射为目标，需结合保存的cluster_purity和contingency文件解释。',
'','## 特征量','',
'Full新增非HVG基因数：最小%d，中位数%.1f，最大%d。'%(df[df.variant=='Full'].n_hig_added.min(),df[df.variant=='Full'].n_hig_added.median(),df[df.variant=='Full'].n_hig_added.max())]
(ROOT/'HIG消融结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
dump={'verified_runs':len(df),'pairs':len(full),'prediction_metric_mismatches':0,'paired_feature_checks':True}
(ROOT/'audit.json').write_text(json.dumps(dump,indent=2),encoding='utf-8')
print('\n'.join(lines))
