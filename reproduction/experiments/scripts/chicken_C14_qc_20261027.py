# -*- coding: utf-8 -*-
"""Chicken C14 QC (v10): detected genes/UMIs and lymphoid vs erythroid marker detection (>=1 UMI)
per ArCHeR cluster and reference label. Counts come from the source Seurat object via
chicken_export_counts_20261027.R (the GEO .mtx.gz on disk is a truncated download)."""
import pandas as pd, numpy as np, scipy.io as sio, scipy.sparse as sp
D='D:/sc_dataset/case_study_datasets/G_G_00018/'
X=sio.mmread('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/.cache_gg18_counts/counts.mtx').T.tocsr()

genes=np.array(open('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/.cache_gg18_counts/genes.txt').read().split())
bc=np.array(open('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/.cache_gg18_counts/cells.txt').read().split())
meta=pd.read_csv(D+'source_scan_seurat_20260810/G_G_00018_SCAN_cell_metadata_20260810.csv').set_index('barcode').loc[bc]
fm=pd.read_csv('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/fig4_v2/source_data/fixed_metadata.csv').set_index('barcode').loc[bc]
cl=fm.archer_cluster.values
nc=np.asarray(X.sum(1)).ravel(); nf=np.asarray((X>0).sum(1)).ravel()
print('nCount match', np.allclose(nc,meta.nCount_RNA.values), 'nFeature match', np.allclose(nf,meta.nFeature_RNA.values))
gi={g:i for i,g in enumerate(genes)}
mt=[i for i,g in enumerate(genes) if g.startswith('MT-') or g in ('ND1','ND2','ND4','ND5','COX1','COX2','COX3','CYTB','ATP6','ATP8','ND3','ND4L','ND6')]
print('mt genes',[genes[i] for i in mt])
pmt=np.asarray(X[:,mt].sum(1)).ravel()/nc*100 if mt else np.zeros(len(nc))
sets={'T':['CD3E','CD3D','CD247','LCK','ZAP70','CD28','CD4','CTLA4','IL2RA'],'globin':['HBAD','HBA1','HBBA'],'RBC':['SLC4A1'],'thromb':['ITGA2B','GP9'],'chap':['HSPA8','HSP90AA1','DNAJB1','HSPA2','HSPH1'],'cycle':['MKI67','TOP2A'],'PTPRC':['PTPRC']}
cp=sp.diags(1e4/nc)@X; lg=cp.copy(); lg.data=np.log1p(lg.data)
def det(g): return np.asarray((X[:,gi[g]]>0).todense()).ravel()
def ex(g): return np.asarray(lg[:,gi[g]].todense()).ravel()
glob=np.asarray(cp[:,[gi[g] for g in sets['globin']]].sum(1)).ravel()   # CP10K globin
T_any=np.any([det(g) for g in ['CD3E','CD3D','CD247','LCK']],axis=0)
groups={'C14':cl==14}
ref=meta.shoudong.values
for lab in ['regulatory T cells','T cells','red blood cells','thrombocytes','B_cells']:
    groups['ref:'+lab+(' (not C14)' if lab=='regulatory T cells' else '')]=(ref==lab)&(cl!=14)
# ArCHeR cluster holding most RBCs
rb=pd.Series(cl[ref=='red blood cells']).value_counts(); print('RBC ref cells by ArCHeR cluster', rb.head(5).to_dict())
tr=pd.Series(cl[ref=='regulatory T cells']).value_counts(); print('Treg ref cells by ArCHeR cluster', tr.to_dict())
print('C14 ref labels', pd.Series(ref[cl==14]).value_counts().to_dict())
rows=[]
for k,m in groups.items():
    r={'group':k,'n':int(m.sum()),'nCount_med':np.median(nc[m]),'nFeature_med':np.median(nf[m]),'pctMT_med':np.median(pmt[m]),
       'globin_CP10K_med':np.median(glob[m]),'T_any_det%':100*T_any[m].mean(),'T+globin>=100 %':100*(T_any[m]&(glob[m]>=100)).mean()}
    for s in ['CD3E','CD3D','LCK','CD4','CTLA4','IL2RA','PTPRC','HBAD','HBA1','HBBA','SLC4A1','ITGA2B','GP9','HSPA8','HSP90AA1','DNAJB1','MKI67']:
        r[s+'%']=100*det(s)[m].mean()
    rows.append(r)
t=pd.DataFrame(rows).set_index('group').round(1); pd.set_option('display.width',250); print(t.T)
# rank of C14 nCount/nFeature among all ArCHeR clusters
cs=pd.DataFrame({'cl':cl,'nc':nc,'nf':nf}).groupby('cl').median(); cs['n']=pd.Series(cl).value_counts()
print(cs.sort_values('nf',ascending=False).head(8))
from scipy.stats import mannwhitneyu
o=groups['ref:regulatory T cells (not C14)']|groups['ref:T cells']
print('MWU nFeature C14 vs other T', mannwhitneyu(nf[cl==14],nf[o]).pvalue, 'nCount', mannwhitneyu(nc[cl==14],nc[o]).pvalue)
t.to_csv('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/manuscript_v10/source_data/chicken_C14_qc_summary.csv')
for g in ['TAL1','CA2','SLC25A37','TOP2A']:
    for k,m in groups.items(): t.loc[k,g+'%']=round(100*det(g)[m].mean(),1)
t.to_csv('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/manuscript_v10/source_data/chicken_C14_qc_summary.csv')
pd.crosstab(cl,ref).to_csv('D:/sc_dataset/paper/ArCHeR_figures_readable_20261026/manuscript_v10/source_data/chicken_cluster_by_reference_label.csv')
print(t[['n','nFeature_med','TAL1%','HBA1%','HBBA%','CD3E%','MKI67%','TOP2A%']])
