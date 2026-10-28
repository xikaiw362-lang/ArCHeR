"""New fixed-configuration paired RF-HIG study; archive existing method unchanged."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
import sys, shutil, hashlib, json, time, argparse, traceback
from pathlib import Path
ROOT=Path(r'D:\sc_dataset')
OUT=ROOT/'experiments/metrics/HIG500_paired_rerun_20260915_v1'
SRC=ROOT/'experiments/scripts'
parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
OUT.mkdir(exist_ok=True)
ARCH=OUT/'code_snapshot';ARCH.mkdir(exist_ok=True)
for name in ['method_archer.py','method_archer_configurable.py','method_archer_final.py','run_archer_ablation_multimetric_fair.py',Path(__file__).name]:
    target=ARCH/name
    if not target.exists(): shutil.copy2(SRC/name,target)
sys.path.insert(0,str(ARCH))
import method_archer_configurable as m
import method_archer_final as final
import numpy as np,pandas as pd,scanpy as sc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import run_archer_ablation_multimetric_fair as util

def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default),encoding='utf-8')
def matrix_hash(x):
    x=csr_matrix(x);h=hashlib.sha256()
    for a in [x.data,x.indices,x.indptr,np.asarray(x.shape)]:h.update(a.tobytes())
    return h.hexdigest()

params=final.get_default_params({'NEIGHBOR_BACKEND':'exact','HNSW_ENABLED':False,'NEIGHBOR_N_JOBS':2,'USE_TARGET_CLUSTER_COUNT':False})
datasets=['Quake_10x_Limb_Muscle','Quake_Smart-seq2_Heart']
manifest={'status':'running','method_version':final.METHOD_VERSION,'parameters':params,'seed_formula':'2026+100*r+rep','graph_stage_seed':'sample_seed+100000','design':'same upstream Full pipeline twice; use_hig=False only at final graph input in NoHIG','scope':'120 pairs on rare-cell subsamples, not 16 full datasets','mapping':'Hungarian maximum overlap, unmatched clusters majority mapped','code_hashes':{p.name:sha(p) for p in ARCH.glob('*.py')},'inputs':{},'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scanpy':sc.__version__},'note':'New experiment using current final defaults; not exact reproduction of historical mixed Fig3 runs.'}
for ds in datasets:
    p=ROOT/'data/benchmarks'/ds/'sc_cluster_fixed_result.h5ad'
    manifest['inputs'][ds]={'path':str(p),'sha256':sha(p)}
dump(OUT/'version_manifest.json',manifest)
state={}
graph_original=m.run_aknno_with_params
importance_original=m.compute_rf_hig_importance_with_params
neighbors_original=m.find_neighbors_aknn_with_params
def importance_hook(adata,p,labels):
    result=importance_original(adata,p,labels)
    state['views'].append((np.asarray(labels).copy(),result[0].copy()))
    return result
def neighbor_hook(*a,**kw):
    result=neighbors_original(*a,**kw)
    g=result.obsp['aknn_connectivities'].tocsr();n,c=connected_components(g,directed=False)
    deg=np.asarray((g!=0).sum(axis=1)).ravel()
    state['graph']={'connected_components':n,'isolate_fraction':float(np.mean(deg==0)),'largest_component_fraction':float(np.bincount(c).max()/len(c)),'degree_min':int(deg.min()),'degree_median':float(np.median(deg)),'degree_max':int(deg.max())}
    pd.DataFrame({'cell_id':state['ids'],'weighted_graph_degree':deg,'component':c}).to_csv(state['dir']/'graph_cells.csv',index=False)
    return result
def graph_hook(adata,p,use_hig=True):
    assert 'true_label' not in adata.obs
    hvg=np.asarray(adata.var.highly_variable,bool);hig=np.asarray(adata.var.is_HIG,bool)
    used=hvg|hig if state['variant']=='Full' else hvg
    state['hvg']=hvg.copy();state['hig']=hig.copy();state['hash']=matrix_hash(adata.X)
    frame=pd.DataFrame({'gene':adata.var_names,'HVG':hvg,'HIG':hig,'used_final':used})
    for i,(labels,imp) in enumerate(state['views']):
        frame['view%d_importance'%i]=imp;frame['view%d_rank'%i]=np.argsort(np.argsort(-imp))+1
        pd.DataFrame({'cell_id':adata.obs_names,'pseudo_label':labels}).to_csv(state['dir']/('view%d_labels.csv'%i),index=False)
    frame.to_csv(state['dir']/'genes.csv',index=False)
    util.set_case_seed(state['seed']+100000)
    return graph_original(adata,p,use_hig=state['variant']=='Full')
m.compute_rf_hig_importance_with_params=importance_hook
m.find_neighbors_aknn_with_params=neighbor_hook
m.run_aknno_with_params=graph_hook

records=[];sample_records=[]
for ds in datasets:
    a=sc.read_h5ad(manifest['inputs'][ds]['path'],backed='r')
    labels=pd.to_numeric(a.obs.true_label).astype(int).values
    major1,major2,hard,easy=util.pick_labels(labels)
    assert hard!=easy
    manifest['inputs'][ds].update({'classes':pd.Series(labels).value_counts().to_dict(),'major':[major1,major2],'challenging':hard,'easier':easy,'already_log':util.infer_already_log(a)})
    dump(OUT/'version_manifest.json',manifest)
    tasks=[(setting,target,r,rep) for setting,target in [('challenging',hard),('easier',easy)] for r in range(2,21,2) for rep in [1,2,3]]
    tasks.sort(key=lambda z:0 if z[0]=='challenging' and z[2:]==(4,1) else 1)
    for setting,target,r,rep in tasks:
        if args.smoke and (ds!=datasets[0] or (setting,r,rep)!=('challenging',4,1)):continue
        key='%s_%s_r%d_rep%d'%(ds,setting,r,rep);folder=OUT/'runs'/key;folder.mkdir(parents=True,exist_ok=True)
        seed=2026+100*r+rep
        for label,count in [(major1,200),(major2,200),(target,r)]:assert np.sum(labels==label)>=count
        ids=util.sample_cells(labels,major1,major2,target,200,r,seed)
        subset=a[ids].to_memory();y=labels[ids]
        assert len(ids)==400+r and len(np.unique(ids))==len(ids)
        cells=pd.DataFrame({'cell_id':subset.obs_names,'source_index':ids,'true_label':y})
        cell_file=folder/'cells.csv'
        if cell_file.exists():assert pd.read_csv(cell_file).cell_id.astype(str).tolist()==cells.cell_id.astype(str).tolist()
        else:cells.to_csv(cell_file,index=False)
        sample_records.append({'key':key,'dataset':ds,'setting':setting,'r':r,'rep':rep,'seed':seed,'sample_hash':sha(cell_file)})
        pd.DataFrame(sample_records).to_csv(OUT/'sample_manifest.csv',index=False)
        subset.obs=pd.DataFrame(index=subset.obs_names);subset.var=pd.DataFrame(index=subset.var_names)
        subset.uns.clear();subset.obsm.clear();subset.obsp.clear();subset.layers.clear();subset.raw=None
        subset.X=subset.X.astype(np.float32)
        paired=[]
        for variant in ['Full','NoHIG']:
            dest=folder/variant;dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():
                row=json.loads((dest/'result.json').read_text());records.append(row);paired.append(row);continue
            state.clear();state.update({'variant':variant,'seed':seed,'dir':dest,'views':[],'ids':subset.obs_names})
            util.set_case_seed(seed);start=time.time()
            print('START',key,variant,flush=True)
            try:
                pred,details=final.run_archer_final_pipeline(subset.copy(),true_labels=None,already_log=manifest['inputs'][ds]['already_log'],params=params,return_details=True,random_seed=seed)
                assert len(pred)==len(y) and len(state['views'])==2
                mapped=util.hungarian_match(y,pred)
                pd.DataFrame({'cell_id':subset.obs_names,'true_label':y,'pred_cluster':pred,'mapped_label':mapped}).to_csv(dest/'predictions.csv',index=False)
                pd.crosstab(pd.Series(y,name='true_label'),pd.Series(pred,name='pred_cluster')).to_csv(dest/'contingency.csv')
                sizes=[]
                for cl in np.unique(pred):
                    mask=pred==cl;nt=int(np.sum(y[mask]==target));sz=int(mask.sum())
                    sizes.append({'cluster':int(cl),'size':sz,'target_cells':nt,'target_purity':nt/sz,'target_recall':nt/r,'mapped_label':int(mapped[mask][0])})
                pd.DataFrame(sizes).to_csv(dest/'cluster_purity.csv',index=False)
                dump(dest/'diagnostics.json',{'details':details,'graph':state.get('graph',{}),'resolution':float(np.clip(params['RESOLUTION_BASE'],params['RESOLUTION_MIN'],params['RESOLUTION_MAX']))})
                row={'key':key,'dataset':ds,'setting':setting,'rare_count':r,'rep':rep,'seed':seed,'variant':variant,'target_label':int(target),'seconds':time.time()-start,'matrix_hash':state['hash'],'hvg_hash':hashlib.sha256(state['hvg'].tobytes()).hexdigest(),'hig_hash':hashlib.sha256(state['hig'].tobytes()).hexdigest(),'n_hvg':int(state['hvg'].sum()),'n_hig':int(state['hig'].sum()),'n_hig_added':int(np.sum(state['hig']&~state['hvg'])),'status':'ok'}
                row.update(util.compute_metrics(y,pred,target))
                assert row['rare_tp']+row['rare_fn']==r
                dump(dest/'result.json',row);records.append(row);paired.append(row)
                print('DONE',key,variant,'P/R/F1',row['Rare_Precision'],row['Rare_Recall'],row['Rare_F1'],'seconds',round(row['seconds'],2),flush=True)
            except Exception:
                (dest/'failure.txt').write_text(traceback.format_exc());manifest['status']='failed';dump(OUT/'version_manifest.json',manifest);raise
        for field in ['matrix_hash','hvg_hash','hig_hash']:assert paired[0][field]==paired[1][field],field
        pd.DataFrame(records).to_csv(OUT/'per_run_metrics.csv',index=False)
    a.file.close()
df=pd.DataFrame(records)
metrics=['ACC','ARI','NMI','Macro_F1','Rare_Precision','Rare_Recall','Rare_F1']
df.groupby(['dataset','setting','variant'])[metrics].mean().to_csv(OUT/'summary_by_scenario.csv')
df.groupby(['dataset','setting','rare_count','variant'])[metrics].mean().to_csv(OUT/'summary_by_scenario_and_abundance.csv')
parts=[]
for scope,frame in [('low_2_8',df[df.rare_count<=8]),('all_2_20',df)]:
    s=frame.groupby('variant')[metrics].mean().reset_index();s['scope']=scope;parts.append(s)
pd.concat(parts).to_csv(OUT/'summary.csv',index=False)
full=df[df.variant=='Full'].set_index('key');no=df[df.variant=='NoHIG'].set_index('key')
(full[metrics]-no[metrics]).to_csv(OUT/'paired_metrics.csv')
assert len(df)==(2 if args.smoke else 240)
manifest['status']='smoke_passed' if args.smoke else 'complete';manifest['completed_runs']=len(df);dump(OUT/'version_manifest.json',manifest)
print(manifest['status'],len(df),flush=True)
