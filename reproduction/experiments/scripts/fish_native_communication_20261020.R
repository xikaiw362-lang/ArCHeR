# Native zebrafish CellChat exploration; fixed ArCHeR groups, no identity relabelling.
# Stages: primary (100-bootstrap native network); sensitivity (fixed LR universe).
.libPaths(c('D:/Rlibs/4.5','D:/sc_dataset/paper/r_libs_seurat_20260817',.libPaths()))
suppressPackageStartupMessages({library(CellChat);library(Matrix);library(jsonlite)})
Sys.setenv(TZ='Asia/Shanghai')
args0<-commandArgs(trailingOnly=TRUE);stage<-if(length(args0))args0[1] else 'primary'
OUT<-'D:/sc_dataset/paper/fish_c21_c22_native_comm_20261020_v1'
dir.create(OUT,showWarnings=FALSE,recursive=TRUE)
for(d in c('source_data','models','logs','sensitivity'))dir.create(file.path(OUT,d),showWarnings=FALSE)
sink(file.path(OUT,'logs',paste0(stage,'.log')),split=TRUE)
options(future.globals.maxSize=2*1024^3)
# Sequential execution avoids this Windows R build's invalid native-encoding
# command line when future launches workers under the non-ASCII user profile.
future::plan('sequential')
log0<-function(...)cat(format(Sys.time(),'%Y-%m-%d %H:%M:%S'),..., '\n')
write0<-function(x,name,folder='source_data')write.csv(x,file.path(OUT,folder,name),row.names=FALSE)
FOCAL<-c('C21','C22')

focal_table<-function(cc){
 ar<-cc@net$prob;pv<-cc@net$pval;dn<-dimnames(ar)
 base<-expand.grid(source=dn[[1]],target=dn[[2]],stringsAsFactors=FALSE)
 keep<-base$source%in%FOCAL | base$target%in%FOCAL
 sizes<-table(cc@idents)
 keep<-keep & sizes[base$source]>=10 & sizes[base$target]>=10
 idx<-which(keep);lr<-cc@LR$LRsig
 out<-lapply(seq_len(dim(ar)[3]),function(k){
  z<-base[idx,,drop=FALSE];nm<-dn[[3]][k];j<-match(nm,rownames(lr))
  stopifnot(!is.na(j))
  z$interaction_name<-nm;z$ligand<-lr$ligand[j];z$receptor<-lr$receptor[j]
  z$pathway_name<-lr$pathway_name[j];z$prob<-as.vector(ar[,,k])[idx];z$pval<-as.vector(pv[,,k])[idx]
  z
 })
 do.call(rbind,out)
}

save_network<-function(cc,stem,full=FALSE){
 raw<-focal_table(cc);write0(raw,paste0(stem,'_focal_all_modeled_LR.csv'))
 cc<-filterCommunication(cc,min.cells=10)
 lr<-subsetCommunication(cc);write0(lr,paste0(stem,'_lr_nominal_p005.csv'))
 cc<-computeCommunProbPathway(cc);cc<-aggregateNet(cc)
 W<-cc@net$weight;C<-cc@net$count
 write.csv(W,file.path(OUT,'source_data',paste0(stem,'_net_weight.csv')))
 write.csv(C,file.path(OUT,'source_data',paste0(stem,'_net_count.csv')))
 role<-data.frame(cluster=rownames(W),outgoing_weight=rowSums(W),incoming_weight=colSums(W),
                  outgoing_n=rowSums(C),incoming_n=colSums(C))
 write0(role,paste0(stem,'_roles.csv'))
 write0(subsetCommunication(cc,slot.name='netP'),paste0(stem,'_pathway_nominal_p005.csv'))
 if(full)saveRDS(cc,file.path(OUT,'models',paste0(stem,'_cellchat.rds')))
 log0(stem,'network saved; nominal LR edges',nrow(lr),'pathways',length(cc@netP$pathways))
 cc
}

if(stage=='primary'){
 BASE<-'D:/sc_dataset/case_study_datasets/D_R_00002/exported'
 PRED<-'D:/sc_dataset/case_study_datasets/D_R_00002/case_reruns/archer_hig500_adopted_20260724/adopted_predictions_hig500.csv'
 log0('Read frozen original counts, native symbols, and labels')
 X<-as(readMM(file.path(BASE,'counts.mtx')),'CsparseMatrix')
 genes<-read.csv(file.path(BASE,'genes.csv'),stringsAsFactors=FALSE)[[1]]
 meta<-read.csv(file.path(BASE,'meta.csv'),stringsAsFactors=FALSE)
 pred<-read.csv(PRED,stringsAsFactors=FALSE)
 stopifnot(nrow(X)==length(genes),ncol(X)==nrow(meta),!anyDuplicated(genes),!anyDuplicated(meta$barcode),!anyDuplicated(pred$barcode))
 rownames(X)<-genes;colnames(X)<-meta$barcode
 cl<-setNames(pred$archer_cluster,pred$barcode)[colnames(X)]
 stopifnot(!anyNA(cl));cl<-paste0('C',cl)
 stopifnot(sum(cl=='C21')==39,sum(cl=='C22')==22,length(unique(cl))==23)
 metadata<-data.frame(barcode=colnames(X),cluster=cl,reference_fine=meta$Cell_type,reference_broad=meta$Big_Cell_type)
 write0(metadata,'frozen_query_metadata.csv')
 db<-CellChatDB.zebrafish
 getparts<-function(s){
  if(s%in%rownames(db$complex)){v<-as.character(unlist(db$complex[s,,drop=FALSE]));return(v[!is.na(v)&nzchar(v)])}
  s
 }
 cv<-db$interaction
 cv$ligand_subunits_present<-vapply(cv$ligand,function(s)all(getparts(s)%in%genes),logical(1))
 cv$receptor_subunits_present<-vapply(cv$receptor,function(s)all(getparts(s)%in%genes),logical(1))
 cv$symbol_covered<-cv$ligand_subunits_present & cv$receptor_subunits_present
 cv$database_row<-rownames(cv)
 write0(cv,'native_database_symbol_coverage.csv')
 contract<-list(date='2026-10-20',stage='native exploratory communication',question='Shared pigment background with distinct signaling preferences',
  fixed_cells=ncol(X),fixed_genes=nrow(X),fixed_clusters=23,focus=list(C21=39,C22=22),
  no_reference_driven_relabelling=TRUE,native_gene_case_preserved=TRUE,original_exact_duplicates=0,
  uppercase_collisions_in_previous_human_workflow=sum(duplicated(toupper(genes))),
  native_db_total=nrow(cv),native_db_symbol_covered=sum(cv$symbol_covered),
  package_version=as.character(packageVersion('CellChat')),database='CellChatDB.zebrafish',
  main_parameters=list(type='triMean',population.size=FALSE,raw.use=TRUE,nboot=100,seed.use=1,min.cells=10),
  sensitivity=list(smoothing='truncatedMean trim=0.1, nboot=100, same LR universe',
   balanced='20 deterministic seeds 101:120; C21 downsampled to22, C22 fixed22; other clusters unchanged; fixed LR universe',
   balanced_probability_only='nboot=1 computational setting; bootstrap p-values from these runs are not used for any inference'),
  independent_biological_units='Unknown: exported metadata has no donor/library fields; samples placeholder is not a biological replicate',
  inference_scope='Same-RNA exploratory predictions, not independent subtype validation or evidence of physical contact',
  old_human_run_preserved='D:/sc_dataset/paper/fish_cellchat_20261018',
  raw_inputs_md5=as.list(tools::md5sum(c(file.path(BASE,'counts.mtx'),file.path(BASE,'genes.csv'),file.path(BASE,'meta.csv'),PRED))))
 write_json(contract,file.path(OUT,'analysis_contract.json'),auto_unbox=TRUE,pretty=TRUE)
 norm<-CellChat::normalizeData(X,scale.factor=1e4)
 md<-data.frame(labels=cl,samples='unresolved_pool',row.names=colnames(X))
 cc<-createCellChat(object=norm,meta=md,group.by='labels');cc@DB<-db
 cc<-subsetData(cc)
 log0('Native signaling data',nrow(cc@data.signaling),'genes; overexpression screening')
 cc<-identifyOverExpressedGenes(cc,do.fast=FALSE)
 cc<-identifyOverExpressedInteractions(cc)
 write0(cc@LR$LRsig,'native_modeled_LR_universe.csv')
 saveRDS(cc,file.path(OUT,'models','native_prepared.rds'))
 log0('Compute native primary network; modeled LR',nrow(cc@LR$LRsig))
 cc<-computeCommunProb(cc,type='triMean',raw.use=TRUE,population.size=FALSE,nboot=100,seed.use=1)
 cc<-save_network(cc,'native_triMean',full=TRUE)
 write_json(list(completed=TRUE,stage='primary',time=format(Sys.time()),LR=nrow(subsetCommunication(cc))),
            file.path(OUT,'primary_completed.json'),auto_unbox=TRUE,pretty=TRUE)
 log0('Primary completed')
} else if(stage%in%c('sensitivity','balanced')){
 log0('Start outcome-independent sensitivity checks on saved native universe')
 prepared<-readRDS(file.path(OUT,'models','native_prepared.rds'))
 baseline<-readRDS(file.path(OUT,'models','native_triMean_cellchat.rds'))
 # This CellChat build leaves factor names unset on createCellChat. The
 # library's subsetCellChat indexes those names, so attach the verified
 # barcode order before subsetting (identity values and memberships unchanged).
 names(prepared@idents)<-colnames(prepared@data)
 names(baseline@idents)<-colnames(baseline@data)
 stopifnot(identical(names(prepared@idents),rownames(prepared@meta)))
 if(!file.exists(file.path(OUT,'models','native_truncatedMean10_cellchat.rds'))){
  cc<-computeCommunProb(prepared,type='truncatedMean',trim=.1,raw.use=TRUE,population.size=FALSE,nboot=100,seed.use=1)
  invisible(save_network(cc,'native_truncatedMean10',full=TRUE));rm(cc);gc()
 } else log0('Reuse completed truncatedMean10 model; do not rerun an unchanged sensitivity')
 log0('Balanced C21 subsamples; probability stability only, no replicate-based inference')
 # Verify that reducing the internal permutations changes p-values only,
 # never the unfiltered communication scores used in balanced diagnostics.
 probe<-computeCommunProb(prepared,type='triMean',raw.use=TRUE,population.size=FALSE,nboot=1,seed.use=1)
 gold<-read.csv(file.path(OUT,'source_data','native_triMean_focal_all_modeled_LR.csv'),stringsAsFactors=FALSE)
 test<-focal_table(probe)
 stopifnot(identical(unname(as.matrix(gold[,c('source','target','interaction_name')])),
                     unname(as.matrix(test[,c('source','target','interaction_name')]))))
 score_error<-max(abs(gold$prob-test$prob));stopifnot(score_error<1e-12)
 write_json(list(max_score_difference=score_error,probability_shortcut_validated=TRUE,pvalues_not_used=TRUE),
            file.path(OUT,'sensitivity','nboot1_score_equivalence.json'),auto_unbox=TRUE,pretty=TRUE)
 rm(probe,test,gold);gc(FALSE)
 selections<-list();barcodes0<-colnames(prepared@data)
 stopifnot(length(barcodes0)==length(prepared@idents),identical(barcodes0,colnames(baseline@data)))
 for(s in 101:120){
  set.seed(s);chosen<-sample(barcodes0[prepared@idents=='C21'],22,replace=FALSE)
  kept<-c(barcodes0[prepared@idents!='C21'],chosen)
  cc<-subsetCellChat(baseline,cells.use=kept)
  stopifnot(identical(rownames(cc@LR$LRsig),rownames(prepared@LR$LRsig)))
  stopifnot(sum(cc@idents=='C21')==22,sum(cc@idents=='C22')==22)
  cc<-computeCommunProb(cc,type='triMean',raw.use=TRUE,population.size=FALSE,nboot=1,seed.use=s)
  z<-focal_table(cc);z$pval<-NULL;z$seed<-s
  write.csv(z,file.path(OUT,'sensitivity',paste0('balanced_seed_',s,'_probability_only.csv')),row.names=FALSE)
  selections[[as.character(s)]]<-data.frame(seed=s,barcode=chosen)
  log0('Balanced seed completed',s,'rows',nrow(z));rm(cc,z);gc(FALSE)
 }
 write0(do.call(rbind,selections),'balanced_C21_cell_selections.csv',folder='sensitivity')
 write_json(list(completed=TRUE,seeds=101:120,smoothing='truncatedMean trim0.1',
  probability_only_bootstrap_pvalues_not_used=TRUE,time=format(Sys.time())),file.path(OUT,'sensitivity_completed.json'),auto_unbox=TRUE,pretty=TRUE)
 log0('Sensitivity completed')
} else if(stage=='expression'){
 log0('Audit native signaling expression; all available signaling genes form the exploratory test family')
 cc<-readRDS(file.path(OUT,'models','native_prepared.rds'))
 genes0<-rownames(cc@data.signaling)
 literature<-read.csv('D:/sc_dataset/paper/ArCHeR_fish_case_A4_20261017_v12/source_data/literature_marker_registry.csv',stringsAsFactors=FALSE)
 literature_names<-if('gene'%in%names(literature))literature$gene else character(0)
 allg<-unique(c(genes0,intersect(literature_names,rownames(cc@data))))
 labels0<-as.character(cc@idents);exp0<-cc@data[allg,,drop=FALSE]
 tab<-lapply(levels(cc@idents),function(cl0){
  k<-labels0==cl0;v<-exp0[,k,drop=FALSE]
  data.frame(gene=allg,cluster=cl0,n=sum(k),mean_log1p_CP10K=as.numeric(Matrix::rowMeans(v)),
   detected_n=as.numeric(Matrix::rowSums(v>0)),pct_detected=as.numeric(Matrix::rowMeans(v>0))*100,
   native_signaling_gene=allg%in%genes0)
 })
 write0(do.call(rbind,tab),'native_gene_expression_by_cluster.csv')
 a<-as.matrix(cc@data[genes0,labels0=='C21',drop=FALSE]);b0<-as.matrix(cc@data[genes0,labels0=='C22',drop=FALSE])
 rows<-lapply(seq_along(genes0),function(i){
  x<-a[i,];y<-b0[i,]
  pv<-if(all(x==0)&&all(y==0))1 else suppressWarnings(wilcox.test(x,y,alternative='two.sided',exact=FALSE,correct=TRUE)$p.value)
  if(is.na(pv))pv<-1
  data.frame(gene=genes0[i],n_C21=length(x),n_C22=length(y),
   mean_C21=mean(x),mean_C22=mean(y),mean_log1p_difference_C21_minus_C22=mean(x)-mean(y),
   detected_C21=sum(x>0),detected_C22=sum(y>0),p_raw=pv)
 })
 de<-do.call(rbind,rows);de$p_BH_native_signaling_family<-p.adjust(de$p_raw,method='BH')
 de$p_Bonferroni_native_signaling_family<-p.adjust(de$p_raw,method='bonferroni')
 de<-de[order(de$p_BH_native_signaling_family,-abs(de$mean_log1p_difference_C21_minus_C22)),]
 write0(de,'native_signaling_gene_C21_C22_exploratory_tests.csv')
 write_json(list(completed=TRUE,genes=length(genes0),test_family='All available native signaling genes, including unexpressed-in-focus genes',
  test='Two-sided asymptotic Wilcoxon rank sum with continuity correction',
  effect='Mean log1p(CP10K) C21 minus C22, NOT log2 fold change',
  limitation='Cell-level exploratory tests; no independent biological replicate inference',time=format(Sys.time())),
  file.path(OUT,'expression_completed.json'),auto_unbox=TRUE,pretty=TRUE)
 log0('Expression audit completed, family size',length(genes0),'BH<0.05',sum(de$p_BH_native_signaling_family<.05))
} else stop('Unknown stage')
future::plan('sequential')
writeLines(capture.output(sessionInfo()),file.path(OUT,'logs',paste0(stage,'_sessionInfo.txt')))
sink()
