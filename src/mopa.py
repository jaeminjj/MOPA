import sys
import numpy as np
import pandas as pd
from multiprocessing import Process
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from scipy import stats
from functools import partial
from tqdm import tqdm
import qnorm
## enrichment, UMAP ,HEATMAP ,survival analysis , accuracy
#ranks=0
#threshold=0.6
#genenum=0
##### inputdata - gmtfile,rank_selection , npy data, genelist
def reverse(data):
    ndata = data.to_numpy()
    ndata = ndata.T
    data = pd.DataFrame(data = ndata[:,:])
    return data


def gmt_to_binary(gmt,genelist):
    lines=gmt.readlines()
    pathway= pd.DataFrame(columns=range(0,len(lines)+1),index=range(0,len(genelist)))
    pathway=pathway.fillna(0)
    pathway.iloc[:,0]=genelist.iloc[:,1]
    pathway_name=[]
    for num,write in enumerate(lines):
        write = write.split("\t")
        temp=pd.DataFrame(data=write)
        pathway_name.append(temp.iloc[0,0])
        pathway.iloc[:,num+1][pathway.iloc[:,0].isin(temp.iloc[:,0])]=1
    gmt.close()
    pathway=pathway.iloc[:,1:]
    pathway.columns=pathway_name    
    return pathway

def gene_ranks_file(tensor,sample_info,gmt_binary,threshold): ##ranksample generater

    omics_tensor = pd.DataFrame(data=tensor[0])
    gene_tensor =pd.DataFrame(data=tensor[1])
    sample_tensor =pd.DataFrame(data=tensor[2])
    ##sample_info;;
    ##
    
    ranksample = pd.DataFrame(index=range(0,len(gene_tensor.iloc[:,0])), columns=(sample_info.iloc[:,0]))
    ranksample=ranksample.fillna(0)
    
    ### semi-supervised or unsupervised
    #if rank_select ==True:
    #ranks=ranks.drop_duplicates(ranks.columns[0],'first') ## rank selection
    #sample_tensor_selec=sample_tensor.iloc[:,ranks.iloc[:,0]]
    #omics_tensor_selec=omics_tensor.iloc[:,ranks.iloc[:,0]]
    #gene_tensor_selec=gene_tensor.iloc[:,ranks.iloc[:,0]]
    #elif rank_select == False:
    sample_tensor_selec=sample_tensor.iloc[:,:]
    gomics_tensor_selec=omics_tensor.iloc[:,:]
    gene_tensor_selec=gene_tensor.iloc[:,:]
    
    
    ###generating mES matrix
    mopa = pd.DataFrame(index=range(0,len(gmt_binary.iloc[0,:])), columns=(sample_info.iloc[:,0])) ##result file 
    mopa = mopa.fillna(0)
    
    ### normalization gene matirx
    scal = StandardScaler()
    gene_tensor_norm = StandardScaler().fit(gene_tensor_selec).transform(gene_tensor_selec)
    gene_tensor_norm =pd.DataFrame(data=gene_tensor_norm)
    minvalue=0
    for  num,i in enumerate(gene_tensor_norm.iloc[0,:]):
        if min(gene_tensor_norm.iloc[:,num])< minvalue :
            minvalue=min(gene_tensor_norm.iloc[:,num])
    gene_tensor_norm+=(abs(minvalue)+0.00000000000000001)
    
    
    
    ### rank feature selection using CDF
    point = pd.DataFrame(index=range(0,len(sample_tensor.iloc[:,0])), columns=range(0,len(sample_tensor_selec.iloc[0,:])))
    point=point.fillna(0)
    point.columns = sample_tensor_selec.columns
    
    
    ### normalization sample matrix
    sample_tensor_selec=qnorm.quantile_normalize(sample_tensor_selec, axis=1, ncpus=8)
    sample_tensor_norm=pd.DataFrame(data=sample_tensor_selec)
    
    
    
    ###selecting feature in sample matrix using CDF
    for numb,i in enumerate(sample_tensor_norm.index):
        estimator=stats.gaussian_kde(sample_tensor_norm.iloc[i,:], bw_method='silverman')
        X=np.array(sample_tensor_norm.iloc[i,:])
        C = [estimator.integrate_box_1d(-np.Inf,x) for x in X]
        for tttt,t in enumerate(C):
            if t>threshold:
              point.iloc[i,tttt]=2

    
    point1=reverse(point)
    gene_tensor_selec.columns = point1.index
    re_gene_tensor = reverse(gene_tensor_norm)
    sample_tensor_selec.columns = gene_tensor_norm.columns
    sample_tensor_selec.iloc[1,:].sort_values(ascending=False)
    d=0
    
    for i in range(len(sample_tensor_norm.iloc[:,0])):
        sample_tensor_norm.iloc[i,:]=sample_tensor_norm.iloc[i,:]/sample_tensor_norm.iloc[i,:].sum()

    
    
    
    for i in range(len(sample_tensor_selec.iloc[:,0])):
        c=point1[point1[i]==2].index
        d=0
        for col in c:
            a=sample_tensor_norm.iloc[i:i+1,col:col+1].to_numpy()
            b=gene_tensor_norm.iloc[:,col:col+1].to_numpy()
            d+=a*b
        ranksample.iloc[:,i]=d.copy()
    return ranksample,mopa
    
def gene_rank_file(tensor,ranks,sample_info,gmt_binary,threshold): ##ranksample generater
    
    omics_tensor = pd.DataFrame(data=tensor[0])
    gene_tensor =pd.DataFrame(data=tensor[1])
    sample_tensor =pd.DataFrame(data=tensor[2])
    ##sample_info;;
    ##
    
    ranksample = pd.DataFrame(index=range(0,len(gene_tensor.iloc[:,0])), columns=(sample_info.iloc[:,0]))
    ranksample=ranksample.fillna(0)
    
    ### semi-supervised or unsupervised
    #if rank_select ==True:
    ranks=ranks.drop_duplicates(ranks.columns[0],'first') ## rank selection
    sample_tensor_selec=sample_tensor.iloc[:,ranks.iloc[:,0]]
    omics_tensor_selec=omics_tensor.iloc[:,ranks.iloc[:,0]]
    gene_tensor_selec=gene_tensor.iloc[:,ranks.iloc[:,0]]
    #elif rank_select == False:
    #    sample_tensor_selec=sample_tensor.iloc[:,:]
    #    omics_tensor_selec=omics_tensor.iloc[:,:]
    #    gene_tensor_selec=gene_tensor.iloc[:,:]
    
    
    ###generating mES matrix
    mopa = pd.DataFrame(index=range(0,len(gmt_binary.iloc[0,:])), columns=(sample_info.iloc[:,0])) ##result file 
    mopa = mopa.fillna(0)
    
    ### normalization gene matirx
    scal = StandardScaler()
    gene_tensor_norm = StandardScaler().fit(gene_tensor_selec).transform(gene_tensor_selec)
    gene_tensor_norm =pd.DataFrame(data=gene_tensor_norm)
    minvalue=0
    for  num,i in enumerate(gene_tensor_norm.iloc[0,:]):
        if min(gene_tensor_norm.iloc[:,num])< minvalue :
            minvalue=min(gene_tensor_norm.iloc[:,num])
    gene_tensor_norm+=(abs(minvalue)+0.00000000000000001)
    
    
    
    ### rank feature selection using CDF
    point = pd.DataFrame(index=range(0,len(sample_tensor.iloc[:,0])), columns=range(0,len(sample_tensor_selec.iloc[0,:])))
    point=point.fillna(0)
    point.columns = sample_tensor_selec.columns
    
    
    ### normalization sample matrix
    sample_tensor_selec=qnorm.quantile_normalize(sample_tensor_selec, axis=1, ncpus=8)
    sample_tensor_norm=pd.DataFrame(data=sample_tensor_selec)
    
    
    
    ###selecting feature in sample matrix using CDF
    for numb,i in enumerate(sample_tensor_norm.index):
        estimator=stats.gaussian_kde(sample_tensor_norm.iloc[i,:], bw_method='silverman')
        X=np.array(sample_tensor_norm.iloc[i,:])
        C = [estimator.integrate_box_1d(-np.Inf,x) for x in X]
        for tttt,t in enumerate(C):
            if t>threshold:
              point.iloc[i,tttt]=2

    
    point1=reverse(point)
    gene_tensor_selec.columns = point1.index
    re_gene_tensor = reverse(gene_tensor_norm)
    sample_tensor_selec.columns = gene_tensor_norm.columns
    sample_tensor_selec.iloc[1,:].sort_values(ascending=False)
    d=0
    
    for i in range(len(sample_tensor_norm.iloc[:,0])):
        sample_tensor_norm.iloc[i,:]=sample_tensor_norm.iloc[i,:]/sample_tensor_norm.iloc[i,:].sum()

    
    
    
    for i in range(len(sample_tensor_selec.iloc[:,0])):
        c=point1[point1[i]==2].index
        d=0
        for col in c:
            a=sample_tensor_norm.iloc[i:i+1,col:col+1].to_numpy()
            b=gene_tensor_norm.iloc[:,col:col+1].to_numpy()
            d+=a*b
        ranksample.iloc[:,i]=d.copy()
    return ranksample,mopa
    
def enrichment(gmt_binary,ranksample,mopa):
    for i in tqdm(mopa.index):
        t=gmt_binary[gmt_binary.iloc[:,i]==0].index
        pathwaylen=len(gmt_binary[gmt_binary.iloc[:,i]==0])
        for jnum,j  in enumerate (mopa.columns):
        #for i in range(len(mopa.iloc[:,0])):
            c=ranksample.sort_values(by=j,ascending=False)
            c[c.index.isin(t)]=0
            d=np.array(c.loc[:,j])
            currentnot=0
            currentin=0
            max1=0
            min1=0
            sumc=sum(d)
            for k in d:
              if k ==0: ## notgeneset
                currentnot += 1.0/pathwaylen
              else: ## geneset
                currentin += k/sumc
        
              if currentin>=currentnot:
                if max1 <(currentin-currentnot):
                  max1= (currentin-currentnot)
              elif currentin <= currentnot:
                if min1 <(currentnot-currentin):
                  min1= (currentnot-currentin)
            mopa.loc[i,j]=max1-(min1)
    return mopa
    
    
def mopa_calculation(gmt_binary,ranksample,mopa,func,n_cores):
    df_split = np.array_split(mopa, n_cores,axis=0)
    pool = Pool(n_cores)
    func1 = partial(func,gmt_binary,ranksample)
    df = pd.concat(pool.map(func1, df_split),axis=0)
    pool.close()
    pool.join()
    df.index=gmt_binary.columns
    return df
def checknum(genenum,gmt_binary,mopa):
    checknum=mopa.iloc[:,0:2].copy()
    for i in range(len(mopa.iloc[:,0])):    
        genesetindex=gmt_binary[gmt_binary.iloc[:,i]> 0].index
        if len(genesetindex)<genenum:
            checknum.iloc[i,1]=1
        else:
            checknum.iloc[i,1]=0
    return checknum
def mopa_mes(tensor,sample_info,genelist,gmt,ranks,threshold,thread,genenum):
    gmt_binary=gmt_to_binary(gmt,genelist)
    if ranks !=0: # semi supervised
      ranksample,mopa=gene_rank_file(tensor,ranks,sample_info,gmt_binary,threshold)
    elif ranks==0:
      ranksample,mopa=gene_ranks_file(tensor,sample_info,gmt_binary,threshold)
    #t=mopa
    mopa=mopa_calculation(gmt_binary,ranksample,mopa,enrichment,thread)
    check=checknum(genenum,gmt_binary,mopa)          
    mopa=mopa.iloc[check[check.iloc[:,1]==0].index,:]
    return mopa
def mopa_OCR(tensor,gmt,ranks,genelist,sample_info,clinical_type,threshold):
    gmt_binary=gmt_to_binary(gmt,genelist)
    if ranks==0: ##unsupervised
        cyto,re=OCR(tensor,gmt_binary,genelist,sample_info,clinical_type,threshold)
    else: ## semi_supervised
        cyto,re=OCR_ip(tensor,gmt_binary,ranks,genelist,sample_info,clinical_type,threshold)
    return cyto,re
def OCR(tensor,gmt_binary,genelist,sample_info,clinical_type,threshold):
    groups=[]
    omics_tensor = pd.DataFrame(data=tensor[0])
    gene_tensor =pd.DataFrame(data=tensor[1])
    sample_tensor =pd.DataFrame(data=tensor[2])
    sample_tensor_selec=sample_tensor.iloc[:,:]
    gomics_tensor_selec=omics_tensor.iloc[:,:]
    gene_tensor_selec=gene_tensor.iloc[:,:]
    ### normalization gene matirx
    scal = StandardScaler()
    gene_tensor_norm = StandardScaler().fit(gene_tensor_selec).transform(gene_tensor_selec)
    gene_tensor_norm =pd.DataFrame(data=gene_tensor_norm)
    minvalue=0
    for  num,i in enumerate(gene_tensor_norm.iloc[0,:]):
        if min(gene_tensor_norm.iloc[:,num])< minvalue :
            minvalue=min(gene_tensor_norm.iloc[:,num])
    gene_tensor_norm+=(abs(minvalue)+0.00000000000000001)
    point = pd.DataFrame(index=range(0,len(sample_tensor.iloc[:,0])), columns=range(0,len(sample_tensor_selec.iloc[0,:])))
    point=point.fillna(0)
    point.columns = sample_tensor_selec.columns
    sample_tensor_selec=qnorm.quantile_normalize(sample_tensor_selec, axis=1, ncpus=8)
    sample_tensor_norm=pd.DataFrame(data=sample_tensor_selec)
    for numb,i in enumerate(sample_tensor_norm.index):
        estimator=stats.gaussian_kde(sample_tensor_norm.iloc[i,:], bw_method='silverman')
        X=np.array(sample_tensor_norm.iloc[i,:])
        C = [estimator.integrate_box_1d(-np.Inf,x) for x in X]
        for tttt,t in enumerate(C):
            if t>threshold:
              point.iloc[i,tttt]=2
    sample_info.columns =['sample','type']
    point2=pd.merge(point,sample_info['type'],left_index=True,right_index=True,how='left')
    groups.append(clinical_type)
    subtype_rank = pd.DataFrame(index=groups, columns=point2.columns[:-1])
    subtype_rank=subtype_rank.fillna(0)
    for num,group in enumerate (groups):
        point3=point2[point2['type']==group]
        for i in point3.columns[:-1]:    
            if point3[i].sum() >= len(point3)*2*0.5: ####more than 80%
                #print(num,i)
                subtype_rank.loc[group,i]=1
    re_subtype_rank=reverse(subtype_rank)
    re_subtype_rank.index=subtype_rank.columns
    re_subtype_rank.columns=subtype_rank.index
    gmt_pathway=gmt_binary.columns



    point4=pd.DataFrame(data=point2)
    re_subtype_rank1=pd.DataFrame(data=re_subtype_rank)
    rank1=[]
    rank1=re_subtype_rank[re_subtype_rank.loc[:,group]==1].index.to_list() ## group belong rank list

    cyto=pd.DataFrame(index=gmt_binary.columns, columns=range(0,3)) ## change to omics number
    cyto =cyto.fillna(0)
    for num,pathway in enumerate(gmt_binary.columns):
        pathway_gene=gmt_binary[gmt_binary[pathway]==1].index.to_list() ###index of gene belong in one pathway
        for i in gene_tensor.loc[pathway_gene,rank1].index:
            max1=-1000
            max2=0
                
            for k in rank1:
                  #gene_tensor
                if max1<=gene_tensor.loc[i,k]:
                      
                    max1=gene_tensor.loc[i,k]
                    max2=k
            if len(rank1)>0:
                a11=omics_tensor.loc[0,max2]
                b11=omics_tensor.loc[1,max2]
                c11=omics_tensor.loc[2,max2]
                sum11=a11+b11+c11
                cyto.iloc[num,0]+=a11/sum11
                cyto.iloc[num,1]+=b11/sum11
                cyto.iloc[num,2]+=c11/sum11
            else:
                cyto.iloc[num,0]='nan'
                cyto.iloc[num,1]='nan'
                cyto.iloc[num,2]='nan'
    cyto.columns = ['GE','methy','miRNA']
    reactom_name=pd.DataFrame
    gmt_name=pd.DataFrame(data=gmt_binary.columns)
    re=gmt_name[gmt_name[0].isin(cyto.index)].iloc[:,0:1]
    #re.to_csv('%s/re_percentage_v.txt'%(info.outdir),sep='\t')
    re.columns=['ID']
    re=re.reset_index(drop=True)
    re['name']=cyto.index
    cyto.index=re['ID']
    cyto['name']=cyto.index
    cyto=cyto.reset_index(drop=True)
    for i in range(0,len(cyto['name'])):
        cyto.iloc[i,3]=cyto.iloc[i,3].upper()
    #re=pd.merge(re,pval,how='left',left_on='name',right_index=True)
    return cyto,re
    
def OCR_ip(tensor,gmt_binary,ranks,genelist,sample_info,clinical_type,threshold):
    ranks=ranks.drop_duplicates(ranks.columns[0],'first')
    omics_tensor = pd.DataFrame(data=tensor[0])
    gene_tensor =pd.DataFrame(data=tensor[1])
    groups=[]
    sample_tensor =pd.DataFrame(data=tensor[2])
    sample_tensor_selec=sample_tensor.iloc[:,:]
    gomics_tensor_selec=omics_tensor.iloc[:,:]
    gene_tensor_selec=gene_tensor.iloc[:,:]
    ### normalization gene matirx
    scal = StandardScaler()
    gene_tensor_norm = StandardScaler().fit(gene_tensor_selec).transform(gene_tensor_selec)
    gene_tensor_norm =pd.DataFrame(data=gene_tensor_norm)
    minvalue=0
    for  num,i in enumerate(gene_tensor_norm.iloc[0,:]):
        if min(gene_tensor_norm.iloc[:,num])< minvalue :
            minvalue=min(gene_tensor_norm.iloc[:,num])
    gene_tensor_norm+=(abs(minvalue)+0.00000000000000001)
    point = pd.DataFrame(index=range(0,len(sample_tensor.iloc[:,0])), columns=range(0,len(sample_tensor_selec.iloc[0,:])))
    point=point.fillna(0)
    point.columns = sample_tensor_selec.columns
    sample_tensor_selec=qnorm.quantile_normalize(sample_tensor_selec, axis=1, ncpus=8)
    sample_tensor_norm=pd.DataFrame(data=sample_tensor_selec)
    for numb,i in enumerate(sample_tensor_norm.index):
        estimator=stats.gaussian_kde(sample_tensor_norm.iloc[i,:], bw_method='silverman')
        X=np.array(sample_tensor_norm.iloc[i,:])
        C = [estimator.integrate_box_1d(-np.Inf,x) for x in X]
        for tttt,t in enumerate(C):
            if t>threshold:
              point.iloc[i,tttt]=2
    sample_info.columns =['sample','type']
    point2=pd.merge(point,sample_info['type'],left_index=True,right_index=True,how='left')
    groups.append(clinical_type)
    subtype_rank = pd.DataFrame(index=range(0,len(groups)), columns=point2.columns[:-1])
    subtype_rank=subtype_rank.fillna(0)
    for num,group in enumerate (groups):
        point3=point2[point2['type']==group]
        for i in point3.columns[:-1]:    
            if point3[i].sum() >= len(point3)*2*0.5: ####more than 80%
                #print(num,i)
                subtype_rank.loc[num,i]=1
    re_subtype_rank=reverse(subtype_rank)
    re_subtype_rank.index=subtype_rank.columns
    re_subtype_rank.columns=subtype_rank.index
    gmt_pathway=gmt_binary.columns



    point4=pd.DataFrame(data=point2)
    re_subtype_rank1=pd.DataFrame(data=re_subtype_rank)
    rank1=[]
    rank1=re_subtype_rank[re_subtype_rank.loc[:,group]==1].index.to_list() ## group belong rank list

    cyto=pd.DataFrame(index=gmt_binary.columns, columns=range(0,3)) ## change to omics number
    cyto =cyto.fillna(0)
    for num,pathway in enumerate(gmt_binary.columns):
        pathway_gene=gmt_binary[gmt_binary[pathway]==1].index.to_list() ###index of gene belong in one pathway
        for i in gene_tensor.loc[pathway_gene,rank1].index:
            max1=-1000
            max2=0
                
            for k in rank1:
                  #gene_tensor
                if max1<=gene_tensor.loc[i,k]:
                      
                    max1=gene_tensor.loc[i,k]
                    max2=k
            if len(rank1)>0:
                a11=omics_tensor.loc[0,max2]
                b11=omics_tensor.loc[1,max2]
                c11=omics_tensor.loc[2,max2]
                sum11=a11+b11+c11
                cyto.iloc[num,0]+=a11/sum11
                cyto.iloc[num,1]+=b11/sum11
                cyto.iloc[num,2]+=c11/sum11
            else:
                cyto.iloc[num,0]='nan'
                cyto.iloc[num,1]='nan'
                cyto.iloc[num,2]='nan'
    cyto.columns = ['GE','methy','miRNA']
    re=reactom_name[reactom_name[1].isin(cyto.index)].iloc[:,0:1]
    #re.to_csv('%s/re_percentage_v.txt'%(info.outdir),sep='\t')
    re.columns=['ID']
    re=re.reset_index(drop=True)
    re['name']=cyto.index
    cyto.index=re['ID']
    #re=pd.merge(re,pval,how='left',left_on='name',right_index=True)
    return cyto,re      
