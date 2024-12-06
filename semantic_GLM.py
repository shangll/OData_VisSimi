#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2023.11.17
# linlin.shang@donders.ru.nl


from config import set_filepath,rootPath,figPath
import os
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from sklearn import preprocessing
from mne.stats import permutation_cluster_1samp_test,\
    permutation_t_test,permutation_cluster_test

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



# file_tag = 'sgpt'
file_tag = 'w2v'
sgpt_filepath = set_filepath(rootPath,'res_%s'%file_tag)
exp1b_distr = pd.read_csv(
    os.path.join(sgpt_filepath,'exp1b_simi_%s.csv'%file_tag))
exp1b_distr['exp'] = 'exp1b'
exp2_distr = pd.read_csv(
    os.path.join(sgpt_filepath,'exp2_simi_%s.csv'%file_tag))
exp2_distr['exp'] = 'exp2'

col_names = [
    'exp','subj','block','cond',
    'setsize','rt','sgpt_mean','sgpt_max']
sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']
cateList = ['within','between']
exp_tags = ['exp1b','exp2']
corr_tags = ['mean','max']
p_crit = 0.05
tail = 0
t_thresh = None
n_permutations = 1000

data_all = pd.concat(
    [exp1b_distr[col_names],exp2_distr[col_names]],axis=0,ignore_index=True)
data_all.rename(columns={'cond':'cate'},inplace=True)
data_all['cate_trans'] = np.where(
    data_all['cate']=='within',1,-1)



def plt_glm(dat,y_name,title_name,fig_name):
    clist = ['#FFBA00','grey']

    mpl.rcParams.update({'font.size':20})
    fig,ax = plt.subplots(1,1,figsize=(16,9))
    ax.axhline(0,color='black',lw=1,linestyle=':')

    sns.lineplot(
        data=dat,x='setsize',y=y_name,hue='exp',hue_order=exp_tags,
        style='exp',markers=True,dashes=False,palette=clist,
        linewidth=2,markersize=10,err_style='bars',
        errorbar=('se',0),legend=True,ax=ax)

    y_gap = 0.008
    y_sig = -0.05-y_gap
    for exp_tag in exp_tags:
        y_sig += y_gap
        dat_cond = dat[(dat['exp']==exp_tag)]
        X = np.array(
            [dat_cond.loc[(dat_cond['setsize']==sizeN),
            'coeff'].values for sizeN in sizeList])
        X = np.transpose(X,(1,0))

        t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
            X,n_jobs=None,threshold=t_thresh,adjacency=None,
            n_permutations=n_permutations,out_type='indices')

        if (len(clusters)!=0):
            for pN in range(len(p_values)):
                if (p_values[pN]<0.05):
                    sig_x = [sizeList[sizeN] for sizeN in list(clusters[pN][0])]
                    if exp_tag=='exp1b':
                        lcolor = clist[0]
                        lstyle = 'o'
                    else:
                        lcolor = clist[1]
                        lstyle = 'x'

                    ax.scatter(
                        sig_x,[y_sig]*len(sig_x),c=lcolor,
                        s=10,marker=lstyle)

    ax.set_xticks(sizeList,labels=sizeList)
    ax.set_xlabel(xlabel='Memory Set Size')
    ax.set_ylabel(ylabel='Beta')
    ax.set_title(title_name)
    h,_ = ax.get_legend_handles_labels()
    ax.legend(
        h,['LTM','STM'],
        loc='lower right',ncol=2,fontsize=12,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figPath,fig_name))
    plt.show(block=True)
    plt.close('all')



# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ semantic)
exp_glm_list = []
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each experiment
    for expN,exp_tag in enumerate(exp_tags):
        exp = data_all[data_all['exp']==exp_tag]
        exp_subj = list(set(exp['subj']))

        glm_subj,glm_size,glm_cond,glm_coeff,glm_corr,glm_exp = \
            [],[],[],[],[],[]

        # each MSS
        for n in sizeList:

            # each subject
            for k in exp_subj:
                exp_simi_indv = exp[
                    (exp['setsize']==n)&
                    (exp['subj']==k)].copy()

                # normalization (Z-score)
                exp_simi_indv['rt_Z'] = preprocessing.scale(
                    exp_simi_indv.loc[:,'rt'])
                exp_simi_indv['simi_Z'] = preprocessing.scale(
                    exp_simi_indv.loc[:,'sgpt_%s'%corr_tag])

                # GLM fit
                y = exp_simi_indv['rt_Z']
                X = exp_simi_indv['simi_Z']
                X = sm.add_constant(X)
                model = sm.GLM(
                    y,X,family=sm.families.Gaussian()).fit()

                glm_cond.append('intc')
                glm_coeff.append(model.params[0])
                glm_cond.append('simi')
                glm_coeff.append(model.params[1])
                glm_subj += [k]*2
                glm_size += [n]*2
                glm_corr += [corr_tag]*2
                glm_exp += [exp_tag]*2
        exp_glm_list.append(pd.DataFrame(
            {'corr':glm_corr,'exp':glm_exp,'subj':glm_subj,
             'setsize':glm_size,'cond':glm_cond,'coeff':glm_coeff}))

glm_data = pd.concat(exp_glm_list,axis=0,ignore_index=True)
glm_data = glm_data[glm_data['cond']=='simi']

glm_data.to_csv(os.path.join(sgpt_filepath,'glm_rt-%s.csv'%file_tag),
                sep=',',mode='w',header=True,index=False)
for corr_tag in corr_tags:
    title_name = 'Semantic Similarity Effect'
    fig_name = 'glm_rt-simi_%s_%s.tif'%(file_tag,corr_tag)
    plt_glm(glm_data[glm_data['corr']==corr_tag],'coeff',title_name,fig_name)

# #
# alex_output = set_filepath(rootPath,'res_alex')
# simi_raw = pd.read_csv(
#     os.path.join(alex_output,'expAll_simi_raw.csv'),sep=',')
# simi_raw['cate_trans'] = np.where(
#     simi_raw['cate']=='within',1,-1)