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
import scipy.stats as stats
from mne.stats import f_oneway,permutation_cluster_test,\
    permutation_cluster_1samp_test

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


Diona = ["#393A53","#DD9E5B","#D58784","#B69480","#EDCAC6"]
Kirara = ["#355D73","#8DC0C8","#D5C7AC","#EAC772","#69A94E"]
clist = sns.color_palette(Diona)[0:3]
# clist = sns.color_palette(Kirara)
save_tag = 1
p_crit = 0.05
activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
sizeList = [1,2,4,8]
condList = ['cate','simi','inter']
alexPath = set_filepath(rootPath,'res_alex')

# corr_tag = 'mean'
corr_tag = 'max'
glm_data = pd.read_csv(
    os.path.join(alexPath,'glm_coeff.csv'),sep=',')

for exp_tag in ['exp1b','exp2']:
    for n in sizeList:
        for k in condList:
            dat = glm_data.loc[
                (glm_data['exp']==exp_tag)&
                (glm_data['cond']==k)&
                (glm_data['setsize']==n),['layer','coeff']]
            X = np.array(
                [dat.loc[(dat['layer']==x_name),'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))
            tail = 0
            t_thresh = None
            n_permutations = 1000
            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('--- * --- * --- * --- * --- * ---')
            print('%s: setsize = %d, %s effect'%(exp_tag,n,k))
            print(clusters)
            print(p_values)
            print('--- * --- * --- * --- * --- * ---')

# Plot
mpl.rcParams.update({'font.size':20})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(18,12))
    ax = ax.ravel()
    for n,name in enumerate(activation_names):
        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[dat['layer']==name],x='setsize',y='coeff',
                     hue='cond',hue_order=condList,style='cond',
                     markers=['^','o','s'],dashes=False,palette=clist,
                     linewidth=2,markersize=10,err_style="bars",
                     errorbar=("se",1),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='MSS')
        ax[n].set_yticks(np.arange(-0.1,0.31,0.1))
        ax[n].set_ylabel(ylabel='Coefficients')
        ax[n].set_title(name)
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category','Similarity','Interaction'],
                 loc='upper left',ncol=1,fontsize=12,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d (%s)'%(k+1,corr_tag))
    sns.despine(offset=15,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_exp%d_coeff.png'%(corr_tag,k+1)))
    plt.show(block=True)
    plt.close('all')


# Plot
mpl.rcParams.update({'font.size':16})
fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(18,8))
ax = ax.ravel()
n = 0
for k,expN in zip(sizeList*2,['exp1b']*4+['exp2']*4):
    dat = glm_data[
        (glm_data['setsize']==k)&
        (glm_data['exp']==expN)&
        (glm_data['cond']!='intc')]

    if n==0:
        leg_tag = True
    else:
        leg_tag = False
    sns.lineplot(data=dat,x='layer',y='coeff',hue='cond',hue_order=condList,
                 markers=['^','o','s'],style='cond',dashes=False,palette=clist,
                 linewidth=2,markersize=10,err_style="bars",
                 errorbar=("se",1),legend=leg_tag,ax=ax[n])
    ax[n].set_xticks(range(0,8),labels=range(1,9))
    ax[n].set_yticks(np.arange(-0.1,0.31,0.1))
    ax[n].set_ylabel(ylabel='Coefficients')
    ax[n].set_title('%s MSS %d'%(expN,k))
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category','Similarity','Interaction'],
                 loc='upper left',ncol=1,fontsize=12,
                 frameon=False).set_title(None)
    n += 1
fig.suptitle('%s'%(corr_tag))
sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'MSS_%s_coeff.png'%(corr_tag)))
plt.show(block=True)
plt.close('all')


