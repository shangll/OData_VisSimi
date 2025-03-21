#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2025.2.25
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
from matplotlib import gridspec
from matplotlib.pyplot import MultipleLocator


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

cateList = ['within','between']
activation_names = [
    'conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
res_output = set_filepath(rootPath,'res_all')
#
img_mtrx = pd.read_csv(
    os.path.join(res_output,'img_alex_simi.csv'),sep=',')
imgNames = img_mtrx.columns.tolist()
imgNames.remove('layer')
img_mtrx['image'] = imgNames*8
w2v_mtrx = pd.read_csv(
    os.path.join(res_output,'img_w2v_simi.csv'),sep=',')

#
# alexnet

#
cate1_list,cate2_list,subcate1_list,subcate2_list,\
    img1_list,img2_list,w2v1_list,w2v2_list,img_corr_list,\
    w2v_corr_list,lyr_list = [],[],[],[],[],[],[],[],[],[],[]
for lyr in activation_names:
    old_imgs = []
    for img1 in imgNames:
        old_imgs.append(img1)

        for img2 in imgNames:
            subcate1 = w2v_mtrx.loc[
                w2v_mtrx['image']==img1,
                'subcate'].values[0]
            subcate2 = w2v_mtrx.loc[
                w2v_mtrx['image']==img2,
                'subcate'].values[0]

            if subcate1 != subcate2:
                if img2 not in old_imgs:
                    img_corr = img_mtrx.loc[
                        (img_mtrx['layer']==lyr)&
                        (img_mtrx['image']==img1),img2].values[0]
                    w2v_corr = w2v_mtrx.loc[
                        w2v_mtrx['image']==img1,img2].values[0]
                    w2v1 = w2v_mtrx.loc[
                        w2v_mtrx['image']==img1,'name'].values[0]
                    w2v2 = w2v_mtrx.loc[
                        w2v_mtrx['image']==img2,'name'].values[0]
                    cate1 = w2v_mtrx.loc[
                        w2v_mtrx['image']==img1,'cate'].values[0]
                    cate2 = w2v_mtrx.loc[
                        w2v_mtrx['image']==img2,'cate'].values[0]
                    img_corr_list.append(img_corr)
                    w2v_corr_list.append(w2v_corr)
                    cate1_list.append(cate1)
                    cate2_list.append(cate2)
                    subcate1_list.append(subcate1)
                    subcate2_list.append(subcate2)
                    img1_list.append(img1)
                    img2_list.append(img2)
                    w2v1_list.append(w2v1)
                    w2v2_list.append(w2v2)
                    lyr_list.append(lyr)
    print(lyr)
corr_dat = pd.DataFrame(
    {'cate1':cate1_list,'cate2':cate2_list,
     'subcate1':subcate1_list,'subcate2':subcate2_list,
     'img1':img1_list,'img2':img2_list,
     'w2v1':w2v1_list,'w2v2':w2v2_list,
     'img_corr':img_corr_list,
     'w2v_corr':w2v_corr_list,
     'layer':lyr_list})
corr_dat['cond'] = 'within'
corr_dat.loc[
    corr_dat['cate1']!=corr_dat['cate2'],'cond'] = 'between'
corr_dat.to_csv(
    os.path.join(
        res_output,'corr_dat.csv'),
    mode='w',header=True,index=False)
corr_dat.shape[0]/8
#
#
#
corr_dat = pd.read_csv(
     os.path.join(res_output,'corr_dat.csv'),sep=',')
corr_dat_sub = corr_dat.groupby(
    ['layer','subcate1','subcate2','cond'])[
    ['img_corr','w2v_corr']].agg('mean').reset_index()
corr_dat_sub.to_csv(
    os.path.join(
        res_output,'corr_dat_sub.csv'),
    mode='w',header=True,index=False)
corr_dat_sub.shape[0]/8
#
corr_dat_sub = pd.read_csv(
     os.path.join(res_output,'corr_dat_sub.csv'),sep=',')
#
# Plot
y_top = 0.985
gs = gridspec.GridSpec(
    1,2,width_ratios=[2,1])
mpl.rcParams.update({'font.size':22})
fig= plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(gs[0,0])
sns.barplot(
    data=corr_dat_sub,
    x='layer',y='img_corr',hue='cond',
    hue_order=cateList,palette='Blues',
    errorbar='se',capsize=0.15,errcolor='grey',
    legend=True,ax=ax1)
ax1.set_xticks(activation_names,labels=np.arange(1,9))
ax1.set_xlabel('Layer')
ax1.set_ylabel('Similarity')
ax1.set_title('Visual',fontsize=21,fontweight='bold')
ax1.set_ylim(0.0,0.46)
y_major_locator = MultipleLocator(0.15)
ax1.yaxis.set_major_locator(y_major_locator)
h, _ = ax1.get_legend_handles_labels()
ax1.legend(
    h, cateList,loc='best',ncol=1,labelcolor=None,
    frameon=False).set_title(None)
#
ax2 = fig.add_subplot(gs[0,1])
sns.barplot(
    data=corr_dat_sub,
    x='cond',y='w2v_corr',hue='cond',
    hue_order=cateList,palette='Blues',
    errorbar='se',capsize=0.15,errcolor='grey',
    legend=False,ax=ax2)
ax2.set_xticks(cateList,labels=[])
ax2.set_xlabel('Category')
ax2.set_ylabel('')
ax2.set_title('Semantic',fontsize=21,fontweight='bold')
ax2.set_ylim(0.0,0.21)
y_major_locator = MultipleLocator(0.1)
ax2.yaxis.set_major_locator(y_major_locator)
fig.text(
    0.025,y_top,'A',ha='center',
    va='top',color='k',fontweight='bold')
fig.text(
    0.7,y_top,'B',ha='center',
    va='top',color='k',fontweight='bold')
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'corr_img_w2v.tif'))
plt.show(block=True)
plt.close('all')
