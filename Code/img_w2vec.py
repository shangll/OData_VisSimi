#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2024.02.23
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

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# file_tag = 'sgpt'
file_tag = 'w2v'
# alex_filepath = set_filepath(rootPath,'res_alex')
# sgpt_filepath = set_filepath(rootPath,'res_%s'%file_tag)
alex_output = set_filepath(rootPath,'res_all')

# simi_mtrx = pd.read_csv(os.path.join(sgpt_filepath,'img_sgpt_simi.csv'))
simi_mtrx = pd.read_csv(
    os.path.join(rootPath,'res_w2v','img_w2v_simi.csv'))
#
exp1b = pd.read_csv(
    os.path.join(alex_output,'exp1b_clean.csv'),sep=',')
exp1b_clean = exp1b[exp1b['acc']==1].copy(deep=True)
exp1b_clean.reset_index(drop=True,inplace=True)
exp1b_subj = list(set(exp1b_clean['subj']))
#
exp2 = pd.read_csv(
    os.path.join(alex_output,'exp2_clean.csv'),sep=',')
exp2_clean = exp2[exp2['acc']==1].copy(deep=True)
exp2_clean.reset_index(drop=True,inplace=True)
exp2_subj = list(set(exp2_clean['subj']))

sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']
cateList = ['within','between']
exp_tags = ['exp1b','exp2']

#
exp1b_copy = exp1b_clean.copy(deep=True)
# get the targets of each block
exp1b_copy['BlockN'] = 1
subjs,blocks,targets = [],[],[]
for subj in exp1b_subj:
    h = 1
    for n in sizeList:
        for cate in blockCate:
            exp1b_copy.loc[
                (exp1b_copy['subj']==subj)&
                (exp1b_copy['block']==cate)&
                (exp1b_copy['setsize']==n),'BlockN'] = h
            targs = exp1b_copy.loc[
                (exp1b_copy['subj']==subj)&
                (exp1b_copy['trialType']=='target')&
                (exp1b_copy['block']==cate)&
                (exp1b_copy['setsize']==n),
                'imgName'].tolist()
            targets += targs
            blocks += [h]*len(targs)
            subjs += [subj]*len(targs)
            h += 1
targImg_1b = pd.DataFrame(
    {'subj':subjs,'BlockN':blocks,'target':targets})
# get similarity between names
for subj in exp1b_subj:
    for n in range(1,9):
        distrImgs = exp1b_copy.loc[
            (exp1b_copy['subj']==subj)&
            (exp1b_copy['BlockN']==n)&
            (exp1b_copy['trialType']=='distractor'),
            'imgName'].tolist()
        targImgs = list(set(
            targImg_1b.loc[
                (targImg_1b['subj']==subj)&
                (targImg_1b['BlockN']==n),'target'].tolist()))
        for h,targImg in enumerate(targImgs):
            for distrImg in distrImgs:
                exp1b_copy.loc[
                    (exp1b_copy['subj']==subj)&
                    (exp1b_copy['BlockN']==n)&
                    (exp1b_copy['imgName']==distrImg),
                    'targ_%d'%(h+1)] = targImg
                exp1b_copy.loc[
                    (exp1b_copy['subj']==subj)&
                    (exp1b_copy['BlockN']==n)&
                    (exp1b_copy['imgName']==distrImg),
                    'simi_val_%d'%(h+1)] = simi_mtrx.loc[
                    simi_mtrx['image']==targImg,distrImg].values[0]
simi_cols = ['simi_val_%d'%h for h in range(1,9)]
exp1b_copy['w2v_mean'] = exp1b_copy[simi_cols].mean(axis=1)
exp1b_copy['w2v_max'] = exp1b_copy[simi_cols].max(axis=1)
exp1b_distr = exp1b_copy[
    exp1b_copy['trialType']=='distractor'].copy(deep=True)
exp1b_distr.reset_index(drop=True,inplace=True)

exp1b_distr.to_csv(
    os.path.join(
        alex_output,'exp1b_simi_%s.csv'%file_tag),
    mode='w',header=True,index=False)
print('exp.1b finished')
print('--- * --- * --- * --- * --- * ---')

#
exp2_copy = exp2_clean.copy(deep=True)
for n in range(exp2_copy.shape[0]):
    for k in range(1,9):
        targImg = exp2_copy.loc[n,'imgName']
        distrImg = exp2_copy.loc[n,'imgName%d'%k]
        if isinstance(distrImg,float):
            continue
        exp2_copy.loc[n,'simi_val_%d'%k] = simi_mtrx.loc[
                    simi_mtrx['image']==targImg,distrImg].values[0]
exp2_copy['w2v_mean'] = exp2_copy[simi_cols].mean(axis=1)
exp2_copy['w2v_max'] = exp2_copy[simi_cols].max(axis=1)
exp2_distr = exp2_copy[
    exp2_copy['trialType']=='distractor'].copy(deep=True)
exp2_distr.reset_index(drop=True,inplace=True)


exp2_distr.to_csv(
    os.path.join(
        alex_output,'exp2_simi_%s.csv'%file_tag),
    mode='w',header=True,index=False)
print('exp.2 finished')
print('--- * --- * --- * --- * --- * ---')

exp1b_distr = pd.read_csv(
    os.path.join(alex_output,'exp1b_simi_w2v.csv'),sep=',')
exp2_distr = pd.read_csv(
    os.path.join(alex_output,'exp2_simi_w2v.csv'),sep=',')
exp1b_distr['exp'] = 'exp1b'
exp2_distr['exp'] = 'exp2'
final_col = ['exp','subj','block','cond','setsize',
             'rt','w2v_mean','w2v_max']
exp_simi = pd.concat(
    [exp1b_distr[final_col],exp2_distr[final_col]],
    axis=0,ignore_index=True)
exp_simi.to_csv(
        os.path.join(
            alex_output,'expAll_simi_w2v.csv'),
    mode='w',header=True,index=False)

