#!/usr/bin/env python
#-*-coding:utf-8 -*-

# ch.4
# linlin.shang@donders.ru.nl


from config import set_filepath,rootPath

import os
from math import log
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import pingouin as pg
import statannot

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

exp_list = ['exp1b','exp2']
crit_sd = 3
tag_savefile = 0
res_output = set_filepath(rootPath,'res_all')
exp1b = pd.read_csv(
    os.path.join(res_output,'exp1b_Raw.csv'),sep=',')
exp1b['trialType'] = exp1b['cond']
exp1b.loc[exp1b['trialType']!='target','trialType'] = 'distractor'
exp1b.rename(columns={'schImg':'image'},inplace=True)
exp1b['imgName'] = exp1b['image'].str.split(
    '/',expand=True)[3].tolist()
exp1b['subcate'] = exp1b['image'].str.split(
    '/',expand=True)[2].tolist()

exp2 = pd.read_csv(
    os.path.join(res_output,'exp2_Raw.csv'),sep=',')
exp2.rename(columns={'testImg':'image'},inplace=True)
exp2['imgName'] = exp2['image'].str.split(
    '/',expand=True)[3].tolist()
exp2['subcate'] = exp2['image'].str.split(
        '/',expand=True)[2].tolist()
old_names = ['stdImg%d'%nameN for nameN in range(8)]
new_names = ['image%d'%nameN for nameN in range(1,9)]
exp2.rename(
    columns=dict(zip(old_names,new_names)),inplace=True)
for nameN,stim_name in enumerate(new_names):
    exp2['imgName%d'%(nameN+1)] = exp2[stim_name].str.split(
        '/',expand=True)[3].tolist()

exp_mean = pd.DataFrame()
for exp_tag in exp_list:
    if exp_tag=='exp1b':
        dat = exp1b
    else:
        dat = exp2

    incorr_points = len(dat[dat['acc']==0])
    # rt
    # # 2.1 <0.2 sec
    # dat.loc[dat['acc']==1,'acc'] = \
    #     np.where((dat.loc[(dat['acc']==1),'rt']<0.2),
    #              0,1)
    # 2.2 ±3 sd
    outRTs = dat[dat['acc']==1].copy(deep=True).groupby(
        ['subj','cond','setsize'])['rt'].transform(
        lambda x:stats.zscore(x))
    dat.loc[np.where(np.abs(outRTs)>crit_sd)[0],'acc'] = 0
    del_points = len(dat[dat['acc']==0])-incorr_points
    print('RT: delete %0.3f%% data points'%(del_points/len(dat)*100))

    # # acc
    # dat_3sd = dat[dat['acc']==1].groupby(
    #     ['subj','setsize'])[
    #     'rt'].agg('mean').reset_index()
    # acc_3sd = dat.groupby(
    #     ['subj','setsize'])[
    #     'acc'].agg('mean').reset_index()
    # dat_3sd['acc'] = acc_3sd['acc']
    # outSubj = dat_3sd.loc[(dat_3sd['acc']<0.7),'subj'].to_list()
    # dat = dat[~dat['subj'].isin(outSubj)]

    # mean data
    dat_mean = dat[dat['acc']==1].groupby(
        ['subj','setsize','cond'])[
        'rt'].agg('mean').reset_index()
    acc_mean = dat.groupby(
        ['subj','setsize','cond'])[
        'acc'].agg('mean').reset_index()
    dat_mean['acc'] = acc_mean['acc']
    dat_mean['exp'] = exp_tag

    #
    acc_mean_dat = dat_mean.groupby(
        ['subj','setsize'])[
        'acc'].agg('mean').reset_index()
    outACCs_mean = acc_mean_dat['acc'].transform(
        lambda x:stats.zscore(x))
    out_subjs_acc = list(
        set(acc_mean_dat.loc[
                np.where(np.abs(outACCs_mean)>crit_sd)[0],
                'subj'].values.tolist()))
    print('%s ourliers (ACC):'%exp_tag,out_subjs_acc)
    print('--- --- ---')

    dat = dat[~dat['subj'].isin(out_subjs_acc)]
    dat_mean = dat_mean[~dat_mean['subj'].isin(out_subjs_acc)]
    exp_mean = pd.concat(
        [exp_mean,dat_mean],axis=0,ignore_index=True)
    #

    # # outlier
    # dat_meanAll = dat_mean[dat_mean['acc']==1].groupby(
    #     ['subj'])[
    #     'rt'].agg('mean').reset_index()
    # acc_meanAll = dat_mean.groupby(
    #     ['subj'])[
    #     'acc'].agg('mean').reset_index()
    # dat_meanAll['acc'] = acc_meanAll['acc']
    # dat_meanAll['exp'] = exp_tag
    # outRTs_mean = dat_meanAll.groupby(
    #     ['exp'])['rt'].transform(
    #     lambda x:stats.zscore(x))
    # out_subjs_rt = list(
    #     set(dat_meanAll.loc[
    #             np.where(np.abs(outRTs_mean)>crit_sd)[0],
    #             'subj'].values.tolist()))
    # print('%s ourliers (RT)'%exp_tag,out_subjs_rt)
    # #
    # outACCs_mean = dat_meanAll.groupby(
    #     ['exp'])['acc'].transform(
    #     lambda x:stats.zscore(x))
    # out_subjs_acc = list(
    #     set(dat_meanAll.loc[
    #             np.where(np.abs(outACCs_mean)>crit_sd)[0],
    #             'subj'].values.tolist()))
    # print('%s ourliers (ACC):'%exp_tag,out_subjs_acc)
    # print('--- --- ---')

    # outSubj = out_subjs_rt+out_subjs_acc
    # dat = dat[~dat['subj'].isin(outSubj)]

    if tag_savefile==1:
        dat.to_csv(
            os.path.join(res_output,'%s_clean.csv'%exp_tag),
            mode='w',header=True,index=False)

if tag_savefile==1:
    exp_mean.to_csv(
        os.path.join(res_output,'exp_mean.csv'),
        mode='w',header=True,index=False)