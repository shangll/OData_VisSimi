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

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


alex_output = set_filepath(rootPath,'res_alex')
activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
sizeList = [1,2,4,8]
p_crit = 0.05


# glm_coeff = pd.read_csv(
#     os.path.join(alex_output,'glm_off_fit.csv'),sep=',')
glm_coeff = pd.read_csv(
    os.path.join(alex_output,'glm_coeff.csv'),sep=',')
# glm_coeff = pd.read_csv(
#     os.path.join(alex_output,'glm_2cate_coeff.csv'),sep=',')


# 1-sample ttest
pd.set_option('display.max_columns',None)
print('single-sample ttest: compare with 0')
for corr_tag in ['mean','max']:
    for exp_tag in ['exp1b','exp2']:
        for name in activation_names:
            print(corr_tag,exp_tag,name)
            print('--- --- --- --- --- ---')
            for n in sizeList:
                df_glm_exp = glm_coeff[
                    (glm_coeff['exp']==exp_tag)&
                    (glm_coeff['corr']==corr_tag)&
                    (glm_coeff['layer']==name)&
                    (glm_coeff['setsize']==n)]

                print('MSS == %d'%n)

                res = stats.ttest_1samp(
                    df_glm_exp.loc[df_glm_exp['cond']=='simi','coeff'].values,
                    popmean=0,alternative='two-sided')
                if res[1]<p_crit:
                    sig_tag = '*'
                else:
                    sig_tag = 'ns'
                print('similarity effect %s'%sig_tag)
                # print(res)
            print('---  ---  ---  ---  ---  ---')
#
# 1-sample permutation
from mne.stats import permutation_cluster_1samp_test
for exp_tag in ['exp1b','exp2']:
    for n in sizeList:
        for corr_tag in ['mean','max']:
            dat = glm_coeff.loc[
                (glm_coeff['exp']==exp_tag)&
                (glm_coeff['cond']=='simi')&
                (glm_coeff['setsize']==n),['layer','coeff']].copy()
            X = np.array(
                [dat.loc[(dat['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))
            tail = 0
            t_thresh = None
            n_permutations = 1000
            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('---  ---  ---  ---  ---  ---')
            print('%s (%s): MSS = %d'%(exp_tag,corr_tag,n))
            print(clusters[0][0]+1)
            print(p_values)
            print('---  ---  ---  ---  ---  ---')

# ind ttest
pd.set_option('display.max_columns',None)
print('ind ttest')
for exp_tag in ['exp1b','exp2']:
    for name in activation_names:
        print(exp_tag,name)
        print('--- --- --- --- --- ---')
        for n in [2,4,8]:
            # df_glm_exp = glm_coeff[
            #     (glm_coeff['exp']==exp_tag)&
            #     (glm_coeff['layer']==name)&
            #     (glm_coeff['setsize']==n)]
            df_glm_exp = glm_coeff[
                (glm_coeff['fit']=='log')&
                (glm_coeff['exp']==exp_tag)&
                (glm_coeff['layer']==name)]

            res = stats.ttest_ind(
                df_glm_exp.loc[df_glm_exp['cond']=='simi','coeff_max'].values,
                df_glm_exp.loc[df_glm_exp['cond']=='simi','coeff_mean'].values,
                alternative='two-sided')
            if res[1]<p_crit:
                sig_tag = '*'
            else:
                sig_tag = 'ns'
            print('MSS == %d %s'%(n,sig_tag))
            # print(res)
        print('--- * --- * --- * --- * --- * ---')

# ttest
pd.set_option('display.max_columns',None)
print('ind ttest')
for exp_tag in ['exp1b','exp2']:
    for name in activation_names:
        print(exp_tag,name)
        print('--- --- --- --- --- ---')
        for n in [2,4,8]:
            # df_glm_exp = glm_coeff[
            #     (glm_coeff['exp']==exp_tag)&
            #     (glm_coeff['layer']==name)&
            #     (glm_coeff['setsize']==n)]
            df_glm_exp = glm_coeff[
                (glm_coeff['fit']=='log')&
                (glm_coeff['exp']==exp_tag)&
                (glm_coeff['layer']==name)]

            res = stats.ttest_ind(
                df_glm_exp.loc[df_glm_exp['cond']=='MSS','coeff_max'].values,
                df_glm_exp.loc[df_glm_exp['cond']=='MSS','coeff_mean'].values,
                alternative='two-sided')
            if res[1]<p_crit:
                sig_tag = '*'
            else:
                sig_tag = 'ns'
            print('MSS == %d %s'%(n,sig_tag))
            # print(res)
        print('--- * --- * --- * --- * --- * ---')

# ttest
pd.set_option('display.max_columns',None)
print('ind ttest')
for exp_tag in ['exp1b','exp2']:
    for name in activation_names:
        print(exp_tag,name)
        print('--- --- --- --- --- ---')
        for n in [2,4,8]:
            # df_glm_exp = glm_coeff[
            #     (glm_coeff['exp']==exp_tag)&
            #     (glm_coeff['layer']==name)&
            #     (glm_coeff['setsize']==n)]

            df_glm_exp = glm_coeff[
                (glm_coeff['fit']=='log')&
                (glm_coeff['exp']==exp_tag)&
                (glm_coeff['layer']==name)]

            res = stats.ttest_ind(
                df_glm_exp.loc[df_glm_exp['cond']=='inter','coeff_max'].values,
                df_glm_exp.loc[df_glm_exp['cond']=='inter','coeff_mean'].values,
                alternative='two-sided')
            if res[1]<p_crit:
                sig_tag = '*'
            else:
                sig_tag = 'ns'
            print('MSS == %d %s'%(n,sig_tag))
            # print(res)
        print('---  ---  ---  ---  ---  ---')

