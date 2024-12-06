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
import statsmodels.api as sm
from sklearn import preprocessing

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



Diona = ["#393A53","#DD9E5B","#D58784","#B69480","#EDCAC6"]
Kirara = ["#355D73","#8DC0C8","#D5C7AC","#EAC772","#69A94E"]
# clist = sns.color_palette(Diona)
alex_output = set_filepath(rootPath,'res_alex')
save_tag = 0
p_crit = 0.05
activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']
cateList = ['within','between']
exp_tags = ['exp1b','exp2']

simi_raw = pd.read_csv(
    os.path.join(alex_output,'expAll_simi_raw.csv'),sep=',')
simi_raw['cate_trans'] = np.where(
    simi_raw['cate']=='within',1,-1)

print(stats.pointbiserialr(simi_raw['cate_trans'],simi_raw['simi_mean']))
print(stats.pointbiserialr(simi_raw['cate_trans'],simi_raw['simi_max']))

# sns.lmplot(data=exp1b_simi,x='simi_mean',y='rt',
#            hue='cond',col='setsize',row='layer')
# plt.savefig(
#         os.path.join(figPath,'mean_overview_exp1b.png'))
# plt.show(block=True)
# plt.close('all')
# sns.lmplot(data=exp1b_simi,x='simi_max',y='rt',
#            hue='cond',col='setsize',row='layer')
# plt.savefig(
#         os.path.join(figPath,'max_overview_exp1b.png'))
# plt.show(block=True)
# plt.close('all')
# sns.lmplot(data=exp2_simi,x='simi_mean',y='rt',
#            hue='cond',col='setsize',row='layer')
# plt.savefig(
#         os.path.join(figPath,'mean_overview_exp2.png'))
# plt.show(block=True)
# plt.close('all')
# sns.lmplot(data=exp2_simi,x='simi_max',y='rt',
#            hue='cond',col='setsize',row='layer')
# plt.savefig(
#         os.path.join(figPath,'max_overview_exp2.png'))
# plt.show(block=True)
# plt.close('all')



# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ similarity)
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(['exp1b','exp2']):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff = [],[],[],[]

            # each MSS
            for n in sizeList:

                # each subject
                for k in exp_subj:
                    exp_simi_indv = exp[
                        (exp['layer']==name)&
                        (exp['setsize']==n)&
                        (exp['subj']==k)].copy()

                    # normalization (Z-score)
                    exp_simi_indv['rt_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'simi_%s'%corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])

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
            exp_glm_list.append(pd.DataFrame(
                {'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff_%s'%corr_tag:glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data_simi = df_glm_list[0]
glm_data_simi['coeff_max'] = df_glm_list[1]['coeff_max']

# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ Category)
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(['exp1b','exp2']):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff = [],[],[],[]

            # each MSS
            for n in sizeList:

                # each subject
                for k in exp_subj:
                    exp_simi_indv = exp[
                        (exp['layer']==name)&
                        (exp['setsize']==n)&
                        (exp['subj']==k)].copy()

                    # normalization (Z-score)
                    exp_simi_indv['rt_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'simi_%s'%corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])

                    # GLM fit
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv['cate_Z']
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    glm_cond.append('intc')
                    glm_coeff.append(model.params[0])
                    glm_cond.append('cate')
                    glm_coeff.append(model.params[1])
                    glm_subj += [k]*2
                    glm_size += [n]*2
            exp_glm_list.append(pd.DataFrame(
                {'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff_%s'%corr_tag:glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data_cate = df_glm_list[0]
glm_data_cate['coeff_max'] = df_glm_list[1]['coeff_max']

# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ Similarity + semi(Category))
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(['exp1b','exp2']):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff = [],[],[],[]

            # each MSS
            for n in sizeList:

                # each subject
                for k in exp_subj:
                    exp_simi_indv = exp[
                        (exp['layer']==name)&
                        (exp['setsize']==n)&
                        (exp['subj']==k)].copy()

                    # normalization (Z-score)
                    exp_simi_indv['rt_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'simi_%s'%corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])

                    # GLM fit
                    # 1st step
                    y = exp_simi_indv['simi_Z']
                    X = exp_simi_indv['cate_Z']
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    # 2nd step
                    y = exp_simi_indv['rt_Z']
                    X = model.predict()-exp_simi_indv['simi_Z'].values
                    X = sm.add_constant(X)
                    model2 = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()


                    glm_cond.append('intc')
                    glm_coeff.append(model2.params[0])
                    glm_cond.append('simi')
                    glm_coeff.append(model2.params[1])
                    glm_subj += [k]*2
                    glm_size += [n]*2
            exp_glm_list.append(pd.DataFrame(
                {'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff_%s'%corr_tag:glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data_simi_semi = df_glm_list[0]
glm_data_simi_semi['coeff_max'] = df_glm_list[1]['coeff_max']

# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ Category + semi(Similarity))
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(['exp1b','exp2']):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff = [],[],[],[]

            # each MSS
            for n in sizeList:

                # each subject
                for k in exp_subj:
                    exp_simi_indv = exp[
                        (exp['layer']==name)&
                        (exp['setsize']==n)&
                        (exp['subj']==k)].copy()

                    # normalization (Z-score)
                    exp_simi_indv['rt_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'simi_%s'%corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])

                    # GLM fit
                    # 1st step
                    y = exp_simi_indv['cate_Z']
                    X = exp_simi_indv['simi_Z']
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    # 2nd step
                    y = exp_simi_indv['rt_Z']
                    X = model.predict()-exp_simi_indv['cate_Z'].values
                    X = sm.add_constant(X)
                    model2 = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()


                    glm_cond.append('intc')
                    glm_coeff.append(model2.params[0])
                    glm_cond.append('cate')
                    glm_coeff.append(model2.params[1])
                    glm_subj += [k]*2
                    glm_size += [n]*2
            exp_glm_list.append(pd.DataFrame(
                {'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff_%s'%corr_tag:glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data_cate_semi = df_glm_list[0]
glm_data_cate_semi['coeff_max'] = df_glm_list[1]['coeff_max']



Diona = sns.color_palette(Diona)
c1,c2,c3,c4 = Diona[0],Diona[1],Diona[2],Diona[3]
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat1 = glm_data_cate[
        (glm_data_cate['exp']==exp_tag)&
        (glm_data_cate['cond']!='intc')]
    dat2 = glm_data_cate_semi[
        (glm_data_cate_semi['exp']==exp_tag)&
        (glm_data_cate_semi['cond']!='intc')]
    dat3 = glm_data_simi[
        (glm_data_simi['exp']==exp_tag)&
        (glm_data_simi['cond']!='intc')]
    dat4 = glm_data_simi_semi[
        (glm_data_simi_semi['exp']==exp_tag)&
        (glm_data_simi_semi['cond']!='intc')]

    fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(12,12))
    ax = ax.ravel()
    n = 0
    for corr_tag in ['mean','max']:
        for name in activation_names:
            if n==0:
                leg_tag = True
            else:
                leg_tag = False

            ax[n].axhline(0,color='tomato',lw=1,linestyle='dashed')

            sns.lineplot(data=dat1[dat1['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c1],
                         linewidth=2,markersize=5,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            sns.lineplot(data=dat2[dat2['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c2],
                         linewidth=2,markersize=5,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            sns.lineplot(data=dat3[dat3['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c3],
                         linewidth=2,markersize=5,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            sns.lineplot(data=dat4[dat4['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c4],
                         linewidth=2,markersize=5,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name+'(%s)'%corr_tag)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    fig.legend(h,['cate','cate(semi)','simi','simi(semi)'],
               loc='upper center',ncol=4,fontsize=9,
               frameon=False).set_title(None)
    ax[0].get_legend().remove()
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'compare_coeffCate_MSS_%s.png'%exp_tag))
    plt.show(block=True)
    plt.close('all')
#
# Plot
mpl.rcParams.update({'font.size':14})
fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(18,12))
ax = ax.ravel()
n = 0
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat1 = glm_data_cate[
        (glm_data_cate['exp']==exp_tag)&
        (glm_data_cate['cond']!='intc')]
    dat2 = glm_data_cate_semi[
        (glm_data_cate_semi['exp']==exp_tag)&
        (glm_data_cate_semi['cond']!='intc')]
    dat3 = glm_data_simi[
        (glm_data_simi['exp']==exp_tag)&
        (glm_data_simi['cond']!='intc')]
    dat4 = glm_data_simi_semi[
        (glm_data_simi_semi['exp']==exp_tag)&
        (glm_data_simi_semi['cond']!='intc')]

    for corr_tag in ['mean','max']:
        for sizeN in sizeList:
            ax[n].axhline(0,color='tomato',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat1[dat1['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c1],
                         linewidth=2,markersize=8,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            sns.lineplot(data=dat2[dat2['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c2],
                         linewidth=2,markersize=8,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            sns.lineplot(data=dat3[dat3['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c3],
                         linewidth=2,markersize=8,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            sns.lineplot(data=dat4[dat4['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=[c4],
                         linewidth=2,markersize=8,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s (%s) MSS%d'%(exp_tag,corr_tag,sizeN))
            n += 1
h,_ = ax[0].get_legend_handles_labels()
fig.legend(h,['cate','cate(semi)','simi','simi(semi)'],
           loc='upper center',ncol=4,fontsize=9,
           frameon=False).set_title(None)
ax[0].get_legend().remove()
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'compare_coeffCate_layer_expAll.png'))
plt.show(block=True)
plt.close('all')