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
                        y,X,offset=exp_simi_indv['cate_Z'],
                        family=sm.families.Gaussian()).fit()

                    simi_raw.loc[
                        (simi_raw['exp']==exp_tag)&
                        (simi_raw['layer']==name)&
                        (simi_raw['setsize']==n)&
                        (simi_raw['subj']==k),'resid_%s'%corr_tag] = \
                        model.predict()-exp_simi_indv['rt_Z'].values

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
glm_data = df_glm_list[0]
glm_data['coeff_max'] = df_glm_list[1]['coeff_max']
final_col = ['exp','subj','block','cate','setsize',
             'rt','layer','simi_mean','simi_max',
             'resid_mean','resid_max']
simi_data = simi_raw[final_col]

if save_tag==1:
    simi_data.to_csv(
        os.path.join(alex_output,'expAll_off_simi.csv'),
        mode='w',header=True,index=False)
    glm_data.to_csv(os.path.join(alex_output,'glm_off_coeff.csv'),
                  sep=',',mode='w',header=True,index=False)


condList = ['cate','simi']
clist = ['dodgerblue']
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(12,12))
    ax = ax.ravel()
    n = 0
    for corr_tag in ['mean','max']:
        for name in activation_names:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Similarity (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Similarity (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'off_coeff_MSS_%s.png'%exp_tag))
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
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    for corr_tag in ['mean','max']:
        for sizeN in sizeList:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s MSS%d'%(exp_tag,sizeN))
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Similarity (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[4].legend(h,['Similarity (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Similarity (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[12].legend(h,['Similarity (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'off_coeff_layer_expAll.png'))
plt.show(block=True)
plt.close('all')


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
                        y,X,offset=exp_simi_indv['simi_Z'],
                        family=sm.families.Gaussian()).fit()

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
glm_data = df_glm_list[0]
glm_data['coeff_max'] = df_glm_list[1]['coeff_max']

if save_tag==1:
    glm_data.to_csv(os.path.join(alex_output,'glm_off_coeff_cate.csv'),
                  sep=',',mode='w',header=True,index=False)

condList = ['cate','simi']
# clist = sns.color_palette(Diona)
clist = ['tomato']
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(12,12))
    ax = ax.ravel()
    n = 0
    for corr_tag in ['mean','max']:
        for name in activation_names:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')
            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Similarity (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'off_coeffCate_MSS_%s.png'%exp_tag))
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
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    for corr_tag in ['mean','max']:
        for sizeN in sizeList:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s MSS%d'%(exp_tag,sizeN))
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[4].legend(h,['Category (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Similarity (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[12].legend(h,['Similarity (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'off_coeffCate_layer_expAll.png'))
plt.show(block=True)
plt.close('all')


# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ similarity * MSS)
fit_subj,fit_size,fit_raw,fit_pred,fit_resid,fit_exp,fit_corr,fit_layer,fit_tagList = \
    [],[],[],[],[],[],[],[],[]
glm_data_all = pd.DataFrame()
for fit_tag in ['linear','log']:
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

                glm_subj,glm_cond,glm_coeff = [],[],[]

                # each subject
                for k in exp_subj:
                    exp_simi_indv = exp[
                        (exp['layer']==name)&
                        (exp['subj']==k)].copy()

                    # normalization (Z-score)
                    exp_simi_indv['rt_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'simi_%s'%corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])
                    if fit_tag=='linear':
                        exp_simi_indv['MSS_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'setsize'])
                        exp_simi_indv['inter'] = \
                            exp_simi_indv.loc[:,
                            'simi_%s'%corr_tag]*exp_simi_indv.loc[:,
                                                'setsize']
                        exp_simi_indv['inter_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'inter'])
                    else:
                        exp_simi_indv['MSS_Z'] = preprocessing.scale(
                            np.log2(exp_simi_indv.loc[:,'setsize']))
                        exp_simi_indv['inter'] =\
                            exp_simi_indv.loc[:,
                            'simi_%s'%corr_tag]*np.log2(
                                exp_simi_indv.loc[:,'setsize'])
                        exp_simi_indv['inter_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'inter'])

                    # GLM fit
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv[['simi_Z','MSS_Z','inter_Z']]
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,offset=exp_simi_indv['cate_Z'],
                        family=sm.families.Gaussian()).fit()

                    glm_cond.append('intc')
                    glm_coeff.append(model.params[0])
                    glm_cond.append('simi')
                    glm_coeff.append(model.params[1])
                    glm_cond.append('MSS')
                    glm_coeff.append(model.params[2])
                    glm_cond.append('inter')
                    glm_coeff.append(model.params[3])
                    glm_subj += [k]*4

                    fit_subj += [k]*len(exp_simi_indv)
                    fit_exp += [exp_tag]*len(exp_simi_indv)
                    fit_corr += [corr_tag]*len(exp_simi_indv)
                    fit_layer += [name]*len(exp_simi_indv)
                    fit_tagList += [fit_tag]*len(exp_simi_indv)
                    fit_size += exp_simi_indv['setsize'].tolist()
                    fit_raw += exp_simi_indv['rt_Z'].tolist()
                    fit_pred += list(model.predict())
                    fit_resid += list(model.predict()-exp_simi_indv['rt_Z'].values)

                exp_glm_list.append(pd.DataFrame(
                    {'subj':glm_subj,'cond':glm_cond,'coeff_%s'%corr_tag:glm_coeff}))
                exp_glm_list[expN]['exp'] = exp_tag

            df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
            df_glm_layer['layer'] = name

            df_glm_list[indx] = pd.concat(
                [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
    glm_data = df_glm_list[0]
    glm_data['coeff_max'] = df_glm_list[1]['coeff_max']
    glm_data['fit'] = fit_tag
    glm_data_all = pd.concat([glm_data_all,glm_data],axis=0,ignore_index=True)

fit_data = pd.DataFrame(
    {'subj':fit_subj,'exp':fit_exp,'corr':fit_corr,'layer':fit_layer,
     'fit':fit_tagList,'setsize':fit_size,'rt':fit_raw,'pred':fit_pred,
     'resid':fit_resid})

if save_tag==1:
    glm_data_all.to_csv(os.path.join(alex_output,'glm_off_fit.csv'),
                        sep=',',mode='w',header=True,index=False)
# Plot
condList = ['simi','MSS','inter']
clist = sns.color_palette(Diona)
mpl.rcParams.update({'font.size':14})
fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(16,8))
ax = ax.ravel()
n = 0
for fit_tag in ['linear','log']:
    for k,exp_tag in enumerate(['exp1b','exp2']):
        dat = glm_data_all[
            (glm_data_all['exp']==exp_tag)&
            (glm_data_all['fit']==fit_tag)&
            (glm_data_all['cond'].isin(condList))]

        for corr_tag in ['mean','max']:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            leg_tag = True

            sns.lineplot(data=dat,x='layer',
                         y='coeff_%s'%corr_tag,
                         hue='cond',hue_order=condList,style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s %s'%(exp_tag,fit_tag))
            h,_ = ax[n].get_legend_handles_labels()
            ax[n].legend(h,['Similarity (%s)'%corr_tag,'MSS','Interaction'],
                         loc='lower left',ncol=1,fontsize=9,
                         frameon=False).set_title(None)
            n += 1
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'off_fit_expAll.png'))
plt.show(block=True)
plt.close('all')

# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ Category * MSS)
fit_subj,fit_size,fit_raw,fit_pred,fit_resid,fit_exp,fit_corr,fit_layer,fit_tagList = \
    [],[],[],[],[],[],[],[],[]
glm_data_all = pd.DataFrame()
for fit_tag in ['linear','log']:
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

                glm_subj,glm_cond,glm_coeff = [],[],[]

                # each subject
                for k in exp_subj:
                    exp_simi_indv = exp[
                        (exp['layer']==name)&
                        (exp['subj']==k)].copy()

                    # normalization (Z-score)
                    exp_simi_indv['rt_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'simi_%s'%corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])
                    if fit_tag=='linear':
                        exp_simi_indv['MSS_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'setsize'])
                        exp_simi_indv['inter'] = \
                            exp_simi_indv.loc[:,
                            'simi_%s'%corr_tag]*exp_simi_indv.loc[:,
                                                'setsize']
                        exp_simi_indv['inter_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'inter'])
                    else:
                        exp_simi_indv['MSS_Z'] = preprocessing.scale(
                            np.log2(exp_simi_indv.loc[:,'setsize']))
                        exp_simi_indv['inter'] =\
                            exp_simi_indv.loc[:,
                            'simi_%s'%corr_tag]*np.log2(
                                exp_simi_indv.loc[:,'setsize'])
                        exp_simi_indv['inter_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'inter'])

                    # GLM fit
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv[['cate_Z','MSS_Z','inter_Z']]
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,offset=exp_simi_indv['simi_Z'],
                        family=sm.families.Gaussian()).fit()

                    glm_cond.append('intc')
                    glm_coeff.append(model.params[0])
                    glm_cond.append('cate')
                    glm_coeff.append(model.params[1])
                    glm_cond.append('MSS')
                    glm_coeff.append(model.params[2])
                    glm_cond.append('inter')
                    glm_coeff.append(model.params[3])
                    glm_subj += [k]*4

                    fit_subj += [k]*len(exp_simi_indv)
                    fit_exp += [exp_tag]*len(exp_simi_indv)
                    fit_corr += [corr_tag]*len(exp_simi_indv)
                    fit_layer += [name]*len(exp_simi_indv)
                    fit_tagList += [fit_tag]*len(exp_simi_indv)
                    fit_size += exp_simi_indv['setsize'].tolist()
                    fit_raw += exp_simi_indv['rt_Z'].tolist()
                    fit_pred += list(model.predict())
                    fit_resid += list(model.predict()-exp_simi_indv['rt_Z'].values)

                exp_glm_list.append(pd.DataFrame(
                    {'subj':glm_subj,'cond':glm_cond,'coeff_%s'%corr_tag:glm_coeff}))
                exp_glm_list[expN]['exp'] = exp_tag

            df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
            df_glm_layer['layer'] = name

            df_glm_list[indx] = pd.concat(
                [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
    glm_data = df_glm_list[0]
    glm_data['coeff_max'] = df_glm_list[1]['coeff_max']
    glm_data['fit'] = fit_tag
    glm_data_all = pd.concat([glm_data_all,glm_data],axis=0,ignore_index=True)

fit_data = pd.DataFrame(
    {'subj':fit_subj,'exp':fit_exp,'corr':fit_corr,'layer':fit_layer,
     'fit':fit_tagList,'setsize':fit_size,'rt':fit_raw,'pred':fit_pred,
     'resid':fit_resid})

if save_tag==1:
    glm_data_all.to_csv(os.path.join(alex_output,'glm_off_fit.csv'),
                        sep=',',mode='w',header=True,index=False)
# Plot
condList = ['cate','MSS','inter']
clist = sns.color_palette(Diona)
mpl.rcParams.update({'font.size':14})
fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(16,8))
ax = ax.ravel()
n = 0
for fit_tag in ['linear','log']:
    for k,exp_tag in enumerate(['exp1b','exp2']):
        dat = glm_data_all[
            (glm_data_all['exp']==exp_tag)&
            (glm_data_all['fit']==fit_tag)&
            (glm_data_all['cond'].isin(condList))]

        for corr_tag in ['mean','max']:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            leg_tag = True

            sns.lineplot(data=dat,x='layer',
                         y='coeff_%s'%corr_tag,
                         hue='cond',hue_order=condList,style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s %s'%(exp_tag,fit_tag))
            h,_ = ax[n].get_legend_handles_labels()
            ax[n].legend(h,['Category (%s)'%corr_tag,'MSS','Interaction'],
                         loc='lower left',ncol=1,fontsize=9,
                         frameon=False).set_title(None)
            n += 1
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'off_fitCate_expAll.png'))
plt.show(block=True)
plt.close('all')









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
                        y,X,
                        family=sm.families.Gaussian()).fit()

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
glm_data = df_glm_list[0]
glm_data['coeff_max'] = df_glm_list[1]['coeff_max']

condList = ['simi']
# clist = sns.color_palette(Diona)
clist = ['tomato']
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(12,12))
    ax = ax.ravel()
    n = 0
    for corr_tag in ['mean','max']:
        for name in activation_names:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Category (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'simp_coeffCate_MSS_%s.png'%exp_tag))
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
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    for corr_tag in ['mean','max']:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        for sizeN in sizeList:
            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s MSS%d'%(exp_tag,sizeN))
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[4].legend(h,['Category (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Category (Mean)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[12].legend(h,['Category (Max)'],
                 loc='lower left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'simp_coeffCate_layer_expAll.png'))
plt.show(block=True)
plt.close('all')






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
glm_data = df_glm_list[0]
glm_data['coeff_max'] = df_glm_list[1]['coeff_max']

condList = ['simi']
# clist = sns.color_palette(Diona)
clist = ['dodgerblue']
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(12,12))
    ax = ax.ravel()
    n = 0
    for corr_tag in ['mean','max']:
        for name in activation_names:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Similarity (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Similarity (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'semi_coeffSimi_MSS_%s.png'%exp_tag))
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
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    for corr_tag in ['mean','max']:
        for sizeN in sizeList:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s MSS%d'%(exp_tag,sizeN))
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Similarity (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[4].legend(h,['Similarity (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Similarity (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[12].legend(h,['Similarity (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'semi_coeffSimi_layer_expAll.png'))
plt.show(block=True)
plt.close('all')
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
                    # X = model.predict()-exp_simi_indv['cate_Z'].values
                    X = np.array(list(model.resid_response))
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
glm_data = df_glm_list[0]
glm_data['coeff_max'] = df_glm_list[1]['coeff_max']



# clist = sns.color_palette(Diona)
clist = ['tomato']
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(['exp1b','exp2']):
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(12,12))
    ax = ax.ravel()
    n = 0
    for corr_tag in ['mean','max']:
        for name in activation_names:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['layer']==name],x='setsize',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Category (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'semi_coeffCate_MSS_%s.png'%exp_tag))
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
    dat = glm_data[
        (glm_data['exp']==exp_tag)&
        (glm_data['cond']!='intc')]

    for corr_tag in ['mean','max']:
        for sizeN in sizeList:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(data=dat[dat['setsize']==sizeN],x='layer',
                         y='coeff_%s'%corr_tag,hue='cond',style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s MSS%d'%(exp_tag,sizeN))
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[4].legend(h,['Category (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Category (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[12].legend(h,['Category (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'semi_coeffCate_layer_expAll.png'))
plt.show(block=True)
plt.close('all')