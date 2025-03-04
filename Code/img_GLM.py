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

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


alex_output = set_filepath(rootPath,'res_alex')
save_tag = 0
p_crit = 0.05
tail = 0
t_thresh = None
n_permutations = 1000
Diona = ['#393A53','#DD9E5B','#D58784','#B69480','#EDCAC6']
Kirara = ["#355D73","#8DC0C8","#D5C7AC","#EAC772","#69A94E"]
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

# GLM (rt ~ category + similarity)
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
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
                    X = exp_simi_indv[['cate_Z','simi_Z']]
                    X = sm.add_constant(X)
                    model = sm.GLM(y,X,family=sm.families.Gaussian()).fit()

                    simi_raw.loc[
                        (simi_raw['exp']==exp_tag)&
                        (simi_raw['layer']==name)&
                        (simi_raw['setsize']==n)&
                        (simi_raw['subj']==k),'resid_%s'%corr_tag] = \
                        list(model.resid_response)

                    glm_cond.append('intc')
                    glm_coeff.append(model.params[0])
                    glm_cond.append('cate')
                    glm_coeff.append(model.params[1])
                    glm_cond.append('simi')
                    glm_coeff.append(model.params[2])
                    glm_subj += [k]*3
                    glm_size += [n]*3
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


condList = ['cate','simi']
clist = ['tomato','dodgerblue']
# Plot
mpl.rcParams.update({'font.size':14})
for k,exp_tag in enumerate(exp_tags):
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
                         y='coeff_%s'%corr_tag,
                         hue='cond',hue_order=condList,style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(sizeList,labels=sizeList)
            ax[n].set_xlabel(xlabel='MSS')
            ax[n].set_yticks(np.arange(-0.1,0.36,0.15))
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title(name)
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category','Similarity (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Category','Similarity (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'coeff_MSS_%s.png'%exp_tag))
    plt.show(block=True)
    plt.close('all')
#
# Plot
mpl.rcParams.update({'font.size':14})
fig,ax = plt.subplots(
        4,4,sharex=True,sharey=True,figsize=(18,12))
ax = ax.ravel()
n = 0
for k,exp_tag in enumerate(exp_tags):
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
                         y='coeff_%s'%corr_tag,
                         hue='cond',hue_order=condList,style='cond',
                         markers='o',dashes=False,palette=clist,
                         linewidth=2,markersize=10,err_style='bars',
                         errorbar=('se',1),legend=leg_tag,ax=ax[n])
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_yticks(np.arange(-0.1,0.36,0.15))
            ax[n].set_ylabel(ylabel='Coefficients')
            ax[n].set_title('%s MSS%d'%(exp_tag,sizeN))
            n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Category','Similarity (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[4].legend(h,['Category','Similarity (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[8].legend(h,['Category','Similarity (Mean)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
    ax[12].legend(h,['Category','Similarity (Max)'],
                 loc='upper left',ncol=1,fontsize=10,
                 frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'coeff_layer_expAll.png'))
plt.show(block=True)
plt.close('all')


# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ similarity * MSS)
fit_subj,fit_size,fit_raw,fit_pred,\
    fit_resid,fit_exp,fit_corr,fit_layer,fit_tagList = \
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
            for expN,exp_tag in enumerate(exp_tags):
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
                    model = sm.GLM(y,X,family=sm.families.Gaussian()).fit()

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
                    fit_resid += list(model.resid_response)

                exp_glm_list.append(pd.DataFrame(
                    {'subj':glm_subj,'cond':glm_cond,
                     'coeff_%s'%corr_tag:glm_coeff}))
                exp_glm_list[expN]['exp'] = exp_tag

            df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
            df_glm_layer['layer'] = name

            df_glm_list[indx] = pd.concat(
                [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
    glm_data = df_glm_list[0]
    glm_data['coeff_max'] = df_glm_list[1]['coeff_max']
    glm_data['fit'] = fit_tag
    glm_data_all = pd.concat(
        [glm_data_all,glm_data],axis=0,ignore_index=True)

fit_data = pd.DataFrame(
    {'subj':fit_subj,'exp':fit_exp,'corr':fit_corr,'layer':fit_layer,
     'fit':fit_tagList,'setsize':fit_size,'rt':fit_raw,'pred':fit_pred,
     'resid':fit_resid})

# Plot
condList = ['simi','MSS','inter']
clist = sns.color_palette(Diona)
mpl.rcParams.update({'font.size':14})
fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(16,8))
ax = ax.ravel()
n = 0
for fit_tag in ['linear','log']:
    for k,exp_tag in enumerate(exp_tags):
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
            ax[n].set_yticks(np.arange(-0.1,0.36,0.15))
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
    os.path.join(figPath,'fit_expAll.png'))
plt.show(block=True)
plt.close('all')

# Plot
leg_tag = True
for k,exp_tag in enumerate(exp_tags):
    for fit_tag in ['linear','log']:
        mpl.rcParams.update({'font.size':14})
        fig,ax = plt.subplots(
            4,4,sharex=True,sharey=True,figsize=(12,8))
        ax = ax.ravel()
        n = 0
        for corr_tag in ['mean','max']:
            for name in activation_names:
                dat = fit_data[
                    (fit_data['fit']==fit_tag)&
                    (fit_data['exp']==exp_tag)&
                    (fit_data['layer']==name)&
                    (fit_data['corr']==corr_tag)]

                sns.lineplot(data=dat,x='setsize',y='pred',color='tomato',
                             linestyle='dashed',
                             linewidth=2,err_style='bars',estimator='mean',
                             errorbar=None,legend=leg_tag,ax=ax[n])
                sns.lineplot(data=dat,x='setsize',y='rt',color='grey',
                             linewidth=2,err_style='bars',estimator='mean',
                             errorbar=None,legend=leg_tag,ax=ax[n])
                ax[n].set_xticks(sizeList,labels=sizeList)
                ax[n].set_title('%s %s'%(corr_tag,name))
                n += 1
        plt.suptitle('%s %s'%(exp_tag,fit_tag))
        sns.despine(offset=10,trim=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(figPath,'fit_data_%s_%s.png'%(exp_tag,fit_tag)))
        plt.show(block=True)
        plt.close('all')


# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ Category)
glm_data = pd.DataFrame()
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)
    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff = \
                [],[],[],[]

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
                 'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        glm_data = pd.concat(
            [glm_data,df_glm_layer],axis=0,ignore_index=True)
        glm_data = glm_data[(glm_data['cond']!='intc')]
    glm_ealy = (glm_data.loc[
                   (glm_data['cond']=='cate')&
                   (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
        (glm_data['cond']=='cate')&
        (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
        (glm_data['cond']=='cate')&
        (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
        (glm_data['cond']=='cate')&
        (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
        (glm_data['cond']=='cate')&
        (glm_data['layer']=='conv_5'),'coeff'].values)/5
    glm_late = (glm_data.loc[
                   (glm_data['cond']=='cate')&
                   (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
        (glm_data['cond']=='cate')&
        (glm_data['layer']=='fc_7'),'coeff'].values)/2
    glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='cate')]
    glm_1 = glm_data[
        (glm_data['layer']=='fc_8')&
        (glm_data['cond']=='cate')].copy(deep=True)
    glm_1['layer'] = 'early'
    glm_1['coeff'] = glm_ealy
    glm_6 = glm_data[
        (glm_data['layer']=='fc_8')&
        (glm_data['cond']=='cate')].copy(deep=True)
    glm_6['layer'] = 'late'
    glm_6['coeff'] = glm_late
    glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)
    glm_data.to_csv(os.path.join(alex_output,'glm_rt-cate.csv'),
                    sep=',',mode='w',header=True,index=False)
    glm_meanLayer.to_csv(os.path.join(alex_output,'glm_rt-cate_3layers.csv'),
                         sep=',',mode='w',header=True,index=False)
#
# #
# # Plot
# #
# mpl.rcParams.update({'font.size':14})
# fig,ax = plt.subplots(
#         2,4,sharex=True,sharey=True,figsize=(16,9))
# ax = ax.ravel()
# n = 0
# for name in activation_names:
#     dat = glm_data[
#         (glm_data['cond']!='intc')&
#         (glm_data['layer']==name)]
#
#     ax[n].axhline(0,color='black',lw=1,linestyle='dashed')
#
#     if n==0:
#         leg_tag = True
#     else:
#         leg_tag = False
#     sns.lineplot(
#         data=dat,x='setsize',y='coeff',hue='exp',style='exp',
#         style_order=exp_tags,markers=True,dashes=False,
#         palette=['#FFBA00','grey'],
#         linewidth=2,markersize=10,err_style='bars',
#         errorbar=('se',0),legend=leg_tag,ax=ax[n])
#     ax[n].set_xticks(sizeList,labels=sizeList)
#     ax[n].set_xlabel(xlabel='Memory Set Size')
#     ax[n].set_ylabel(ylabel='Coefficients')
#     ax[n].set_title(name)
#     y_major_locator = MultipleLocator(0.125)
#     ax[n].yaxis.set_major_locator(y_major_locator)
#     n += 1
# h,_ = ax[0].get_legend_handles_labels()
# ax[0].legend(
#     h,['LTM','STM'],loc='upper left',ncol=1,
#     fontsize=12,frameon=False).set_title(None)
# sns.despine(offset=10,trim=True)
# plt.tight_layout()
# plt.savefig(
#     os.path.join(figPath,'rt-cate_layer.png'))
# plt.show(block=True)
# plt.close('all')
# #
# # Plot
# #
# mpl.rcParams.update({'font.size':14})
# fig,ax = plt.subplots(
#         1,4,sharex=True,sharey=True,figsize=(18,6))
# ax = ax.ravel()
# n = 0
# for sizeN in sizeList:
#     dat = glm_data[
#         (glm_data['cond']!='intc')&
#         (glm_data['setsize']==sizeN)]
#
#     ax[n].axhline(0,color='black',lw=1,linestyle=':')
#
#     if n==0:
#         leg_tag = True
#     else:
#         leg_tag = False
#     sns.lineplot(
#         data=dat[dat['setsize']==sizeN],x='layer',y='coeff',
#         hue='exp',style='exp',style_order=exp_tags,markers=True,
#         dashes=False,palette=['#FFBA00','grey'],
#         linewidth=2,markersize=10,err_style='bars',
#         errorbar=('se',0),legend=leg_tag,ax=ax[n])
#     ax[n].set_xticks(activation_names,labels=range(1,9))
#     ax[n].set_xlabel(xlabel='Layer')
#     ax[n].set_ylabel(ylabel='Coefficients')
#     # ax[n].set_ylim(-0.125,0.32)
#     ax[n].set_title('MSS %d'%sizeN)
#     y_major_locator = MultipleLocator(0.15)
#     ax[n].yaxis.set_major_locator(y_major_locator)
#
#     y_gap = 0.008
#     y_sig = -0.02-y_gap
#     y_fsig = 0.31+y_gap
#     for exp_tag in exp_tags:
#         y_sig += y_gap
#         dat_cond = dat[(dat['exp']==exp_tag)]
#         X = np.array(
#             [dat_cond.loc[(dat_cond['layer']==x_name),
#             'coeff'].values for x_name in activation_names])
#         X = np.transpose(X,(1,0))
#
#         t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
#             X,n_jobs=None,threshold=t_thresh,adjacency=None,
#             n_permutations=n_permutations,out_type='indices')
#
#         if (len(clusters)!=0):
#             if (p_values[0]<p_crit):
#                 sig_x = ['conv_%d'%(layerN+1) \
#                 if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
#                 for layerN in list(clusters[0][0])]
#                 if exp_tag=='exp1b':
#                     lcolor = '#FFBA00'
#                     lstyle = 'o'
#                 else:
#                     lcolor = 'grey'
#                     lstyle = 'x'
#
#                 ax[n].scatter(
#                     sig_x,[y_sig]*len(sig_x),c=lcolor,
#                     s=10,marker=lstyle)
#
#     # exp
#     X = np.array(
#         [[dat.loc[(dat['setsize']==sizeN)&
#                   (dat['exp']=='exp1b')&
#                   (dat['layer']==x_name),
#         'coeff'].values for x_name in activation_names],
#         [dat.loc[(dat['setsize']==sizeN)&
#                  (dat['exp']=='exp2')&
#                  (dat['layer']==x_name),
#         'coeff'].values for x_name in activation_names]])
#     X = np.transpose(X,(0,2,1))
#     t_clust,clusters,p_values,H0 = permutation_cluster_test(
#         X,n_jobs=None,threshold=t_thresh,adjacency=None,
#         n_permutations=n_permutations,out_type='indices')
#
#     if (len(clusters)!=0):
#         if (p_values[0]<p_crit):
#             sig_x = ['conv_%d'%(layerN+1) \
#             if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
#             for layerN in list(clusters[0][0])]
#
#             ax[n].scatter(
#                 sig_x,[y_fsig]*len(sig_x),c='black',
#                 s=10,marker='o')
#
#     n += 1
# h,_ = ax[0].get_legend_handles_labels()
# ax[0].legend(
#     h,['LTM','STM'],loc='upper left',ncol=1,
#     fontsize=12,frameon=False).set_title(None)
# # ax[0].get_legend().remove()
# sns.despine(offset=10,trim=True)
# plt.tight_layout()
# plt.savefig(
#     os.path.join(figPath,'rt-cate_MSS.png'))
# plt.show(block=True)
# plt.close('all')


# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ Similarity)
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff,glm_corr = \
                [],[],[],[],[]

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
            exp_glm_list.append(pd.DataFrame(
                {'corr':glm_corr,'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)
glm_ealy = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')]
glm_1 = glm_data[
    (glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[
    (glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

# glm_data.to_csv(os.path.join(alex_output,'glm_rt-simi.csv'),
#                   sep=',',mode='w',header=True,index=False)
# glm_meanLayer.to_csv(os.path.join(alex_output,'glm_rt-simi_3layers.csv'),
#                   sep=',',mode='w',header=True,index=False)

#
# Plot
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':14})
for corr_tag in ['mean','max']:
    fig,ax = plt.subplots(
            2,4,sharex=True,sharey=True,figsize=(16,9))
    ax = ax.ravel()
    n = 0
    for name in activation_names:
        dat = glm_data[
            (glm_data['cond']!='intc')&
            (glm_data['corr']==corr_tag)&
            (glm_data['layer']==name)]

        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat,x='setsize',
                     y='coeff',hue='exp',style='exp',
                     markers=True,dashes=False,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='MSS')
        ax[n].set_ylabel(ylabel='Beta')
        ax[n].set_title(name)
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_rt-simi_MSS.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
# Plot
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':18})
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        dat = glm_data[
            (glm_data['cond']!='intc')&
            (glm_data['corr']==corr_tag)&
            (glm_data['setsize']==sizeN)]

        ax[n].axhline(0,color='black',lw=1,linestyle=':')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[dat['setsize']==sizeN],x='layer',
                     y='coeff',hue='exp',style='exp',
                     markers=True,dashes=False,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Beta')
        # ax[n].set_ylim(-0.125,0.32)
        ax[n].set_title('MSS %d'%sizeN)
        y_major_locator = MultipleLocator(0.125)
        ax[n].yaxis.set_major_locator(y_major_locator)

        y_gap = 0.008
        y_sig = -0.04-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp']==exp_tag)]
            X = np.array(
                [dat_cond.loc[(dat_cond['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('MSS=%d %s'%(sizeN,exp_tag))

            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<p_crit):
                        sig_x = [
                            'conv_%d'%(layerN+1) \
                                if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                            for layerN in list(clusters[pN][0])]
                        if exp_tag=='exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=10,marker=lstyle)

        # # exp
        # X = np.array(
        #     [[dat.loc[(dat['setsize']==sizeN)&
        #               (dat['exp']=='exp1b')&
        #               (dat['layer']==x_name),
        #     'coeff'].values for x_name in activation_names],
        #     [dat.loc[(dat['setsize']==sizeN)&
        #              (dat['exp']=='exp2')&
        #              (dat['layer']==x_name),
        #     'coeff'].values for x_name in activation_names]])
        # X = np.transpose(X,(0,2,1))
        # t_clust,clusters,p_values,H0 = permutation_cluster_test(
        #     X,n_jobs=None,threshold=t_thresh,adjacency=None,
        #     n_permutations=n_permutations,out_type='indices')

        # if (len(clusters)!=0):
        #     if (p_values[0]<p_crit):
        #         sig_x = [
        #         'conv_%d'%(layerN+1) \
        #         if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
        #         for layerN in list(clusters[0][0])]
        #         ax[n].scatter(
        #             sig_x,[y_fsig]*len(sig_x),c='black',s=10,marker='o')


        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=1,
        fontsize=16,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_rt-simi_layer.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

#
# ANOVA lineplot
#
mpl.rcParams.update({'font.size':19})
for corr_tag in ['mean','max']:

    fig,ax = plt.subplots(
        1,2,sharex=True,sharey=True,figsize=(19,8))
    ax.ravel()
    for n,exp_tag in enumerate(exp_tags):
        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['exp']==exp_tag)]
        if exp_tag=='exp1b':
            leg_tag = True
            exp_title = 'LTM'
            figN = '(A)'
        else:
            leg_tag = False
            exp_title = 'STM'
            figN = '(B)'

        sns.lineplot(
            data=dat,x='setsize',y='coeff',hue='layer',
            hue_order=['early','late','fc_8'],style='layer',
            markers=True,dashes=False,palette=Diona[0:3],
            linewidth=2,markersize=12,err_style='bars',
            errorbar=('se',1),err_kws={'capsize':10},
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        y_major_locator = MultipleLocator(0.1)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(exp_title)
        ax[n].text(0.1,0.27,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_line_simi_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')

#
clist = ['#FFBA00','grey']
for corr_tag in ['mean','max']:
    mpl.rcParams.update({'font.size':14})
    fig,ax = plt.subplots(
                1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        if n==0:
            leg_tag = True
        else:
            leg_tag = False

        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['setsize']==sizeN)]

        sns.barplot(data=dat,x='layer',y='coeff',hue='exp',
                    hue_order=exp_tags,
                    order=['early','late','fc_8'],
                    palette=clist,capsize=0.2,
                    errcolor='grey',legend=leg_tag,ax=ax[n])
        ax[n].set_title('MSS %d'%(sizeN))
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=1,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_bar_rt-simi_MSS.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')

# 6-7
for corr_tag in ['mean','max']:
    mpl.rcParams.update({'font.size':14})
    fig,ax = plt.subplots(
                1,1,sharex=True,sharey=True,figsize=(12,9))
    dat = glm_meanLayer[
        (glm_meanLayer['corr']==corr_tag)&
        (glm_meanLayer['layer']=='late')]

    sns.barplot(data=dat,x='exp',y='coeff',hue='setsize',
                palette=Diona[0:3]+[Diona[4]],capsize=0.2,
                errcolor='grey',legend=True,ax=ax)
    ax.set_xticks(exp_tags,labels=['LTM','STM'])
    ax.set_xlabel(xlabel='Task')
    ax.set_title('Late Layer (FC 6 & FC 7)')
    h,_ = ax.get_legend_handles_labels()
    ax.legend(
        h,['MSS 1','MSS 2','MSS 4','MSS 8'],loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_bar_rt-simi_late.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')



# --- * --- * --- * --- * --- * --- * --- * --- * ---

import pingouin as pg
glm_meanLayer['grp'] = glm_meanLayer['exp']+glm_meanLayer['subj'].astype(str)
for corr_tag in ['mean','max']:
    dat = glm_meanLayer[
        (glm_meanLayer['cond']!='intc')&
        (glm_meanLayer['layer']=='late')&
        (glm_meanLayer['corr']==corr_tag)]
    aov = pg.mixed_anova(
        data=dat,dv='coeff',between='exp',within='setsize',
        subject='grp',correction=True,effsize='np2').round(3)
    print('%s'%(corr_tag))
    print(aov)
    post_hocs = pg.pairwise_tests(
        data=dat,dv='coeff',between='exp',within='setsize',
        subject='grp').round(3).round(3)
    print(post_hocs)
    print('--- --- --- --- --- --- --- --- ---')



# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ similarity) within/between
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cate,glm_cond,glm_coeff,glm_corr = \
                [],[],[],[],[],[]

            # each MSS
            for n in sizeList:
                # each category
                for cate in cateList:
                    # each subject
                    for k in exp_subj:

                        exp_simi_indv = exp[
                            (exp['layer']==name)&
                            (exp['setsize']==n)&
                            (exp['cate']==cate)&
                            (exp['subj']==k)].copy()

                        # normalization (Z-score)
                        exp_simi_indv['rt_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'rt'])
                        exp_simi_indv['simi_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'simi_%s'%corr_tag])

                        # GLM fit
                        y = exp_simi_indv['rt_Z']
                        X = exp_simi_indv['simi_Z']
                        X = sm.add_constant(X)
                        model = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        simi_raw.loc[
                            (simi_raw['exp']==exp_tag)&
                            (simi_raw['layer']==name)&
                            (simi_raw['setsize']==n)&
                            (simi_raw['cate']==cate)&
                            (simi_raw['subj']==k),'resid_%s'%corr_tag] = \
                            list(model.resid_response)

                        glm_cond.append('intc')
                        glm_coeff.append(model.params[0])
                        glm_cond.append('simi')
                        glm_coeff.append(model.params[1])
                        glm_subj += [k]*2
                        glm_size += [n]*2
                        glm_cate += [cate]*2
                        glm_corr += [corr_tag]*2
            exp_glm_list.append(pd.DataFrame(
                {'subj':glm_subj,'setsize':glm_size,'cate':glm_cate,
                 'cond':glm_cond,'coeff':glm_coeff,'corr':glm_corr}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)
glm_data = glm_data[glm_data['cond']!='intc']
glm_data_avg = glm_data[(glm_data['cate']=='within')&
                        (glm_data['cond']=='simi')].copy(deep=True)
glm_data_avg.drop(labels=['cate','coeff'],axis=1,inplace=True)
glm_data_avg['coeff'] = (glm_data.loc[
                             (glm_data['cate']=='within')&
                             (glm_data['cond']=='simi'),
                             'coeff'].values+glm_data.loc[
    (glm_data['cate']=='between')&
    (glm_data['cond']=='simi'),'coeff'].values)/2
final_col = ['exp','subj','block','cate','setsize',
             'rt','layer','simi_mean','simi_max',
             'resid_mean','resid_max']
simi_data = simi_raw[final_col]



glm_ealy = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')]
glm_1 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_ealy = (glm_data_avg.loc[
               (glm_data_avg['cond']=='simi')&
               (glm_data_avg['layer']=='conv_1'),
               'coeff'].values+glm_data_avg.loc[
    (glm_data_avg['cond']=='simi')&
    (glm_data_avg['layer']=='conv_2'),'coeff'].values+glm_data_avg.loc[
    (glm_data_avg['cond']=='simi')&
    (glm_data_avg['layer']=='conv_3'),'coeff'].values+glm_data_avg.loc[
    (glm_data_avg['cond']=='simi')&
    (glm_data_avg['layer']=='conv_4'),'coeff'].values+glm_data_avg.loc[
    (glm_data_avg['cond']=='simi')&
    (glm_data_avg['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data_avg.loc[
               (glm_data_avg['cond']=='simi')&
               (glm_data_avg['layer']=='fc_6'),'coeff'].values+glm_data_avg.loc[
    (glm_data_avg['cond']=='simi')&
    (glm_data_avg['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data_avg[(glm_data_avg['layer']=='fc_8')&
                     (glm_data_avg['cond']=='simi')]
glm_1 = glm_data_avg[(glm_data_avg['layer']=='fc_8')&
                     (glm_data_avg['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&
    (glm_data_avg['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_avg_meanLayer = pd.concat(
    [glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(alex_output,'glm_rt-2cate.csv'),
                sep=',',mode='w',header=True,index=False)
glm_data_avg.to_csv(os.path.join(alex_output,'glm_rt-2avg.csv'),
                    sep=',',mode='w',header=True,index=False)
glm_avg_meanLayer.to_csv(os.path.join(alex_output,'glm_rt-2avg_3layers.csv'),
                         sep=',',mode='w',header=True,index=False)

#
#
# Plot
#
mpl.rcParams.update({'font.size':14})
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
            2,4,sharex=True,sharey=True,figsize=(16,9))
    ax = ax.ravel()
    n = 0
    for name in activation_names:
        dat = glm_data[
            (glm_data['layer']==name)&
            (glm_data['cond']!='intc')]
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(
            data=dat,x='setsize',y='coeff',hue='cate',hue_order=cateList,
            markers=True,style='exp',dashes=True,palette='Blues',
            linewidth=1.5,markersize=6,err_style='bars',errorbar=('se',0),
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='MSS')
        ax[n].set_ylabel(ylabel='Beta')
        ax[n].set_title('%s'%(name))
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Category','within','between','Task','LTM','STM'],
        loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_rt-2cate_layer.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
#
# Plot
#
for corr_tag in ['max','mean']:
    mpl.rcParams.update({'font.size':18})
    fig,ax = plt.subplots(
        1,4,sharex=True,sharey=True,figsize=(19,6))
    ax = ax.ravel()
    dat = glm_data[
        (glm_data['corr']==corr_tag)&
        (glm_data['cond']!='intc')]

    n = 0
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(
            data=dat[dat['setsize']==sizeN],x='layer',y='coeff',hue='cate',
            hue_order=cateList,markers=True,style='exp',
            dashes=True,palette='Blues',linewidth=1.5,markersize=8,
            err_style='bars',errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Beta')
        # ax[n].set_ylim(-0.125,0.25)
        y_major_locator = MultipleLocator(0.1)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title('MSS %d'%(sizeN))

        y_gap = 0.008
        y_sig = -0.095-y_gap
        for exp_tag in exp_tags:
            for cate in cateList:
                y_sig += y_gap
                dat_cond = dat[(dat['exp']==exp_tag)&
                               (dat['setsize']==sizeN)&
                               (dat['cate']==cate)]
                X_layer = np.array(
                    [dat_cond.loc[(dat_cond['layer']==x_name),
                    'coeff'].values for x_name in activation_names])
                X = np.transpose(X_layer,(1,0))

                t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                    X,n_jobs=None,threshold=t_thresh,adjacency=None,
                    n_permutations=n_permutations,out_type='indices')

                print('%s MSS=%d %s %s'%(corr_tag,sizeN,exp_tag,cate))
                print(clusters)
                print(p_values)

                if (len(clusters)!=0):
                    for pN in range(len(p_values)):
                        if (p_values[pN]<p_crit):
                            sig_x = [
                                'conv_%d'%(layerN+1) \
                                    if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                for layerN in list(clusters[pN][0])]
                            if exp_tag=='exp1b':
                                lstyle = 'o'
                            else:
                                lstyle = 'x'
                            if cate=='within':
                                lcolor = 'lightsteelblue'
                            else:
                                lcolor = 'steelblue'

                            ax[n].scatter(
                                sig_x,[y_sig]*len(sig_x),c=lcolor,
                                s=12,marker=lstyle)
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Category','within','between','Task','LTM','STM'],
        loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
            os.path.join(figPath,'%s_rt-2cate_MSS.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')


#
# ANOVA lineplot
#
mpl.rcParams.update({'font.size':19})
for corr_tag in ['mean','max']:

    fig,ax = plt.subplots(
        1,2,sharex=True,sharey=True,figsize=(19,8))
    ax.ravel()
    for n,exp_tag in enumerate(exp_tags):
        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['exp']==exp_tag)]
        if exp_tag=='exp1b':
            leg_tag = True
            exp_title = 'LTM'
            figN = '(A)'
        else:
            leg_tag = False
            exp_title = 'STM'
            figN = '(B)'

        sns.lineplot(
            data=dat,x='setsize',y='coeff',hue='layer',
            hue_order=['early','late','fc_8'],style='layer',
            markers=True,dashes=False,palette=Diona[0:3],
            linewidth=2,markersize=12,err_style='bars',
            errorbar=('se',1),err_kws={'capsize':10},
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        y_major_locator = MultipleLocator(0.1)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(exp_title)
        ax[n].text(0.1,0.27,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_line_avg2cate_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')


#
clist = ['#FFBA00','grey']
for corr_tag in ['mean','max']:
    mpl.rcParams.update({'font.size':14})
    fig,ax = plt.subplots(
                1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        if n==0:
            leg_tag = True
        else:
            leg_tag = False

        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['setsize']==sizeN)]

        sns.barplot(data=dat,x='layer',y='coeff',hue='exp',
                    hue_order=exp_tags,
                    order=['early','late','fc_8'],
                    palette=clist,capsize=0.2,
                    errcolor='grey',legend=leg_tag,ax=ax[n])
        ax[n].set_title('MSS %d'%(sizeN))
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=1,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_bar_rt-2cate_MSS.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

# 6-7
for corr_tag in ['mean','max']:
    mpl.rcParams.update({'font.size':14})
    fig,ax = plt.subplots(
                1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    dat = glm_meanLayer[
        (glm_meanLayer['corr']==corr_tag)&
        (glm_meanLayer['layer']=='late')]

    for n,sizeN in enumerate(sizeList):
        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.barplot(data=dat,x='exp',y='coeff',hue='cate',
                    palette='Blues',capsize=0.2,
                    errcolor='grey',legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(exp_tags,labels=['LTM','STM'])
        ax[n].set_xlabel(xlabel='Task')
        ax[n].set_title('MSS %d'%sizeN)
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['within','between'],loc='upper left',ncol=1,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_bar_rt-2cate_late.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
# Plot average
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':14})
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
            2,4,sharex=True,sharey=True,figsize=(16,9))
    ax = ax.ravel()
    n = 0
    for name in activation_names:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(
            data=glm_data_avg[(glm_data_avg['layer']==name)&
                              (glm_data_avg['corr']==corr_tag)],
            x='setsize',y='coeff',hue='exp',markers=True,style='exp',
            dashes=False,palette=clist,linewidth=1.5,markersize=8,
            err_style='bars',errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='MSS')
        ax[n].set_ylabel(ylabel='Beta')
        ax[n].set_title('%s'%(name))
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_rt-2avg_layer.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':18})
for corr_tag in ['mean','max']:
    fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(19,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(
            data=glm_data_avg[(glm_data_avg['setsize']==sizeN)&
                              (glm_data_avg['corr']==corr_tag)],
            x='layer',y='coeff',hue='exp',hue_order=exp_tags,
            markers=True,style='exp',dashes=False,palette=clist,
            linewidth=2,markersize=10,err_style='bars',
            errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Beta')
        y_major_locator = MultipleLocator(0.1)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title('MSS=%d'%(sizeN))

        y_gap = 0.008
        y_sig = -0.1-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:

            y_sig += y_gap
            dat_cond = glm_data_avg[(glm_data_avg['exp']==exp_tag)&
                                    (glm_data_avg['setsize']==sizeN)&
                                    (glm_data_avg['corr']==corr_tag)]
            X = np.array(
                [dat_cond.loc[(dat_cond['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('%s MSS=%d %s'%(corr_tag,sizeN,exp_tag))
            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<0.05):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                 for layerN in list(clusters[pN][0])]
                        if exp_tag=='exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=12,marker=lstyle)
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=1,
        fontsize=16,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_rt-2avg_MSS_01s.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')


#
clist = ['#FFBA00','grey']
for corr_tag in ['mean','max']:
    mpl.rcParams.update({'font.size':14})
    fig,ax = plt.subplots(
                1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        if n==0:
            leg_tag = True
        else:
            leg_tag = False

        dat = glm_avg_meanLayer[
            (glm_avg_meanLayer['corr']==corr_tag)&
            (glm_avg_meanLayer['setsize']==sizeN)]

        sns.barplot(data=dat,x='layer',y='coeff',hue='exp',
                    hue_order=exp_tags,
                    order=['early','late','fc_8'],
                    palette=clist,capsize=0.2,
                    errcolor='grey',legend=leg_tag,ax=ax[n])
        ax[n].set_title('MSS %d'%(sizeN))
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='upper left',ncol=1,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_bar_rt-2avg_MSS.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')

# 6-7
for corr_tag in ['mean','max']:
    mpl.rcParams.update({'font.size':14})
    fig,ax = plt.subplots(
                1,1,sharex=True,sharey=True,figsize=(12,9))
    dat = glm_avg_meanLayer[
        (glm_avg_meanLayer['corr']==corr_tag)&
        (glm_avg_meanLayer['layer']=='late')]

    sns.barplot(data=dat,x='exp',y='coeff',hue='setsize',
                palette=Diona[0:3]+[Diona[4]],capsize=0.2,
                errcolor='grey',legend=True,ax=ax)
    ax.set_xticks(exp_tags,labels=['LTM','STM'])
    ax.set_xlabel(xlabel='Task')
    ax.set_title('Late Layer (FC 6 & FC 7)')
    h,_ = ax.get_legend_handles_labels()
    ax.legend(
        h,['MSS 1','MSS 2','MSS 4','MSS 8'],loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_bar_rt-2avg_late.png'%corr_tag))
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
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff,glm_corr = \
                [],[],[],[],[]

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
                    glm_corr += [corr_tag]*2
            exp_glm_list.append(pd.DataFrame(
                {'corr':glm_corr,'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)

#
#
# Plot
#
#
# clist = sns.color_palette(Diona)
clist = ['tomato','dodgerblue']
# Plot
mpl.rcParams.update({'font.size':14})
dat = glm_data[(glm_data['cond']!='intc')]

fig,ax = plt.subplots(
    2,4,sharex=True,sharey=True,figsize=(12,9))
ax = ax.ravel()
n = 0
for name in activation_names:
    ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

    if n==0:
        leg_tag = True
    else:
        leg_tag = False
    sns.lineplot(data=dat[dat['layer']==name],x='setsize',
                 y='coeff',hue='exp',style='corr',
                 hue_order=exp_tags,style_order=['max','mean'],
                 markers=True,dashes=False,palette=clist,
                 linewidth=2,markersize=10,err_style='bars',
                 errorbar=('se',0),legend=leg_tag,ax=ax[n])
    ax[n].set_xticks(sizeList,labels=sizeList)
    ax[n].set_xlabel(xlabel='MSS')
    ax[n].set_ylabel(ylabel='Coefficients')
    ax[n].set_title(name)
    n += 1
h,_ = ax[0].get_legend_handles_labels()
fig.legend(
    loc='upper center',ncol=6,fontsize=9,
    frameon=False).set_title(None)
ax[0].get_legend().remove()
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'semi_coeffSimi_layer.png'))
plt.show(block=True)
plt.close('all')
#
#
# Plot
#
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':18})
for corr_tag in ['mean','max']:
    print(corr_tag)

    fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(19,6))
    ax = ax.ravel()
    n = 0
    dat = glm_data[(glm_data['cond']!='intc')]
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
                     x='layer',y='coeff',hue='exp',style='exp',
                     hue_order=exp_tags,style_order=['exp1b','exp2'],
                     markers=True,dashes=False,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Beta')
        # y_major_locator = MultipleLocator(0.15)
        # ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title('MSS%d'%(sizeN))

        y_gap = 0.01
        y_sig = -0.1-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp']==exp_tag)&
                           (dat['setsize']==sizeN)&
                           (dat['corr']==corr_tag)]
            X = np.array(
                [dat_cond.loc[(dat_cond['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('%s MSS=%d %s'%(corr_tag,sizeN,exp_tag))
            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<p_crit):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                 for layerN in list(clusters[pN][0])]
                        if exp_tag=='exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=10,marker=lstyle)
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_rt_simi_semi_MSS.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')




# GLM (resid RT ~ Category + semi(Similarity))

df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff,glm_corr = \
                [],[],[],[],[]

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
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv['simi_Z']
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    # 2nd step
                    y = np.array(list(model.resid_response))
                    # X = model.predict()-exp_simi_indv['cate_Z'].values
                    X = exp_simi_indv['cate_Z']
                    X = sm.add_constant(X)
                    model2 = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    glm_cond.append('intc')
                    glm_coeff.append(model2.params[0])
                    glm_cond.append('cate')
                    glm_coeff.append(model2.params[1])
                    glm_subj += [k]*2
                    glm_size += [n]*2
                    glm_corr += [corr_tag]*2
            exp_glm_list.append(pd.DataFrame(
                {'corr':glm_corr,'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)

glm_ealy = (glm_data.loc[
               (glm_data['cond']=='cate')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='cate')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='cate')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='cate')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='cate')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='cate')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='cate')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='cate')]
glm_1 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='cate')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='cate')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(alex_output,'glm_resid-cate.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(alex_output,'glm_resid-cate_3layers.csv'),
                     sep=',',mode='w',header=True,index=False)

#
#
# Plot
#
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':18})
for corr_tag in ['mean','max']:
    print(corr_tag)

    fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(19,6))
    ax = ax.ravel()
    n = 0
    dat = glm_data[(glm_data['cond']!='intc')]
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[(dat['corr']==corr_tag)&
                              (dat['setsize']==sizeN)],
                     x='layer',y='coeff',hue='exp',style='exp',
                     hue_order=exp_tags,style_order=['exp1b','exp2'],
                     markers=True,dashes=False,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Beta')
        y_major_locator = MultipleLocator(0.15)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title('MSS%d'%(sizeN))

        y_gap = 0.01
        y_sig = -0.1-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp']==exp_tag)&
                           (dat['setsize']==sizeN)&
                           (dat['corr']==corr_tag)]
            X = np.array(
                [dat_cond.loc[(dat_cond['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('%s MSS=%d %s'%(corr_tag,sizeN,exp_tag))
            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<p_crit):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                 for layerN in list(clusters[pN][0])]
                        if exp_tag=='exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=10,marker=lstyle)
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'semi_cate_MSS_%s.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

#
# ANOVA lineplot
#
mpl.rcParams.update({'font.size':19})
for corr_tag in ['mean','max']:

    fig,ax = plt.subplots(
        1,2,sharex=True,sharey=True,figsize=(19,8))
    ax.ravel()
    for n,exp_tag in enumerate(exp_tags):
        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['exp']==exp_tag)]
        if exp_tag=='exp1b':
            leg_tag = True
            exp_title = 'LTM'
            figN = '(A)'
        else:
            leg_tag = False
            exp_title = 'STM'
            figN = '(B)'

        sns.lineplot(
            data=dat,x='setsize',y='coeff',hue='layer',
            hue_order=['early','late','fc_8'],style='layer',
            markers=True,dashes=False,palette=Diona[0:3],
            linewidth=2,markersize=12,err_style='bars',
            errorbar=('se',1),err_kws={'capsize':10},
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        y_major_locator = MultipleLocator(0.15)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(exp_title)
        ax[n].text(0,0.36,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_line_semi_cate_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')

# # early, late, fc8
# glm_meanLayer['grp'] = glm_meanLayer['exp']+glm_meanLayer['subj'].astype(str)
# for exp_tag in exp_tags:
#     for corr_tag in ['mean','max']:
#         dat = glm_meanLayer[
#             (glm_meanLayer['exp']==exp_tag)&
#             (glm_meanLayer['corr']==corr_tag)]
#
#
#         aov = pg.rm_anova(
#             data=dat,dv='coeff',within=['layer','setsize'],
#             subject='subj',correction=True,effsize='np2').round(3)
#         print('%s %s'%(exp_tag,corr_tag))
#         print(aov)
#         # post_hocs = pg.pairwise_tests(
#         #     data=dat,dv='coeff',within=['layer','setsize'],
#         #     subject='subj').round(3)
#         # print(post_hocs)
#         print('--- --- --- --- --- --- --- --- ---')


#
# barplot
#
mpl.rcParams.update({'font.size':14})
fig,ax = plt.subplots(
            2,4,sharex=True,sharey=True,figsize=(16,9))
ax = ax.ravel()
n = 0
for corr_tag in ['mean','max']:
    for sizeN in sizeList:
        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['setsize']==sizeN)]

        sns.barplot(data=dat,x='exp',y='coeff',hue='layer',
                    hue_order=['early','late','fc_8'],
                    order=exp_tags,palette=Diona[0:3],capsize=0.2,
                    errcolor='grey',ax=ax[n])
        ax[n].set_title('%s MSS=%d'%(corr_tag,sizeN))
        n += 1
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'semi_cate_barplt_MSS.png'))
plt.show(block=True)
plt.close('all')

# early, late, fc8
glm_meanLayer['grp'] = glm_meanLayer['exp']+glm_meanLayer['subj'].astype(str)
for exp_tag in exp_tags:
    for corr_tag in ['mean','max']:
        dat = glm_meanLayer[
            (glm_meanLayer['exp']==exp_tag)&
            (glm_meanLayer['corr']==corr_tag)]


        aov = pg.rm_anova(
            data=dat,dv='coeff',within=['layer','setsize'],
            subject='subj',correction=True,effsize='np2').round(3)
        print('%s %s'%(exp_tag,corr_tag))
        print(aov)
        # post_hocs = pg.pairwise_tests(
        #     data=dat,dv='coeff',within=['layer','setsize'],
        #     subject='subj').round(3)
        # print(post_hocs)
        print('--- --- --- --- --- --- --- --- ---')





# GLM (resid RT ~ Similarity + semi(Catgory))

df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff,glm_corr = \
                [],[],[],[],[]

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
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv['cate_Z']
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    # 2nd step
                    y = np.array(list(model.resid_response))
                    # X = model.predict()-exp_simi_indv['cate_Z'].values
                    X = exp_simi_indv['simi_Z']
                    X = sm.add_constant(X)
                    model2 = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    glm_cond.append('intc')
                    glm_coeff.append(model2.params[0])
                    glm_cond.append('simi')
                    glm_coeff.append(model2.params[1])
                    glm_subj += [k]*2
                    glm_size += [n]*2
                    glm_corr += [corr_tag]*2
            exp_glm_list.append(pd.DataFrame(
                {'corr':glm_corr,'subj':glm_subj,'setsize':glm_size,
                 'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)

glm_ealy = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')]
glm_1 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(alex_output,'glm_resid-simi.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(alex_output,'glm_resid-simi_3layers.csv'),
                     sep=',',mode='w',header=True,index=False)

#
# Plot
#
#
clist = ['#FFBA00','grey']
mpl.rcParams.update({'font.size':18})
for corr_tag in ['mean','max']:
    print(corr_tag)

    fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(19,6))
    ax = ax.ravel()
    n = 0
    dat = glm_data[(glm_data['cond']!='intc')]
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
                     x='layer',y='coeff',hue='exp',style='exp',
                     hue_order=exp_tags,style_order=['exp1b','exp2'],
                     markers=True,dashes=False,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Beta')
        # y_major_locator = MultipleLocator(0.15)
        # ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title('MSS%d'%(sizeN))

        y_gap = 0.01
        y_sig = -0.1-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp']==exp_tag)&
                           (dat['setsize']==sizeN)&
                           (dat['corr']==corr_tag)]
            X = np.array(
                [dat_cond.loc[(dat_cond['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,
                n_permutations=n_permutations,out_type='indices')
            print('%s MSS=%d %s'%(corr_tag,sizeN,exp_tag))
            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<p_crit):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                 for layerN in list(clusters[pN][0])]
                        if exp_tag=='exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=10,marker=lstyle)
        n += 1
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'semi_simi_MSS_%s.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

#
# ANOVA lineplot
#
mpl.rcParams.update({'font.size':19})
for corr_tag in ['mean','max']:

    fig,ax = plt.subplots(
        1,2,sharex=True,sharey=True,figsize=(19,8))
    ax.ravel()
    for n,exp_tag in enumerate(exp_tags):
        dat = glm_meanLayer[
            (glm_meanLayer['corr']==corr_tag)&
            (glm_meanLayer['exp']==exp_tag)]
        if exp_tag=='exp1b':
            leg_tag = True
            exp_title = 'LTM'
            figN = '(A)'
        else:
            leg_tag = False
            exp_title = 'STM'
            figN = '(B)'

        sns.lineplot(
            data=dat,x='setsize',y='coeff',hue='layer',
            hue_order=['early','late','fc_8'],style='layer',
            markers=True,dashes=False,palette=Diona[0:3],
            linewidth=2,markersize=12,err_style='bars',
            errorbar=('se',1),err_kws={'capsize':10},
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        y_major_locator = MultipleLocator(0.15)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(exp_title)
        ax[n].text(0,0.36,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_line_semi_simi_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')



# --- * --- * --- * --- * --- * --- * --- * --- * ---



# block
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff,glm_corr,glm_blk = \
                [],[],[],[],[],[]

            # each block
            for blk in blockCate:
                # each MSS
                for n in sizeList:

                    # each subject
                    for k in exp_subj:
                        if (exp_tag=='exp2')&(k==12):
                            continue

                        exp_simi_indv = exp[
                            (exp['layer']==name)&
                            (exp['block']==blk)&
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
                        X = np.array(list(model.resid_response))
                        X = sm.add_constant(X)
                        model2 = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        glm_cond.append('intc')
                        glm_coeff.append(model2.params[0])
                        glm_cond.append('simi')
                        glm_coeff.append(model2.params[1])
                        glm_subj += [k]*2
                        glm_size += [n]*2
                        glm_corr += [corr_tag]*2
                        glm_blk += [blk]*2
            exp_glm_list.append(pd.DataFrame(
                {'corr':glm_corr,'block':glm_blk,'subj':glm_subj,
                 'setsize':glm_size,'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)

glm_ealy = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')]
glm_1 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat(
    [glm_1,glm_6,glm_8],axis=0,ignore_index=True)

#
clist = ['tomato','#4C5698']
mpl.rcParams.update({'font.size':14})
dat = glm_data[(glm_data['cond']!='intc')]
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(16,9))
    ax = ax.ravel()
    n = 0
    for name in activation_names:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[
            (dat['layer']==name)&(dat['corr']==corr_tag)],
                     x='setsize',y='coeff',hue='block',style='exp',
                     markers=True,dashes=True,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='Memory Set Size')
        ax[n].set_ylabel(ylabel='Coefficients')
        ax[n].set_title('%s'%(name))
        n += 1

    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Target','animate','inanimate','Task','LTM','STM'],
        loc='lower left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_blk_rt-cate-semi_layer.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
clist = ['tomato','#4C5698']
mpl.rcParams.update({'font.size':14})
dat = glm_data[(glm_data['cond']!='intc')]
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
        1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[
            (dat['setsize']==sizeN)&(dat['corr']==corr_tag)],
                     x='layer',y='coeff',hue='block',style='exp',
                     markers=True,dashes=True,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Coefficients')
        ax[n].set_title('MSS %d'%(sizeN))

        y_gap = 0.008
        y_sig = -0.2-y_gap
        y_fsig = 0.31+y_gap
        for exp_tag in exp_tags:
            for blk_tag in blockCate:
                y_sig += y_gap
                dat_cond = dat[
                    (dat['block']==blk_tag)&
                    (dat['setsize']==sizeN)&
                    (dat['corr']==corr_tag)&
                    (dat['exp']==exp_tag)]
                X = np.array(
                    [dat_cond.loc[(dat_cond['layer']==x_name),
                    'coeff'].values for x_name in activation_names])
                X = np.transpose(X,(1,0))

                t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                    X,n_jobs=None,threshold=t_thresh,adjacency=None,
                    n_permutations=n_permutations,out_type='indices')

                if (len(clusters)!=0):
                    if (p_values[0]<p_crit):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                 for layerN in list(clusters[0][0])]
                        if exp_tag=='exp1b':
                            lstyle = 'o'
                        else:
                            lstyle = 'x'
                        if blk_tag=='Animals':
                            lcolor = clist[0]
                        else:
                            lcolor = clist[1]

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=10,marker=lstyle)
        n += 1

    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Target','animate','inanimate','Task','LTM','STM'],
        loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_blk_rt-cate-semi_MSS.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
# block
df_glm_list = [pd.DataFrame(),pd.DataFrame()]
# each kind of correlation
for indx,corr_tag in enumerate(['mean','max']):
    print(corr_tag)

    # each layer
    for name in activation_names:
        exp_glm_list = []

        # each experiment
        for expN,exp_tag in enumerate(exp_tags):
            exp = simi_raw[simi_raw['exp']==exp_tag]
            exp_subj = list(set(exp['subj']))

            glm_subj,glm_size,glm_cond,glm_coeff,glm_corr,glm_blk = \
                [],[],[],[],[],[]

            # each block
            for blk in blockCate:
                # each MSS
                for n in sizeList:

                    # each subject
                    for k in exp_subj:
                        if (exp_tag=='exp2')&(k==12):
                            continue

                        exp_simi_indv = exp[
                            (exp['layer']==name)&
                            (exp['block']==blk)&
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
                        X = np.array(list(model.resid_response))
                        X = sm.add_constant(X)
                        model2 = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        glm_cond.append('intc')
                        glm_coeff.append(model2.params[0])
                        glm_cond.append('simi')
                        glm_coeff.append(model2.params[1])
                        glm_subj += [k]*2
                        glm_size += [n]*2
                        glm_corr += [corr_tag]*2
                        glm_blk += [blk]*2
            exp_glm_list.append(pd.DataFrame(
                {'corr':glm_corr,'block':glm_blk,'subj':glm_subj,
                 'setsize':glm_size,'cond':glm_cond,'coeff':glm_coeff}))
            exp_glm_list[expN]['exp'] = exp_tag

        df_glm_layer = pd.concat(exp_glm_list,axis=0,ignore_index=True)
        df_glm_layer['layer'] = name

        df_glm_list[indx] = pd.concat(
            [df_glm_list[indx],df_glm_layer],axis=0,ignore_index=True)
glm_data = pd.concat(df_glm_list,axis=0,ignore_index=True)

glm_ealy = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='simi')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='simi')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')]
glm_1 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[
    (glm_data['layer']=='fc_8')&
    (glm_data['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat(
    [glm_1,glm_6,glm_8],axis=0,ignore_index=True)

#
clist = ['tomato','#4C5698']
mpl.rcParams.update({'font.size':14})
dat = glm_data[(glm_data['cond']!='intc')]
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(16,9))
    ax = ax.ravel()
    n = 0
    for name in activation_names:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[
            (dat['layer']==name)&(dat['corr']==corr_tag)],
                     x='setsize',y='coeff',hue='block',style='exp',
                     markers=True,dashes=True,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='Memory Set Size')
        ax[n].set_ylabel(ylabel='Coefficients')
        ax[n].set_title('%s'%(name))
        n += 1

    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Target','animate','inanimate','Task','LTM','STM'],
        loc='lower left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_blk_rt-simi-semi_layer.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')
#
clist = ['tomato','#4C5698']
mpl.rcParams.update({'font.size':14})
dat = glm_data[(glm_data['cond']!='intc')]
for corr_tag in ['max','mean']:
    fig,ax = plt.subplots(
        1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(data=dat[
            (dat['setsize']==sizeN)&(dat['corr']==corr_tag)],
                     x='layer',y='coeff',hue='block',style='exp',
                     markers=True,dashes=True,palette=clist,
                     linewidth=2,markersize=10,err_style='bars',
                     errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')
        ax[n].set_ylabel(ylabel='Coefficients')
        ax[n].set_title('MSS %d'%(sizeN))

        y_gap = 0.008
        y_sig = -0.2-y_gap
        y_fsig = 0.31+y_gap
        for exp_tag in exp_tags:
            for blk_tag in blockCate:
                y_sig += y_gap
                dat_cond = dat[
                    (dat['block']==blk_tag)&
                    (dat['setsize']==sizeN)&
                    (dat['corr']==corr_tag)&
                    (dat['exp']==exp_tag)]
                X = np.array(
                    [dat_cond.loc[(dat_cond['layer']==x_name),
                    'coeff'].values for x_name in activation_names])
                X = np.transpose(X,(1,0))

                t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                    X,n_jobs=None,threshold=t_thresh,adjacency=None,
                    n_permutations=n_permutations,out_type='indices')

                if (len(clusters)!=0):
                    if (p_values[0]<p_crit):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1) \
                                 for layerN in list(clusters[0][0])]
                        if exp_tag=='exp1b':
                            lstyle = 'o'
                        else:
                            lstyle = 'x'
                        if blk_tag=='Animals':
                            lcolor = clist[0]
                        else:
                            lcolor = clist[1]

                        ax[n].scatter(
                            sig_x,[y_sig]*len(sig_x),c=lcolor,
                            s=10,marker=lstyle)
        n += 1

    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Target','animate','inanimate','Task','LTM','STM'],
        loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_blk_rt-simi-semi_MSS.png'%corr_tag))
    plt.show(block=True)
    plt.close('all')