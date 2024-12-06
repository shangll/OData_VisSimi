#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2 (Behavioural):
# Modeling
# 2023.10.24
# linlin.shang@donders.ru.nl


from config import figPath
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


filePath = os.getcwd()
col_list = ['subj','block','trialType','cond','setsize','rt','acc','resp']
sizeList = [1,2,4,8]
crit_sd,crit_rt,crit_acc = 3,0.2,0.7
exp_tags = ['exp1b','exp2']
trial_cond_list = ['within','between','target']


# 1. raw data loaded
exp1b_raw = pd.read_csv(
    os.path.join(filePath,'exp1b_Raw.csv'),sep=',')
exp1b_raw['trialType'] = np.where(
    exp1b_raw['cond']=='target','target','distractor')
exp1b = exp1b_raw[col_list+['schImg']].copy(deep=True)
exp1b.rename(columns={'schImg':'image'},inplace=True)
exp1b['subcate'] = exp1b['image'].str.split('/',expand=True)[2]
exp1b['imgName'] = exp1b['image'].str.split('/',expand=True)[3]

# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# exp1b_raw.head()

exp2_raw = pd.read_csv(
    os.path.join(filePath,'exp2_Raw.csv'),sep=',')
exp2 = exp2_raw[
    col_list+[
        'stdImg0','stdImg1','stdImg2','stdImg3',
        'stdImg4','stdImg5','stdImg6','stdImg7',
        'testImg']].copy(deep=True)
exp2.rename(columns={'testImg':'image'},inplace=True)
exp2 = exp2[col_list+['image','stdImg0','stdImg1','stdImg2',
                      'stdImg3','stdImg4','stdImg5','stdImg6',
                      'stdImg7']].copy(deep=True)
exp2['subcate'] = exp2['image'].str.split('/',expand=True)[2]
exp2['imgName'] = exp2['image'].str.split('/',expand=True)[3]
exp2['imgName1'] = exp2['stdImg0'].str.split('/',expand=True)[3]
exp2['imgName2'] = exp2['stdImg1'].str.split('/',expand=True)[3]
exp2['imgName3'] = exp2['stdImg2'].str.split('/',expand=True)[3]
exp2['imgName4'] = exp2['stdImg3'].str.split('/',expand=True)[3]
exp2['imgName5'] = exp2['stdImg4'].str.split('/',expand=True)[3]
exp2['imgName6'] = exp2['stdImg5'].str.split('/',expand=True)[3]
exp2['imgName7'] = exp2['stdImg6'].str.split('/',expand=True)[3]
exp2['imgName8'] = exp2['stdImg7'].str.split('/',expand=True)[3]
# # for exp.2 we also remove the outlier trials based on encoding time
# outRTs = exp2.groupby(
#     ['subj','setsize'])['encoding'].transform(
#     lambda x: stats.zscore(x))
# exp2.loc[np.where(np.abs(outRTs)>crit_sd)[0],'acc'] = 0



# 2. data cleaning
# 2.1 RT
out_subjs_rt = []
for indx,exp in enumerate([exp1b,exp2]):
    #
    raw_0 = len(exp[exp['acc']==0])

    # 2.1.1 <0.2 sec
    exp.loc[exp['acc']==1,'acc'] = \
        np.where((exp.loc[(exp['acc']==1),'rt']<crit_rt),
                 0,1)

    # 2.1.2 ±3 sd
    outRTs = exp[exp['acc']==1].copy(deep=True).groupby(
        ['subj','cond','setsize'])['rt'].transform(
        lambda x:stats.zscore(x))
    exp.loc[np.where(np.abs(outRTs)>crit_sd)[0],'acc'] = 0

    exp_mean = exp.groupby(
        ['subj','cond','setsize'])['rt'].agg(np.mean).reset_index()
    outRTs_mean = exp_mean.groupby(
        ['cond','setsize'])['rt'].transform(
        lambda x: stats.zscore(x))
    out_rt = list(set(
        exp_mean.loc[np.where(np.abs(outRTs_mean)>crit_sd)[0],
        'subj'].tolist()))
    print('ourliers (RT):', out_rt)
    out_subjs_rt += out_rt

    clean_0 = len(exp[exp['acc'] == 0])
    print('%.3f%%' % ((clean_0-raw_0)/len(exp)*100))


# 2.2 ACC
# # exp.1b
# # tag = 'cond'
# tag = 'trialType'
# check_outlier = exp1b.groupby(
#     ['subj',tag,'setsize'])['acc'].agg(np.mean).reset_index()
# print(
#     check_outlier.loc[
#         check_outlier['acc']<crit_acc,
#         ['subj',tag,'setsize','acc']])
# outlier_1b = list(set(
#     check_outlier.loc[
#         (check_outlier['acc']<crit_acc)&
#         (check_outlier[tag]!='target'),'subj']))
# print(outlier_1b)
# exp1b = exp1b[~(exp1b['subj'].isin(outlier_1b))]
# exp1b.reset_index(drop=True,inplace=True)
# # exp.2
# check_outlier = exp2.groupby(
#     ['subj',tag,'setsize'])['acc'].agg(np.mean).reset_index()
# print(
#     check_outlier.loc[
#         check_outlier['acc']<crit_acc,
#         ['subj',tag,'setsize','acc']])
# outlier_2 = list(set(
#     check_outlier.loc[
#         (check_outlier['acc']<0.5)&
#         # (check_outlier['setsize']<8)&
#         (check_outlier[tag]!='target'),'subj']))
# print(outlier_2)
# exp2 = exp2[~(exp2['subj'].isin(outlier_2))]
# exp2.reset_index(drop=True,inplace=True)

check_outlier = exp1b.groupby(
    ['subj','setsize'])['acc'].agg(np.mean).reset_index()
print(
    check_outlier.loc[
        check_outlier['acc']<crit_acc,
        ['subj','setsize','acc']])
outlier_1b = list(set(
    check_outlier.loc[
        (check_outlier['acc']<crit_acc),'subj']))
print(outlier_1b)
exp1b = exp1b[~(exp1b['subj'].isin(outlier_1b))]
exp1b.reset_index(drop=True,inplace=True)
# exp.2
check_outlier = exp2.groupby(
    ['subj','setsize'])['acc'].agg(np.mean).reset_index()
print(
    check_outlier.loc[
        check_outlier['acc']<crit_acc,
        ['subj','setsize','acc']])
outlier_2 = list(set(
    check_outlier.loc[
        (check_outlier['acc']<crit_acc),'subj']))
print(outlier_2)
exp2 = exp2[~(exp2['subj'].isin(outlier_2))]
exp2.reset_index(drop=True,inplace=True)

#
#
#
exp1b_raw_0 = len(exp1b_raw[
                      (exp1b_raw['acc']==0)&
                      (~(exp1b_raw['subj'].isin(outlier_1b)))])
exp1b_0 = len(exp1b[exp1b['acc']==0])
exp2_raw_0 = len(exp2_raw[
                     (exp2_raw['acc']==0)&
                     (~(exp2_raw['subj'].isin(outlier_2)))])
exp2_0 = len(exp2[exp2['acc']==0])

print('exp.1b: %.3f%%'%((exp1b_0-exp1b_raw_0)/len(exp1b)*100))
print('exp.2: %.3f%%'%((exp2_0-exp2_raw_0)/len(exp2)*100))

# exp1b.to_csv(os.path.join(filePath,'exp1b_clean.csv'),
#              mode='w',header=True,index=False)
# exp2.to_csv(os.path.join(filePath,'exp2_clean.csv'),
#             mode='w',header=True,index=False)



pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


#
# plot
#
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

exp1b['exp'] = 'exp1b'
exp2['exp'] = 'exp2'
col_names = ['exp','subj','block','trialType','cond','setsize','rt','acc']
expAll = pd.concat([exp1b[col_names],exp2[col_names]],axis=0,ignore_index=True)

# mean
exp1b_mean_subjs = exp1b[exp1b['acc']==1].groupby(
    ['subj','setsize','cond'])['rt'].agg(np.mean).reset_index().round(3)
exp1b_meanACC_subjs = exp1b.groupby(
    ['subj','setsize','cond'])['acc'].agg(np.mean).reset_index().round(3)
exp1b_mean_subjs['acc'] = exp1b_meanACC_subjs['acc']
exp2_mean_subjs = exp2[exp2['acc']==1].groupby(
    ['subj','setsize','cond'])['rt'].agg(np.mean).reset_index().round(3)
exp2_meanACC_subjs = exp2.groupby(
    ['subj','setsize','cond'])['acc'].agg(np.mean).reset_index().round(3)
exp2_mean_subjs['acc'] = exp2_meanACC_subjs['acc']
exp1b_mean_subjs['exp'] = 'exp1b'
exp2_mean_subjs['exp'] = 'exp2'
expMean = pd.concat([exp1b_mean_subjs,exp2_mean_subjs],axis=0,ignore_index=True)
expMean['setsize'] = expMean['setsize'].astype(int)


# cList = ['tomato','dodgerblue']
mpl.rcParams.update({'font.size':20})
fig,ax = plt.subplots(
    2,2,figsize=(20,16))
ax = ax.ravel()
indx = 0
for y_var in ['rt','acc']:
    for exp_tag in exp_tags:
        if indx==0 or indx==2:
            leg_tag = True
        else:
            leg_tag = False

        if y_var=='rt':
            exp_plt = expMean[
                (expMean['exp']==exp_tag)&
                (expMean['cond']!='target')]
            exp_plt['setsize'] = exp_plt['setsize'].astype(str)
        else:
            exp_plt = expMean[
                (expMean['exp']==exp_tag)&
                (expMean['cond']!='target')]

        if y_var=='rt':
            sns.barplot(
                data=exp_plt,x='setsize',y=y_var,hue='cond',
                hue_order=['within','between'],errorbar='se',
                capsize=0.15,errcolor='grey',errwidth=1.5,
                palette='Blues',legend=leg_tag,ax=ax[indx])
        else:
            sns.lineplot(
                data=exp_plt,x='setsize',y=y_var,hue='cond',
                hue_order=['within','between'],err_style='bars',
                markers=True,linestyle='-',palette='Blues',
                errorbar=('sd',0),markersize=16,style='cond',
                style_order=['within','between'],
                dashes=False,linewidth=5,legend=leg_tag,ax=ax[indx])

        if y_var=='rt':
            y_name = 'RT (sec)'
            ax[indx].set_ylim(0,0.85)
            ax[indx].set_xticks(
                ticks=['1','2','4','8'])
            if indx==0:
                fig_lab = '(A)'
                ax[indx].set_yticks(
                    np.arange(0,0.81,0.2))
            else:
                fig_lab = '(B)'
                ax[indx].set_yticks(
                    np.arange(0,0.81,0.2),labels=[])
            fig_y = 0.92
            ax[indx].text(-0.8,fig_y,fig_lab,ha='center',va='top',color='k')

        else:
            y_name = 'ACC'
            ax[indx].set_ylim(0.7,1)
            ax[indx].set_yticks(
                np.arange(0.7,1,0.1))
            ax[indx].set_xticks(
                ticks=sizeList,labels=sizeList)
            if indx==2:
                fig_lab = '(C)'
            else:
                fig_lab = '(D)'
                ax[indx].set_yticks(
                    np.arange(0.7,1,0.1),labels=[])
            fig_y = 1.05
            ax[indx].text(0.05,fig_y,fig_lab,ha='center',va='top',color='k')

        # ax[indx].set_xticks(
        #     [str(str_num) for str_num in sizeList],labels=sizeList)

        ax[indx].set_xlabel('Memory Set Size')
        if indx==0 or indx==2:
            ax[indx].set_ylabel(y_name)
            ax[indx].set_title('LTM')
        else:
            ax[indx].set_ylabel('')
            ax[indx].set_title('STM')

        indx += 1
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(
    h,['within','between'],loc='upper left',ncol=1,
    fontsize=16,frameon=True).set_title(None)
h,_ = ax[2].get_legend_handles_labels()
ax[2].legend(
    h,['within','between'],loc='lower left',ncol=1,
    fontsize=16,frameon=True).set_title(None)

sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figPath,'behav_descr_barln.tif'))
plt.show(block=True)
plt.close('all')




# cList = ['tomato','dodgerblue']
mpl.rcParams.update({'font.size':20})
fig,ax = plt.subplots(
    2,2,sharex=True,figsize=(18,18))
ax = ax.ravel()
indx = 0
for y_var in ['rt','acc']:
    for exp_tag in exp_tags:
        if indx==0:
            leg_tag = True
        else:
            leg_tag = False

        if y_var=='rt':
            exp_plt = expMean[
                (expMean['exp']==exp_tag)&
                (expMean['cond']!='target')]
        else:
            exp_plt = expMean[
                (expMean['exp']==exp_tag)&
                (expMean['cond']!='target')]
        sns.barplot(
            data=exp_plt,x='setsize',y=y_var,hue='cond',
            hue_order=['within','between'],errorbar='se',
            capsize=0.15,errcolor='grey',errwidth=1.5,
            palette='Blues',legend=leg_tag,ax=ax[indx])

        if y_var=='rt':
            y_name = 'RT (sec)'
            ax[indx].set_ylim(0,0.85)
            if indx==0:
                fig_lab = '(A)'
                ax[indx].set_yticks(
                    np.arange(0,0.81,0.2))
            else:
                fig_lab = '(B)'
                ax[indx].set_yticks(
                    np.arange(0,0.81,0.2),labels=[])
            fig_y = 0.92
        else:
            y_name = 'ACC'
            ax[indx].set_ylim(0.65,1)
            if indx==2:
                fig_lab = '(C)'
            else:
                fig_lab = '(D)'
                ax[indx].set_yticks(
                    np.arange(0.65,1,0.1),labels=[])
            fig_y = 1.05

        ax[indx].set_xticks(
            [str(str_num) for str_num in sizeList],labels=sizeList)

        ax[indx].set_xlabel('Memory Set Size')
        if indx==0 or indx==2:
            ax[indx].set_ylabel(y_name)
            ax[indx].set_title('LTM')
        else:
            ax[indx].set_ylabel('')
            ax[indx].set_title('STM')
        ax[indx].text(-0.8,fig_y,fig_lab,ha='center',va='top',color='k')

        indx += 1
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(
    h,['within','between'],loc='lower left',ncol=1,
    fontsize=12,frameon=True).set_title(None)

sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figPath,'behav_descr_2exp.tif'))
plt.show(block=True)
plt.close('all')

# predict
for exp_tag in exp_tags:
    exp_data = expMean[expMean['exp']==exp_tag]
    subj_list = list(set(exp_data['subj'].tolist()))

    for n in subj_list:
        for cond in trial_cond_list:
            df_train = exp_data[
                (exp_data['subj']==n)&
                (exp_data['cond']==cond)&
                (exp_data['setsize']!=sizeList[-1])].copy(
                deep=True).reset_index(drop=True)
            df_test = exp_data[
                (exp_data['subj']==n)&
                (exp_data['cond']==cond)].copy(
                deep=True).reset_index(drop=True)

            # linear
            x = df_train['setsize'].values
            y = list(df_train['rt'].values)
            model = sm.OLS(y,sm.add_constant(x)).fit()
            pred_value = df_test['setsize'].values
            pred_res = model.predict(sm.add_constant(pred_value))
            '''
            model.fit(x.reshape(-1,1),y)
            pred_value = df_test['setsize'].values
            pred_res = model.predict(pred_value.reshape(-1,1))
            '''
            expMean = expMean.copy()
            expMean.loc[
                (expMean['exp']==exp_tag)&
                (expMean['subj']==n)&
                (expMean['cond']==cond),'lm'] = pred_res

            # log2
            x = df_train['setsize'].apply(np.log2).values
            y = list(df_train['rt'].values)
            model = sm.OLS(y,sm.add_constant(x)).fit()
            pred_value = df_test['setsize'].apply(np.log2).values
            pred_res = model.predict(sm.add_constant(pred_value))
            '''
            model.fit(x.reshape(-1,1),y)
            pred_value = df_test['setsize'].apply(np.log2).values
            pred_res = model.predict(pred_value.reshape(-1,1))
            '''
            expMean.loc[
                (expMean['exp']==exp_tag)&
                (expMean['subj']==n)&
                (expMean['cond']==cond),'log'] = pred_res

expMean.to_csv(
        os.path.join(filePath,'exp_mean.csv'),mode='w',header=True,index=False)

#grp level
exp1b_mean = exp1b[exp1b['acc']==1].groupby(
    ['setsize','cond'])['rt'].agg(np.mean).reset_index().round(3)
exp1b_meanACC = exp1b.groupby(
    ['setsize','cond'])['acc'].agg(np.mean).reset_index().round(3)
exp1b_mean['acc'] = exp1b_meanACC['acc']
exp2_mean = exp2[exp2['acc']==1].groupby(
    ['setsize','cond'])['rt'].agg(np.mean).reset_index().round(3)
exp2_meanACC = exp2.groupby(
    ['setsize','cond'])['acc'].agg(np.mean).reset_index().round(3)
exp2_mean['acc'] = exp2_meanACC['acc']



