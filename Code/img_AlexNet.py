#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2023.11.17
# linlin.shang@donders.ru.nl


from config import set_filepath,rootPath,figPath
import os
from PIL import Image
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torchvision import transforms,models

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

save_tag = 1
p_crit = 0.05
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
alex_output = set_filepath(rootPath,'res_all')

# 1. loading data
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
#
# alex_output = set_filepath(rootPath,'res_alex')
# load image file into list
animFileList = pd.read_csv(
    os.path.join(
        rootPath,'StimList','animList.csv'),sep=',')['stimSubCate'].tolist()
objFileList = pd.read_csv(
    os.path.join(
        rootPath,'StimList','objList.csv'),sep=',')['stimSubCate'].tolist()
imgDF = pd.DataFrame()
for imgPath in animFileList+objFileList:
    subCateDF = pd.read_csv(os.path.join(rootPath,imgPath),sep=',')
    imgDF = pd.concat([imgDF,subCateDF],axis=0,ignore_index=True)
imgPathList = imgDF['stimulus'].tolist()
imgNameList = imgDF['stimulus'].str.split('/',expand=True)[3].tolist()


# 2. AlexNet 5+3
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
alex_model = models.alexnet(pretrained=True)
model_children = list(alex_model.children())
for name,n,k in zip(activation_names,[0]*5+[2]*3,[1,4,7,9,11,2,5,6]):
    model_children[n][k].register_forward_hook(
        get_activation(name))
#
preprocess = transforms.Compose([
    transforms.Resize(256),transforms.CenterCrop(224),
    transforms.ToTensor(),transforms.Normalize(
        mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
#
layerVectors = {}
for name in activation_names:
    imgVectors = {}
    for imgPath in imgPathList:
        img = Image.open(os.path.join(rootPath,imgPath))
        input_tensor = preprocess(img)
        input_batch = torch.unsqueeze(input_tensor,dim=0)
        with torch.no_grad():
            outputs = alex_model(input_batch)
        imgName = imgPath.split('/')[3]
        imgVectors[imgName] = activation[name]
        img.close()
    layerVectors[name] = imgVectors
    print(activation[name].shape)
print('--- * --- * --- * --- * --- * ---')


# 3. correlation: RDMs for each layer
def corrMtrx(a,b):
    arr_a = np.ndarray.flatten(a)
    arr_b = np.ndarray.flatten(b)
    return stats.spearmanr(arr_a,arr_b)[0]

sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']
cateList = ['within','between']
exp1b_simi,exp2_simi = pd.DataFrame(),pd.DataFrame()

for name in activation_names:
    # 3.1 get similarity matrix
    #
    # exp.1b
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
    # get similarity between images
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
                        'simi_val_%d'%(h+1)] = corrMtrx(
                        layerVectors[name][targImg].detach().numpy()[0],
                        layerVectors[name][distrImg].detach().numpy()[0])
                    # exp1b_copy.loc[
                    #     (exp1b_copy['subj']==subj)&
                    #     (exp1b_copy['BlockN']==n)&
                    #     (exp1b_copy['imgName']==distrImg),
                    #     'simi_val_%d'%h] = np.linalg.norm(
                    #     layerVectors[layerN][targImg][0]-\
                    #     layerVectors[layerN][distrImg][0])
    simi_cols = ['simi_val_%d'%h for h in range(1,9)]
    exp1b_copy['simi_mean'] = exp1b_copy[simi_cols].mean(axis=1)
    exp1b_copy['simi_max'] = exp1b_copy[simi_cols].max(axis=1)
    exp1b_distr = exp1b_copy[
        exp1b_copy['trialType']=='distractor'].copy(deep=True)
    exp1b_distr.reset_index(drop=True,inplace=True)
    exp1b_distr['layer'] = name
    print('%s finished'%name)

    exp1b_simi = pd.concat(
        [exp1b_simi,exp1b_distr],axis=0,ignore_index=True)
if save_tag==1:
    exp1b_simi.to_csv(
        os.path.join(
            alex_output,'exp1b_simi_img.csv'),
        mode='w',header=True,index=False)
print('--- --- --- --- --- ---')
print('exp.1b finished')
print('--- * --- * --- * --- * --- * ---')

# exp.2
for name in activation_names:
    exp2_copy = exp2_clean.copy(deep=True)
    for n in range(exp2_copy.shape[0]):
        for k in range(1,9):
            targImg = exp2_copy.loc[n,'imgName']
            distrImg = exp2_copy.loc[n,'imgName%d'%k]
            if isinstance(distrImg,float):
                continue
            exp2_copy.loc[n,'simi_val_%d'%k] = corrMtrx(
                layerVectors[name][targImg].detach().numpy()[0],
                layerVectors[name][distrImg].detach().numpy()[0])
    exp2_copy['simi_mean'] = exp2_copy[simi_cols].mean(axis=1)
    exp2_copy['simi_max'] = exp2_copy[simi_cols].max(axis=1)
    exp2_distr = exp2_copy[
        exp2_copy['trialType']=='distractor'].copy(deep=True)
    exp2_distr.reset_index(drop=True,inplace=True)
    exp2_distr['layer'] = name
    print('%s finished'%name)

    exp2_simi = pd.concat(
        [exp2_simi,exp2_distr],axis=0,ignore_index=True)
if save_tag==1:
    exp2_simi.to_csv(
        os.path.join(
            alex_output,'exp2_simi_img.csv'),
        mode='w',header=True,index=False)
print('--- --- --- --- --- ---')
print('exp.2 finished')
print('--- * --- * --- * --- * --- * ---')

exp1b_simi['exp'] = 'exp1b'
exp2_simi['exp'] = 'exp2'
final_col = ['exp','subj','block','cond','setsize',
             'rt','layer','simi_mean','simi_max']
exp_simi = pd.concat(
    [exp1b_simi[final_col],exp2_simi[final_col]],
    axis=0,ignore_index=True)
if save_tag==1:
    exp_simi.to_csv(
            os.path.join(
                alex_output,'expAll_simi_img.csv'),
        mode='w',header=True,index=False)
#

# ###
exp1b_simi = pd.read_csv(os.path.join(alex_output,'exp1b_simi.csv'),sep=',')
name_conv = ['Conv 1','Conv 2','Conv 3','Conv 4','Conv 5','FC 6','FC 7','FC 8']
mpl.rcParams.update({'font.size':20})
fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(16,9))
ax = ax.ravel()
for n,name in enumerate(activation_names):
    if n==0:
        leg_tag = True
    else:
        leg_tag = False
    sns.lineplot(data=exp1b_simi[exp1b_simi['layer']==name],x='setsize',
                 y='simi_mean',hue='cond',hue_order=cateList,
                 style='cond',markers=['^','o'],dashes=[(2,2),(2,2)],
                 palette='Reds',linewidth=2,markersize=10,
                 err_style="bars",errorbar=("se",1),
                 legend=leg_tag,ax=ax[n])
    sns.lineplot(data=exp1b_simi[exp1b_simi['layer']==name],x='setsize',
                 y='simi_max',hue='cond',hue_order=cateList,
                 style='cond',markers=['^','o'],dashes=False,
                 palette='Blues',linewidth=2,markersize=10,
                 err_style="bars",errorbar=("se",1),
                 legend=leg_tag,ax=ax[n])
    ax[n].set_xticks(sizeList,labels=sizeList)
    ax[n].set_xlabel(xlabel='MSS')
    ax[n].set_ylabel(ylabel='Correlation')
    ax[n].set_title(name_conv[n])
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['Within-Category (Mean)','Between-Category(Mean)',
                'Within-Category (Max)','Between-Category(Max)'],
             loc='lower left',ncol=1,fontsize=12,
             frameon=False).set_title(None)
fig.suptitle('Visual similarity incearse with MSS')
sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'simi_exp1b_vis.png'))
plt.show(block=True)
plt.close('all')
#
dat = exp1b_simi[exp1b_simi['layer']=='conv_1'].groupby(
    ['subj','cond','setsize'])['rt'].agg(np.mean).reset_index()
fig,ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(10,9))
for n,name in enumerate(activation_names):
    sns.lineplot(data=dat,x='setsize',y='rt',hue='cond',
                 hue_order=cateList,style='cond',
                 markers=['^','o'],dashes=False,
                 palette='Blues',linewidth=2,markersize=10,
                 err_style='bars',errorbar=("se",0),
                 legend=True,ax=ax)
    ax.set_xticks(sizeList,labels=sizeList)
    ax.set_xlabel(xlabel='MSS')
    ax.set_ylabel(ylabel='RT (sec)')
    ax.set(ylim=(0.5,0.65))
    ax.set_yticks(np.arange(0.5,0.651,0.05))
h,_ = ax.get_legend_handles_labels()
ax.legend(h,['Within-Category','Between-Category'],
          loc='upper left',ncol=1,fontsize=16,
          frameon=False).set_title(None)
fig.suptitle('RT increase with MSS')
sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'simi_exp1b_rt.png'))
plt.show(block=True)
plt.close('all')


corr_tag = 'simi_mean'
# corr_tag = 'simi_max'
# plot the mean similarity
mpl.rcParams.update({'font.size':20})
for k,exp in enumerate([exp1b_simi,exp2_simi]):
    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(21,9))
    ax = ax.ravel()
    for n,name in enumerate(activation_names):
        if n==0:
            leg_tag = True
        else:
            leg_tag = False

        sns.lineplot(data=exp[exp['layer']==name],x='setsize',
                     y=corr_tag,hue='cond',hue_order=cateList,
                     style='cond',markers=['^','o'],dashes=False,
                     palette='Blues',linewidth=2,markersize=10,
                     err_style="bars",errorbar=("se",1),
                     legend=leg_tag,ax=ax[n])
        ax[1].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel(xlabel='MSS')
        ax[n].set_ylabel(ylabel='Correlation')
        ax[n].set_title(name)
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(h,['Within-Category','Between-Category'],
                 loc='lower left',ncol=1,fontsize=12,
                 frameon=False).set_title(None)
    fig.suptitle('Exp.%d'%(k+1))
    sns.despine(offset=15,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'%s_exp%d.png'%(corr_tag,k+1)))
    plt.show(block=False)
    plt.close('all')


# # 4. correlate similarity to rt
# for expN,exp in enumerate([exp1b_simi,exp2_simi]):
#     layer_names,subjs,cates,exps,setsizes,r_list,p_list,sig_list = \
#         [],[],[],[],[],[],[],[]
#     subjs_exp = [exp1b_subj,exp2_subj]
#     for name in activation_names:
#         for n in subjs_exp[expN]:
#             for h in sizeList:
#                 for k in cateList:
#                     simi = exp.loc[
#                         (exp['subj']==n)&
#                         (exp['setsize']==h)&
#                         (exp['cond']==k)&
#                         (exp['layer']==name),corr_tag].values
#                     rt = exp.loc[
#                         (exp['subj']==n)&(exp['layer']==name),'rt'].values
#
#                     r_list.append(stats.spearmanr(simi,rt)[0])
#                     p_list.append(stats.spearmanr(simi,rt)[1])
#                     if stats.spearmanr(simi,rt)[1]<p_crit:
#                         sig_tag = '*'
#                     else:
#                         sig_tag = 'ns'
#                     sig_list.append(sig_tag)
#                     subjs.append(n)
#                     setsizes.append(h)
#                     cates.append(k)
#                     layer_names.append(name)
#                     exps.append(['exp1b','exp2'][expN])
#     if expN==0:
#         exp1b_corr = pd.DataFrame(
#             {'exp':exps,'layer':layer_names,'subj':subjs,
#              'setsize':setsizes,'cate':cates,'r':r_list,
#              'p':p_list,'sig':sig_list})
#     else:
#         exp2_corr = pd.DataFrame(
#             {'exp':exps,'layer':layer_names,'subj':subjs,
#              'setsize':setsizes,'cate':cates,'r':r_list,
#              'p':p_list,'sig':sig_list})
# exp_corr = pd.concat([exp1b_corr,exp2_corr],axis=0,ignore_index=True)
# exp_corr.to_csv(os.path.join(alex_output,'corr_%s.csv'%corr_tag),
#                 sep=',',mode='w',header=True,index=False)


# # 5. GLM
# print(corr_tag)
# exp1b_simi = pd.read_csv(
#     os.path.join(alex_output,'exp1b_simi.csv'),sep=',')
# exp2_simi = pd.read_csv(
#     os.path.join(alex_output,'exp2_simi.csv'),sep=',')
#
# import statsmodels.api as sm
# from sklearn import preprocessing
#
# df_glm = pd.DataFrame()
# for name in activation_names:
#     for exp_tag in ['exp1b','exp2']:
#         if exp_tag=='exp1b':
#             exp = exp1b_simi
#             exp_subj = exp1b_subj
#         else:
#             exp = exp2_simi
#             exp_subj = exp2_subj
#         exp['cond_trans'] = np.where(
#             exp['cond']=='within',1,-1)
#
#         glm_subj,glm_size,glm_cond,glm_coeff = [],[],[],[]
#         for n in sizeList:
#             for k in exp_subj:
#                 exp_simi_indv = exp[
#                     (exp['layer']==name)&
#                     (exp['setsize']==n)&
#                     (exp['subj']==k)].copy()
#
#                 exp_simi_indv['rt_Z'] = preprocessing.scale(
#                     exp_simi_indv.loc[:,'rt'])
#                 exp_simi_indv[corr_tag+'_Z'] = preprocessing.scale(
#                     exp_simi_indv.loc[:,corr_tag])
#                 exp_simi_indv['cond_Z'] = preprocessing.scale(
#                     exp_simi_indv.loc[:,'cond_trans'])
#                 exp_simi_indv['inter'] = \
#                     exp_simi_indv.loc[:,'cond_Z']*exp_simi_indv.loc[:,corr_tag+'_Z']
#                 exp_simi_indv['inter_Z'] = preprocessing.scale(
#                     exp_simi_indv.loc[:,'inter'])
#
#                 # GLM fit
#                 y = exp_simi_indv['rt_Z']
#                 X = exp_simi_indv[['cond_Z',corr_tag+'_Z','inter_Z']]
#                 X = sm.add_constant(X)
#                 model = sm.GLM(y,X,family=sm.families.Gaussian()).fit()
#
#                 if exp_tag=='exp1b':
#                     exp1b_simi.loc[(exp1b_simi['layer']==name)&
#                     (exp1b_simi['setsize']==n)&
#                     (exp1b_simi['subj']==k),'resid'] = \
#                         model.predict()-exp_simi_indv['rt_Z'].values
#                 else:
#                     exp2_simi.loc[(exp2_simi['layer']==name)&
#                     (exp2_simi['setsize']==n)&
#                     (exp2_simi['subj']==k),'resid'] = \
#                         model.predict()-exp_simi_indv['rt_Z'].values
#
#                 glm_cond.append('intc')
#                 glm_coeff.append(model.params[0])
#                 glm_cond.append('cate')
#                 glm_coeff.append(model.params[1])
#                 glm_cond.append('simi')
#                 glm_coeff.append(model.params[2])
#                 glm_cond.append('inter')
#                 glm_coeff.append(model.params[3])
#                 glm_subj += [k]*4
#                 glm_size += [n]*4
#         if exp_tag=='exp1b':
#             df1b_glm = pd.DataFrame(
#                 {'subj':glm_subj,'setsize':glm_size,
#                  'cond':glm_cond,'coeff':glm_coeff})
#             df1b_glm['exp'] = [exp_tag]*len(df1b_glm)
#             df1b_glm['layer'] = [name]*len(df1b_glm)
#         else:
#             df2_glm = pd.DataFrame(
#                 {'subj':glm_subj,'setsize':glm_size,
#                  'cond':glm_cond,'coeff':glm_coeff})
#             df2_glm['exp'] = [exp_tag]*len(df2_glm)
#             df2_glm['layer'] = [name]*len(df2_glm)
#
#     df_glm_layer = pd.concat(
#         [df1b_glm,df2_glm],axis=0,ignore_index=True)
#     df_glm_layer['layer'] = name
#     df_glm = pd.concat([df_glm,df_glm_layer],axis=0,ignore_index=True)
# if save_tag==1:
#     df_glm.to_csv(os.path.join(alex_output,'glm_%s.csv'%corr_tag),
#                   sep=',',mode='w',header=True,index=False)
# # if corr_tag=='simi_mean':
# #     df_glmAll = df_glm.copy(deep=True)
# #     df_glmAll.rename(columns={'coeff':'coeff_mean'},inplace=True)
# # #
# # df_glmAll['coeff_max'] = df_glm['coeff']
#
#
# # import pingouin as pg
# pd.set_option('display.max_columns',None)
#
# print('single-sample ttest: compare with 0')
# for exp_tag in ['exp1b','exp2']:
#     for name in activation_names:
#         print(exp_tag,name)
#         print('--- --- --- --- --- ---')
#         for n in sizeList:
#             df_glm_exp = df_glm[
#                 (df_glm['exp']==exp_tag)&
#                 (df_glm['layer']==name)&
#                 (df_glm['setsize']==n)]
#
#             print('MSS == %d'%n)
#
#             # res = pg.ttest(
#             #     df_glm_exp.loc[df_glm_exp['cond']=='cate','coeff'].values,0)
#             res = stats.ttest_1samp(
#                 df_glm_exp.loc[df_glm_exp['cond']=='cate','coeff'].values,
#                 popmean=0,alternative='two-sided')
#             if res[1]<p_crit:
#                 sig_tag = '*'
#             else:
#                 sig_tag = 'ns'
#             print('1. category effect %s'%sig_tag)
#             # print(res)
#
#             res = stats.ttest_1samp(
#                 df_glm_exp.loc[df_glm_exp['cond']=='simi','coeff'].values,
#                 popmean=0,alternative='two-sided')
#             if res[1]<p_crit:
#                 sig_tag = '*'
#             else:
#                 sig_tag = 'ns'
#             print('2. similarity effect %s'%sig_tag)
#             # print(res)
#
#             res = stats.ttest_1samp(
#                 df_glm_exp.loc[df_glm_exp['cond']=='inter','coeff'].values,
#                 popmean=0,alternative='two-sided')
#             if res[1]<p_crit:
#                 sig_tag = '*'
#             else:
#                 sig_tag = 'ns'
#             print('3. interaction %s'%sig_tag)
#             # print(res)
#         print('--- * --- * --- * --- * --- * ---')
#
# print('ind ttest: compare two tasks')
# for name in activation_names:
#     # print('--- --- --- --- --- ---')
#     for n in sizeList:
#         df_glm_exp = df_glm[
#             (df_glm['layer']==name)&
#             (df_glm['setsize']==n)].copy()
#
#         print('%s MSS == %d'%(name,n))
#
#         # res = pg.ttest(
#         #     df_glm_exp.loc[df_glm_exp['cond']=='cate','coeff'].values,0)
#         res = stats.ttest_ind(
#             df_glm_exp.loc[
#                 (df_glm_exp['exp']=='exp1b')&
#                 (df_glm_exp['cond']=='cate'),'coeff'].values,
#             df_glm_exp.loc[
#                 (df_glm_exp['exp']=='exp2')&
#                 (df_glm_exp['cond']=='cate'),'coeff'].values,)
#         if res[1]<p_crit:
#             sig_tag = '*'
#         else:
#             sig_tag = 'ns'
#         print('1. category effect %s'%sig_tag)
#         # print(res)
#
#         res = stats.ttest_ind(
#             df_glm_exp.loc[
#                 (df_glm_exp['exp']=='exp1b')&
#                 (df_glm_exp['cond']=='simi'),'coeff'].values,
#             df_glm_exp.loc[
#                 (df_glm_exp['exp']=='exp2')&
#                 (df_glm_exp['cond']=='simi'),'coeff'].values,)
#         if res[1]<p_crit:
#             sig_tag = '*'
#         else:
#             sig_tag = 'ns'
#         print('2. similarity effect %s'%sig_tag)
#         # print(res)
#
#         res = stats.ttest_ind(
#             df_glm_exp.loc[
#                 (df_glm_exp['exp']=='exp1b')&
#                 (df_glm_exp['cond']=='inter'),'coeff'].values,
#             df_glm_exp.loc[
#                 (df_glm_exp['exp']=='exp2')&
#                 (df_glm_exp['cond']=='inter'),'coeff'].values,)
#         if res[1]<p_crit:
#             sig_tag = '*'
#         else:
#             sig_tag = 'ns'
#         print('3. interaction %s'%sig_tag)
#         # print(res)
#     print('--- * --- * --- * --- * --- * ---')

exp_simi_temp = exp_simi.drop(columns='simi_max',axis=1)
exp_simi_temp.rename(columns={'simi_mean':'simi'},inplace=True)
exp_simi_temp['corr'] = 'mean'
exp_simi.drop(columns='simi_mean',axis=1,inplace=True)
exp_simi.rename(columns={'simi_max':'simi'},inplace=True)
exp_simi['corr'] = 'max'
exp_simi = pd.concat(
    [exp_simi,exp_simi_temp],axis=0,ignore_index=True)

# plot
mpl.rcParams.update({'font.size':20})
fig,ax = plt.subplots(
    2,4,sharex=True,sharey=True,figsize=(18,9))
ax = ax.ravel()
for n,name in enumerate(activation_names):
    if n==0:
        leg_tag = True
    else:
        leg_tag = False

    sns.lineplot(
        data=exp_simi[exp_simi['layer']==name],x='setsize',
        y='simi',hue='cate',hue_order=cateList,style='corr',
        markers=['^','s'],dashes=True,palette='Blues',
        linewidth=2,markersize=10,err_style='bars',
        errorbar=('se',1),legend=leg_tag,ax=ax[n])
    ax[n].set_xticks(sizeList,labels=sizeList)
    ax[n].set_xlabel(xlabel='Memory Set Size')
    ax[n].set_ylabel(ylabel='Similarity')
    ax[n].set_title(name)
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(
    h,['Category','within','between',
       'Correlation','max','mean'],
    loc='lower right',ncol=2,fontsize=12,
    frameon=False).set_title(None)

sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(
    os.path.join(figPath,'paired_corr.png'))
plt.show(block=False)
plt.close('all')
