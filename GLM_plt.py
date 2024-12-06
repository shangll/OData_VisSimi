#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2024.03.05
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
w2v_output = set_filepath(rootPath,'res_w2v')
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
clist = ['#FFBA00','grey']


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
imgDF['img'] = imgNameList


image_simi_mtrx_alllayers = pd.read_csv(
    os.path.join(alex_output,'img_alex_simi.csv'),sep=',')
img_col = image_simi_mtrx_alllayers.columns.tolist()
img_col.remove('layer')
image_simi_mtrx_alllayers['image'] = img_col*8
w2v_simi_mtrx = pd.read_csv(
    os.path.join(w2v_output,'img_w2v_simi.csv'),sep=',')
anim_list = w2v_simi_mtrx.loc[
    w2v_simi_mtrx['cate']=='Animals','image'].to_list()
obj_list = w2v_simi_mtrx.loc[
    w2v_simi_mtrx['cate']=='Objects','image'].to_list()

# alexnet
val_list,cate_list,layer_list = [],[],[]
for name in activation_names:
    image_simi_mtrx = image_simi_mtrx_alllayers[
        image_simi_mtrx_alllayers['layer']==name]
    old_imgs = []
    for img_N in anim_list:
        old_imgs.append(img_N)
        for img_K in anim_list:
            if img_K not in old_imgs:
                if imgDF.loc[imgDF['img']==img_N,'subCate'].values != \
                        imgDF.loc[imgDF['img']==img_K,'subCate'].values:
                    corr_val = image_simi_mtrx.loc[
                        image_simi_mtrx['image']==img_K,img_N].values[0]
                    val_list.append(corr_val)
                    cate_list.append('within')
                    layer_list.append(name)
    for img_N in obj_list:
        old_imgs.append(img_N)
        for img_K in anim_list:
            if img_K not in old_imgs:
                if imgDF.loc[imgDF['img'] == img_N, 'subCate'].values != \
                        imgDF.loc[imgDF['img'] == img_K, 'subCate'].values:
                    corr_val = image_simi_mtrx[
                        image_simi_mtrx['image']==img_K,img_N].values[0]
                    val_list.append(corr_val)
                    cate_list.append('within')
                    layer_list.append(name)
    for img_N in obj_list:
        for img_K in anim_list:
            corr_val = image_simi_mtrx.loc[
                image_simi_mtrx['image']==img_K,img_N].values[0]
            val_list.append(corr_val)
            cate_list.append('between')
            layer_list.append(name)
img_simi_cate_alllayers = pd.DataFrame(
    {'simi':val_list,'cate':cate_list,'layer':layer_list})
img_simi_cate_alllayers.to_csv(
    os.path.join(w2v_output,'img_simi_cate.csv'),
    sep=',',mode='w',header=True,index=False)

def permu_t(grp1,grp2):
    observed_difference = np.mean(grp1)-np.mean(grp2)
    pooled = np.append(grp1,grp2)
    n_iters = 10000
    n1,n2 = len(grp1),len(grp2)
    fake_differences = np.zeros(n_iters)
    for n in np.arange(n_iters):
        shuffled = np.random.permutation(pooled)
        fake_differences[n] = np.mean(shuffled[:n1])-np.mean(shuffled[n1:])
    permutation_p = np.count_nonzero(fake_differences>=observed_difference)/n_iters
    # sampling_sd = np.std(fake_differences)
    # fake_mean = np.mean(fake_differences)
    # like_t = observed_difference/sampling_sd

    sw_errors = grp1-np.mean(grp1)
    sb_errors = grp2-np.mean(grp2)
    all_errors = np.append(sw_errors,sb_errors)
    est_error_sd = np.sqrt(np.sum(all_errors**2)/(n1+n2-2))
    sampling_sd_estimate = est_error_sd*np.sqrt(1/n1+1/n2)

    t_statistic = observed_difference/sampling_sd_estimate
    print('t = %0.3f, p = %0.3f'%(t_statistic,permutation_p))

img_simi_cate_alllayers = pd.read_csv(
    os.path.join(w2v_output,'img_simi_cate.csv'),sep=',')
for name in activation_names:
    img_simi_cate = img_simi_cate_alllayers[
        img_simi_cate_alllayers['layer']==name]
    print(name)
    print(round(img_simi_cate.loc[img_simi_cate['cate']=='within','simi'].mean(),3))
    print(round(img_simi_cate.loc[img_simi_cate['cate']=='between','simi'].mean(),3))

    i_w = img_simi_cate.loc[(img_simi_cate['cate']=='within'),'simi'].values
    i_b = img_simi_cate.loc[(img_simi_cate['cate']=='between'),'simi'].values

    permu_t(i_w,i_b)


# w2v
val_list,cate_list = [],[]
old_imgs = []
for img_N in anim_list:
    old_imgs.append(img_N)
    for img_K in anim_list:
        if img_K not in old_imgs:
            if imgDF.loc[imgDF['img']==img_N,'subCate'].values != \
                    imgDF.loc[imgDF['img']==img_K,'subCate'].values:
                corr_val = w2v_simi_mtrx.loc[w2v_simi_mtrx['image']==img_K,img_N].values[0]
                val_list.append(corr_val)
                cate_list.append('within')
for img_N in obj_list:
    old_imgs.append(img_N)
    for img_K in anim_list:
        if img_K not in old_imgs:
            if imgDF.loc[imgDF['img'] == img_N, 'subCate'].values != \
                    imgDF.loc[imgDF['img'] == img_K, 'subCate'].values:
                corr_val = w2v_simi_mtrx.loc[w2v_simi_mtrx['image']==img_K,img_N].values[0]
                val_list.append(corr_val)
                cate_list.append('within')
for img_N in obj_list:
    for img_K in anim_list:
        corr_val = w2v_simi_mtrx.loc[w2v_simi_mtrx['image']==img_K,img_N].values[0]
        val_list.append(corr_val)
        cate_list.append('between')
w2v_simi_cate = pd.DataFrame({'simi':val_list,'cate':cate_list})
print(round(w2v_simi_cate.loc[w2v_simi_cate['cate']=='within','simi'].mean(),3))
print(round(w2v_simi_cate.loc[w2v_simi_cate['cate']=='between','simi'].mean(),3))

s_w = w2v_simi_cate.loc[(w2v_simi_cate['cate']=='within'),'simi'].values
s_b = w2v_simi_cate.loc[(w2v_simi_cate['cate']=='between'),'simi'].values

# n_permutations = 1000
# from scipy.stats import permutation_test
# def statistic(x,y,axis):
#     return np.mean(x,axis=axis)-np.mean(y,axis=axis)
# res = permutation_test(
#     (w,b),statistic,permutation_type='independent',
#     n_resamples=n_permutations,alternative='two-sided')

permu_t(s_w,s_b)

# w2v_simi_cate.to_csv(
#     os.path.join(w2v_output,'w2v_simi_cate.csv'),
#     sep=',',mode='w',header=True,index=False)
#
#
#
for name in activation_names:
    img_simi_cate = img_simi_cate_alllayers[
        img_simi_cate_alllayers['layer']==name]
    print(name)

    i_val = img_simi_cate['simi'].values
    s_val = w2v_simi_cate['simi'].values

    def statistic(x,y):
        return stats.spearmanr(x,y).statistic

    r,p = stats.spearmanr(i_val,s_val)
    # res = stats.permutation_test(
    #     (i_val,s_val),statistic,vectorized=False,n_resamples=1000,
    #     permutation_type='pairings',alternative='two-sided')
    # r,p = res.statistic,res.pvalue
    print('correlation: %0.3f, %0.3f'%(r,p))


def line_plt(dat1,dat2,refs1,refs2,whicheff,outeff_list):
    mpl.rcParams.update({'font.size':18})

    for corr_tag in ['mean','max']:
        print(corr_tag)

        fig,ax = plt.subplots(
            2,4,sharex=True,sharey=True,figsize=(19,12))
        ax = ax.ravel()
        n = 0
        out_label_list = []

        for dat,outeff,refs in zip([dat1,dat2],outeff_list,[refs1,refs2]):
            if isinstance(refs,pd.DataFrame):
                refs['layer'] = 'conv_1'

            if outeff=='v':
                out_label = '(without visual similarity)'
                out_label_list.append(out_label)
            elif outeff=='s':
                out_label = '(without semantic similarity)'
                out_label_list.append(out_label)
            elif outeff=='cv':
                out_label = '(without categorical and visual similarity)'
                out_label_list.append(out_label)
            elif outeff=='cs':
                out_label = '(without categorical and semantic similarity)'
                out_label_list.append(out_label)
            else:
                out_label = '(without categorical similarity)'
                out_label_list.append(out_label)

            for sizeN in sizeList:

                ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

                if n==0:
                    leg_tag = True
                else:
                    leg_tag = False
                sns.lineplot(
                    data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
                    x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
                    style_order=['exp1b','exp2'],markers=True,dashes=False,
                    palette=clist,linewidth=2,markersize=10,err_style='bars',
                    errorbar=('se',0),legend=leg_tag,ax=ax[n])
                if isinstance(refs,pd.DataFrame):
                    sns.scatterplot(
                        data=refs[(refs['corr']==corr_tag)&
                                  (refs['setsize']==sizeN)],
                        x='layer',y='coeff',hue='exp',
                        palette=['tomato','tomato'],markers=True,
                        s=120,style='exp',style_order=['exp1b','exp2'],
                        legend=False,ax=ax[n])

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

                    t_clust,clusters,p_values,H0 = \
                        permutation_cluster_1samp_test(
                            X,n_jobs=None,threshold=t_thresh,adjacency=None,
                            n_permutations=n_permutations,out_type='indices')
                    print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
                    print(clusters)
                    print(p_values)

                    if (len(clusters)!=0):
                        for pN in range(len(p_values)):
                            if (p_values[pN]<p_crit):
                                sig_x = ['conv_%d'%(layerN+1) \
                                             if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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

        if whicheff=='v':
            eff_label = 'Visual similarity'
        elif whicheff=='s':
            eff_label = 'Semantic similarity'
        elif (whicheff=='vs') or (whicheff=='cs'):
            eff_label = ''
        else:
            eff_label = 'Categorical Similarity'

        if whicheff not in ['vs','cs']:
            fig.text(
                0.5,0.98,'%s %s'%(eff_label,out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'%s %s'%(eff_label,out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)
        elif whicheff=='cs':
            fig.text(
                0.5,0.98,'Categorical similarity %s'%(out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'Semantic similarity %s'%(out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)
        elif whicheff=='vs':
            fig.text(
                0.5,0.98,'Visual similarity %s'%(out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'Semantic similarity %s'%(out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)

        fig.text(
            0.05,0.98,'(A)',ha='center',va='top',color='k',fontsize=18)
        fig.text(
            0.05,0.51,'(B)',ha='center',va='top',color='k',fontsize=18)
        sns.despine(offset=10,trim=True)
        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
        # plt.margins(0,0)
        plt.savefig(
            os.path.join(figPath,'glm_%s_%s.tif'%(whicheff,corr_tag)))
        plt.show(block=True)
        plt.close('all')

def cate_plt(dat1,dat2,refs1,refs2,eff_list):
    clist = ['#FFBA00','grey']

    mpl.rcParams.update({'font.size':18})
    for corr_tag in ['mean','max']:
        print(corr_tag)

        fig,ax = plt.subplots(
            2,4,sharex=True,sharey=True,figsize=(19,12))
        ax = ax.ravel()
        n = 0

        for refs,dat in zip([refs1,refs2],[dat1,dat2]):
            if isinstance(refs,pd.DataFrame):
                refs['layer'] = 'conv_1'
            for sizeN in sizeList:

                ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

                if n==0:
                    leg_tag = True
                else:
                    leg_tag = False
                sns.lineplot(
                    data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
                    x='layer',y='coeff',hue='cate',
                    hue_order=cateList,markers=True,style='exp',
                    style_order=['exp1b','exp2'],
                    dashes=True,palette='Blues',linewidth=1.5,markersize=8,
                    err_style='bars',errorbar=('se',0),legend=leg_tag,ax=ax[n])
                if isinstance(refs,pd.DataFrame):
                    sns.scatterplot(
                        data=refs[(refs['corr']==corr_tag)&
                                  (refs['setsize']==sizeN)],
                        x='layer',y='coeff',hue='cate',
                        hue_order=cateList,markers=True,style='exp',
                        palette='Reds',
                        s=120,style_order=['exp1b','exp2'],
                        legend=False,ax=ax[n])

                ax[n].set_xticks(activation_names,labels=range(1,9))
                ax[n].set_xlabel(xlabel='Layer')
                ax[n].set_ylabel(ylabel='Beta')
                # y_major_locator = MultipleLocator(0.15)
                # ax[n].yaxis.set_major_locator(y_major_locator)
                ax[n].set_title('MSS%d'%(sizeN))

                y_gap = 0.01
                y_sig = -0.1-y_gap
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

                        t_clust,clusters,p_values,H0 = \
                            permutation_cluster_1samp_test(
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
                                            if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        fig.text(
            0.5,0.98,eff_list[0],
            ha='center',va='top',color='k',fontsize=18)
        fig.text(
            0.5,0.51,eff_list[1],
            ha='center',va='top',color='k',fontsize=18)
        fig.text(
            0.05,0.98,'(A)',ha='center',va='top',color='k',fontsize=18)
        fig.text(
            0.05,0.51,'(B)',ha='center',va='top',color='k',fontsize=18)
        sns.despine(offset=10,trim=True)
        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
        # plt.margins(0,0)
        plt.savefig(
            os.path.join(figPath,'glm_2cate_%s.tif'%(corr_tag)))
        plt.show(block=True)
        plt.close('all')
#
# ANOVA lineplot
#
def aov_plt(dat1,dat2,whicheff,outeff_list):
    mpl.rcParams.update({'font.size':16})
    for corr_tag in ['mean','max']:

        fig,ax = plt.subplots(
            2,2,sharex=True,sharey=True,figsize=(18,14))
        ax = ax.ravel()
        n = 0
        out_label_list = []
        for dat,outeff in zip([dat1,dat2],outeff_list):
            if outeff=='v':
                out_label = '(without visual similarity)'
            elif outeff=='s':
                out_label = '(without semantic similarity)'
            elif outeff=='c':
                out_label = '(without categorical similarity)'
            elif outeff=='cv':
                out_label = '(without categorical and visual similarity)'
            elif outeff=='cs':
                out_label = '(without categorical and semantic similarity)'
            else:
                out_label = ' '
            out_label_list.append(out_label)

            if whicheff=='v':
                eff_label = 'Visual Similarity'
            elif whicheff=='s':
                eff_label = 'Semantic Similarity'
            elif (whicheff=='vs') or (whicheff=='cs'):
                eff_label = ''
            else:
                eff_label = 'Categorical Similarity'

            if n==0:
                leg_tag = True
                figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.5,0.98
            else:
                figN,xlab_pos,x_pos,y_pos = '(B)',0.05,0.5,0.51


            fig.text(
                xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)

            if whicheff not in ['vs','cs']:
                fig.text(
                    x_pos,y_pos,'%s %s'%(eff_label,out_label),
                    ha='center',va='top',color='k',fontsize=18)

            for exp_tag in exp_tags:
                if exp_tag=='exp1b':
                    exp_title = 'LTM'
                else:
                    exp_title = 'STM'

                if n!=0:
                    leg_tag = False

                sns.lineplot(
                    data=dat[(dat['corr']==corr_tag)&(dat['exp']==exp_tag)],
                    x='setsize',y='coeff',hue='layer',
                    hue_order=['early','late','fc_8'],style='layer',
                    markers=True,dashes=False,palette=Diona[0:3],
                    linewidth=2,markersize=12,err_style='bars',
                    errorbar=('se',1),err_kws={'capsize':10},
                    legend=leg_tag,ax=ax[n])
                ax[n].set_xticks(sizeList,labels=sizeList)
                ax[n].set_xlabel('Memory Set Size')
                ax[n].set_ylabel('Beta')
                # y_major_locator = MultipleLocator(0.1)
                # ax[n].yaxis.set_major_locator(y_major_locator)
                ax[n].set_title(exp_title)

                n += 1
        if whicheff=='cs':
            fig.text(
                0.5,0.98,'Categorical Similarity %s'%(out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'Semantic Similarity %s'%(out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)
        elif whicheff=='vs':
            fig.text(
                0.5,0.98,'Visual Similarity %s'%(out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'Semantic Similarity %s'%(out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)

        h,_ = ax[0].get_legend_handles_labels()
        ax[0].legend(
            h,['Early','Late','FC 8'],
            loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

        sns.despine(offset=10,trim=True)
        plt.subplots_adjust(
            left=0.05,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
        plt.savefig(
            os.path.join(figPath,'aov_%s_%s.tif'%(whicheff,corr_tag)))
        plt.show(block=True)
        plt.close('all')

def aov_barplt(dat1,dat2,whicheff,outeff_list):
    mpl.rcParams.update({'font.size':20})
    for corr_tag in ['mean','max']:

        fig,ax = plt.subplots(
            2,2,sharex=True,sharey=True,figsize=(16,14))
        ax = ax.ravel()
        n = 0
        out_label_list = []
        for dat,outeff in zip([dat1,dat2],outeff_list):
            if outeff=='v':
                out_label = '(without visual similarity)'
            elif outeff=='s':
                out_label = '(without semantic similarity)'
            elif outeff=='c':
                out_label = '(without categorical similarity)'
            elif outeff=='cv':
                out_label = '(without categorical and visual similarity)'
            elif outeff=='cs':
                out_label = '(without categorical and semantic similarity)'
            else:
                out_label = ' '
            out_label_list.append(out_label)

            if whicheff=='v':
                eff_label = 'Visual Similarity'
            elif whicheff=='s':
                eff_label = 'Semantic Similarity'
            elif (whicheff=='vs') or (whicheff=='cs'):
                eff_label = ''
            else:
                eff_label = 'Categorical Similarity'

            if n==0:
                leg_tag = True
                figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.5,0.98
            else:
                figN,xlab_pos,x_pos,y_pos = '(B)',0.05,0.5,0.51


            fig.text(
                xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)

            if whicheff not in ['vs','cs']:
                fig.text(
                    x_pos,y_pos,'%s %s'%(eff_label,out_label),
                    ha='center',va='top',color='k',fontsize=18)

            for exp_tag in exp_tags:
                if exp_tag=='exp1b':
                    exp_title = 'LTM'
                else:
                    exp_title = 'STM'

                if n!=0:
                    leg_tag = False

                sns.barplot(
                    data=dat[(dat['corr']==corr_tag)&(dat['exp']==exp_tag)],
                    x='setsize',y='coeff',hue='layer',
                    hue_order=['early','late','fc_8'],palette=Diona[0:3],
                    errorbar='se',capsize=0.15,errcolor='grey',legend=leg_tag,
                    ax=ax[n])
                ax[n].set_xlabel('Memory Set Size')
                ax[n].set_ylabel('Beta')
                # y_major_locator = MultipleLocator(0.1)
                # ax[n].yaxis.set_major_locator(y_major_locator)
                ax[n].set_title(exp_title)

                n += 1
        if whicheff=='cs':
            fig.text(
                0.5,0.98,'Categorical Similarity %s'%(out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'Semantic Similarity %s'%(out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)
        elif whicheff=='vs':
            fig.text(
                0.5,0.98,'Visual Similarity %s'%(out_label_list[0]),
                ha='center',va='top',color='k',fontsize=18)
            fig.text(
                0.5,0.51,'Semantic Similarity %s'%(out_label_list[1]),
                ha='center',va='top',color='k',fontsize=18)

        h,_ = ax[0].get_legend_handles_labels()
        ax[0].legend(
            h,['Early','Late','FC 8'],
            loc='upper left',ncol=1,fontsize=18,frameon=False).set_title(None)

        sns.despine(offset=10,trim=True)
        plt.subplots_adjust(
            left=0.1,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
        plt.savefig(
            os.path.join(figPath,'aov_bar_%s_%s.tif'%(whicheff,corr_tag)))
        plt.show(block=True)
        plt.close('all')

# img
# visual similarity effect
v_rt_simi = pd.read_csv(
    os.path.join(alex_output,'glm_rt-simi.csv'),sep=',')
v_rt_simi = v_rt_simi[(v_rt_simi['cond']!='intc')]
v_rt_simi_3layers = pd.read_csv(
    os.path.join(alex_output,'glm_rt-simi_3layers.csv'),sep=',')
v_rt_simi_3layers = v_rt_simi_3layers[(v_rt_simi_3layers['cond']!='intc')]

# visual similarity effect without category
v_rt_2cate = pd.read_csv(
    os.path.join(alex_output,'glm_rt-2cate.csv'),sep=',')
v_rt_2cate = v_rt_2cate[(v_rt_2cate['cond']!='intc')]
v_avg2cate = pd.read_csv(
    os.path.join(alex_output,'glm_rt-2avg.csv'),sep=',')
v_avg2cate = v_avg2cate[(v_avg2cate['cond']!='intc')]
v_avg2cate_3layers = pd.read_csv(
    os.path.join(alex_output,'glm_rt-2avg_3layers.csv'),sep=',')
v_avg2cate_3layers = v_avg2cate_3layers[(v_avg2cate_3layers['cond']!='intc')]

# visual similarity effect without semantic similarity
s_resid_simi = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-simi.csv'),sep=',')
s_resid_simi = s_resid_simi[(s_resid_simi['cond']!='intc')]
s_resid_simi_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-simi_3layers.csv'),sep=',')
s_resid_simi_3layers = s_resid_simi_3layers[(s_resid_simi_3layers['cond']!='intc')]

# visual similarity effect in 2 categories without semantic similarity
s_resid_2cate_v = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-2cate_v.csv'),sep=',')
s_resid_2cate_v = s_resid_2cate_v[(s_resid_2cate_v['cond']!='intc')]

# visual similarity effect without categorical and semantic similarity
s_resid_simi_v = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-2avg_v.csv'),sep=',')
s_resid_simi_v = s_resid_simi_v[(s_resid_simi_v['cond']!='intc')]
s_resid_simi_v_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-2avg_3layers_v.csv'),sep=',')
s_resid_simi_v_3layers = s_resid_simi_v_3layers[
    (s_resid_simi_v_3layers['cond']!='intc')]


# category effect
c_rt_2cate = pd.read_csv(
    os.path.join(alex_output,'glm_rt-cate.csv'),sep=',')
c_rt_2cate = c_rt_2cate[(c_rt_2cate['cond']!='intc')]
c_rt_cate_3layers = pd.read_csv(
    os.path.join(alex_output,'glm_rt-cate_3layers.csv'),sep=',')
c_rt_cate_3layers = c_rt_cate_3layers[(c_rt_cate_3layers['cond']!='intc')]
# category effect without visual similarity
c_resid_cate = pd.read_csv(
    os.path.join(alex_output,'glm_resid-cate.csv'),sep=',')
c_resid_cate = c_resid_cate[(c_resid_cate['cond']!='intc')]
c_resid_cate_3layers = pd.read_csv(
    os.path.join(alex_output,'glm_resid-cate_3layers.csv'),sep=',')
c_resid_cate_3layers = c_resid_cate_3layers[(c_resid_cate_3layers['cond']!='intc')]
# category effect without semantic
s_resid_cate = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-c.csv'),sep=',')
s_resid_cate = s_resid_cate[(s_resid_cate['cond']!='intc')]
s_resid_cate_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-c_3layers.csv'),sep=',')
s_resid_cate_3layers = s_resid_cate_3layers[(s_resid_cate_3layers['cond']!='intc')]
# category effect without visual&semantic
vs_resid_cate = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-c_vs.csv'),sep=',')
vs_resid_cate = vs_resid_cate[(vs_resid_cate['cond']!='intc')]
vs_resid_cate_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-c_vs_3layers.csv'),sep=',')
vs_resid_cate_3layers = vs_resid_cate_3layers[(vs_resid_cate_3layers['cond']!='intc')]



# w2v
# semantic effect
s_rt_w2v = pd.read_csv(
    os.path.join(w2v_output,'glm_rt-w2v.csv'),sep=',')
s_rt_w2v = s_rt_w2v[(s_rt_w2v['cond']!='intc')]

# semantic effect without visual similarity
s_resid_w2v = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-w2v.csv'),sep=',')
s_resid_w2v = s_resid_w2v[(s_resid_w2v['cond']!='intc')]
s_resid_w2v_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-w2v_3layers.csv'),sep=',')
s_resid_w2v_3layers = s_resid_w2v_3layers[(s_resid_w2v_3layers['cond']!='intc')]

# semantic similarity effect in 2 categories
s_rt_w2v_2cate = pd.read_csv(
    os.path.join(w2v_output,'glm_rt-s-2cate.csv'),sep=',')
s_rt_w2v_2cate = s_rt_w2v_2cate[(s_rt_w2v_2cate['cond']!='intc')]
s_rt_w2v_2avg = pd.read_csv(
    os.path.join(w2v_output,'glm_rt-s-2avg.csv'),sep=',')
s_rt_w2v_2avg = s_rt_w2v_2avg[(s_rt_w2v_2avg['cond']!='intc')]
s_rt_w2v_2avg_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_rt-s-2avg_3layers.csv'),sep=',')
s_rt_w2v_2avg_3layers = s_rt_w2v_2avg_3layers[
    (s_rt_w2v_2avg_3layers['cond']!='intc')]

# semantic similarity effect in two categories without visual similarity
s_resid_2cate_s = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-2cate_s.csv'),sep=',')
s_resid_2cate_s = s_resid_2cate_s[(s_resid_2cate_s['cond']!='intc')]

# semantic similarity effect without categorical and visual similarity
s_resid_simi_s = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-2avg_s.csv'),sep=',')
s_resid_simi_s = s_resid_simi_s[(s_resid_simi_s['cond']!='intc')]
s_resid_simi_s_3layers = pd.read_csv(
    os.path.join(w2v_output,'glm_resid-2avg_3layers_s.csv'),sep=',')
s_resid_simi_s_3layers = s_resid_simi_s_3layers[
    (s_resid_simi_s_3layers['cond']!='intc')]

# v
v_rt_simi['eff'] = 'v'
# v w/o c
v_avg2cate['eff'] = 'v w/o c'
# v w/o s
s_resid_simi['eff'] = 'v w/o s'
# v w/o c&s
s_resid_simi_v['eff'] = 'v w/o cs'
#c
c_rt_2cate['eff'] = 'c'
# c w/o v
c_resid_cate['eff'] = 'c w/o v'
# c w/o s
s_resid_cate['eff'] = 'c w/o s'
# c w/o vs
vs_resid_cate['eff'] = 'c w/o vs'
# s
s_rt_w2v['eff'] = 's'
# s w/o v
s_resid_w2v['eff'] = 's w/o v'
# s w/o c
s_rt_w2v_2avg['eff'] = 's w/o c'
# s w/o cv
s_resid_2cate_s['eff'] = 's w/o cv'




#
#
#
# plot
#
#
#

# layer 6
# v
v = v_rt_simi[v_rt_simi['layer']=='fc_6']
v['eff'] = 'v'
# v w/o c
v_c = v_avg2cate[v_avg2cate['layer']=='fc_6']
v_c['eff'] = 'v w/o c'
# v w/o s
v_s = s_resid_simi[s_resid_simi['layer']=='fc_6']
v_s['eff'] = 'v w/o s'
# v w/o c&s
v_cs = s_resid_simi_v[s_resid_simi_v['layer']=='fc_6']
v_cs['eff'] = 'visual\n(w/o cs)'
# c w/o v
c_v = c_resid_cate[c_resid_cate['layer']=='fc_6']
c_v['eff'] = 'c w/o v'
# c w/o s
c_s = s_resid_cate[s_resid_cate['layer']=='fc_6']
c_s['eff'] = 'c w/o s'
# c w/o vs
c_vs = vs_resid_cate[vs_resid_cate['layer']=='fc_6']
c_vs['eff'] = 'categorical\n(w/o vs)'
# c
c = c_rt_2cate[c_rt_2cate['layer']=='fc_6']
c['eff'] = 'c'
# s
s = s_rt_w2v[s_rt_w2v['layer']=='fc_6']
s['eff'] = 's'
# s w/o v
s_v = s_resid_w2v[s_resid_w2v['layer']=='fc_6']
s_v['eff'] = 's w/o v'
# s w/o c
s_c = s_rt_w2v_2avg[s_rt_w2v_2avg['layer']=='fc_6']
s_c['eff'] = 's w/o c'
# s w/o cv
s_cv = s_resid_2cate_s[s_resid_2cate_s['layer']=='fc_6']
s_cv['eff'] = 'semantic\n(w/o cv)'

reg_data = pd.concat(
    [v,v_c,v_s,v_cs,s_v,s_cv,c_v,c_vs],axis=0,ignore_index=True)
eff_tags = ['v','v w/o c','v w/o s','v w/o c&s',
            's w/o v','s w/o cv','c w/o v','c w/o vs']
eff_labs = ['visual','visual w/o categorical','visual w/o semantic',
            'visual w/o categorical & semantic','semantic w/o visual',
            'semantic w/o categorical & visual',
            'categorical w/o visual','categorical w/o visual & semantic']
bar_colr = Kirara[0:2]+['#48D1CC',Kirara[4]]+[Diona[2],Diona[4],Diona[1],Diona[3]]
#
# lineplot
mpl.rcParams.update({'font.size':18})
for corr_tag in ['mean','max']:
    print(corr_tag)
    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(12,9))
    ax = ax.ravel()

    for n,eff_tag in enumerate(eff_tags):
        if n==0:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(
            data=reg_data[
                (reg_data['eff']==eff_tag)&(reg_data['corr']==corr_tag)],
            hue='exp',hue_order=exp_tags,
            x='setsize',y='coeff',palette=[bar_colr[n],bar_colr[n]],
            style='exp',style_order=exp_tags,markers=True,dashes=True,
            linewidth=2,markersize=12,err_style='bars',
            errorbar=('se',1),err_kws={'capsize':10},
            legend=leg_tag,ax=ax[n])

        ax[n].set_xticks(sizeList,labels=sizeList)
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        y_major_locator = MultipleLocator(0.125)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(eff_tags[n])
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],loc='lower right',ncol=1,labelcolor=None,
        fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    fig.text(
        0.05,0.97,'(A)',ha='center',va='top',color='k',fontsize=18)
    plt.savefig(
        os.path.join(figPath,'barln_eff_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')

# barplot
mpl.rcParams.update({'font.size':18})
for corr_tag in ['mean','max']:
    print(corr_tag)
    fig,ax = plt.subplots(
        2,1,sharex=True,sharey=True,figsize=(9,9))
    ax = ax.ravel()

    for n,exp_tag in enumerate(exp_tags):
        if exp_tag=='exp1b':
            exp_title = 'LTM'
            leg_tag = True
        else:
            exp_title = 'STM'
            leg_tag = False

        sns.barplot(
            data=reg_data[(reg_data['exp']==exp_tag)&(reg_data['corr']==corr_tag)],
            x='setsize',y='coeff',hue='eff',
            hue_order=eff_tags,palette=bar_colr,errorbar='se',
            capsize=0.15,errcolor='grey',legend=leg_tag,ax=ax[n])
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        y_major_locator = MultipleLocator(0.125)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(exp_title)
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,eff_tags,loc='upper left',ncol=2,
        fontsize=14,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    fig.text(
        0.05,0.97,'(B)',ha='center',va='top',color='k',fontsize=18)
    plt.savefig(
        os.path.join(figPath,'bar_allEffs_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')





# visual similarity
refs = v_rt_simi[v_rt_simi['layer']=='conv_1'].groupby(
    ['corr','exp','setsize'])['coeff'].agg(np.mean).reset_index()
line_plt(v_avg2cate,s_resid_simi,refs,refs,'v',['c','s'])

# semantic similarity
refs = s_rt_w2v.groupby(
    ['corr','exp','setsize'])['coeff'].agg(np.mean).reset_index()
line_plt(s_resid_w2v,s_resid_simi_s,refs,refs,'s',['v','cv'])


# 2 categories
refs1 = s_rt_w2v_2cate[s_rt_w2v_2cate['layer']=='conv_1'].groupby(
    ['corr','exp','cate','setsize'])['coeff'].agg(np.mean).reset_index()
refs2 = v_rt_2cate[v_rt_2cate['layer']=='conv_1'].groupby(
    ['corr','exp','cate','setsize'])['coeff'].agg(np.mean).reset_index()
cate_plt(s_resid_2cate_s,s_resid_2cate_v,refs1,refs2,
         ['Semantic similarity (without visual similarity)',
          'Visual similarity (without semantic similarity)'])

#
cate_plt(v_rt_2cate,s_resid_2cate_v,'','',
         ['Visual similarity (in 2 categories)',
          'Visual similarity (w/o semantic similarity in 2 categories)'])

# average
refs1 = v_rt_simi[v_rt_simi['layer']=='conv_1'].groupby(
    ['corr','exp','setsize'])['coeff'].agg(np.mean).reset_index()
refs2 = s_rt_w2v.groupby(
    ['corr','exp','setsize'])['coeff'].agg(np.mean).reset_index()
line_plt(s_resid_simi_v,s_resid_simi_s,refs1,refs2,'vs',['cs','cv'])
# cs without v
line_plt(c_resid_cate,s_resid_w2v,'','','cs',['v','v'])
aov_plt(c_resid_cate_3layers,s_resid_w2v_3layers,'cs',['v','v'])
aov_barplt(c_resid_cate_3layers,s_resid_w2v_3layers,'cs',['v','v'])


# 3 subplots:
# visual similarity
# mpl.rcParams.update({'font.size':16})
# for corr_tag in ['mean','max']:
#     print(corr_tag)
#
#     fig,ax = plt.subplots(
#         2,8,sharex=True,sharey=True,figsize=(21,10))
#     ax = ax.ravel()
#     n = 0
#     out_label_list = []
#
#     for k,dat in enumerate(
#             [v_rt_simi,v_avg2cate,s_resid_simi,s_resid_simi_v]):
#         if k==0:
#             eff_label = 'Visual similarity'
#             figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.25,0.98
#         elif k==1:
#             eff_label = 'Visual similarity (w/o categorical similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(B)',0.55,0.75,0.98
#         elif k==2:
#             eff_label = 'Visual similarity (w/o semantic similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.25,0.51
#         else:
#             eff_label = 'Visual similarity (w/o categorical and semantic similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(D)',0.55,0.75,0.51
#
#         for sizeN in sizeList:
#             ax[n].axhline(0,color='black',lw=1,linestyle='dashed')
#
#             if n==0:
#                 leg_tag = True
#             else:
#                 leg_tag = False
#             sns.lineplot(
#                 data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
#                 x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
#                 style_order=['exp1b','exp2'],markers=True,dashes=False,
#                 palette=clist,linewidth=2,markersize=10,err_style='bars',
#                 errorbar=('se',0),legend=leg_tag,ax=ax[n])
#
#             ax[n].set_xticks(activation_names,labels=range(1,9))
#             ax[n].set_xlabel(xlabel='Layer')
#             ax[n].set_ylabel(ylabel='Beta')
#             # y_major_locator = MultipleLocator(0.15)
#             # ax[n].yaxis.set_major_locator(y_major_locator)
#             ax[n].set_title('MSS%d'%(sizeN),fontsize=14)
#
#             y_gap = 0.01
#             y_sig = -0.1-y_gap
#             y_fsig = 0.265+y_gap
#             for exp_tag in exp_tags:
#                 y_sig += y_gap
#                 dat_cond = dat[(dat['exp']==exp_tag)&
#                                (dat['setsize']==sizeN)&
#                                (dat['corr']==corr_tag)]
#                 X = np.array(
#                     [dat_cond.loc[(dat_cond['layer']==x_name),
#                     'coeff'].values for x_name in activation_names])
#                 X = np.transpose(X,(1,0))
#
#                 t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
#                     X,n_jobs=None,threshold=t_thresh,adjacency=None,
#                     n_permutations=n_permutations,out_type='indices')
#                 print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
#                 print(clusters)
#                 print(p_values)
#
#                 if (len(clusters)!=0):
#                     for pN in range(len(p_values)):
#                         if (p_values[pN]<p_crit):
#                             sig_x = ['conv_%d'%(layerN+1) \
#                                          if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
#                                      for layerN in list(clusters[pN][0])]
#                             if exp_tag=='exp1b':
#                                 lcolor = clist[0]
#                                 lstyle = 'o'
#                             else:
#                                 lcolor = clist[1]
#                                 lstyle = 'x'
#
#                             ax[n].scatter(
#                                 sig_x,[y_sig]*len(sig_x),c=lcolor,
#                                 s=10,marker=lstyle)
#             n += 1
#         fig.text(
#             x_pos,y_pos,eff_label,ha='center',va='top',color='k',fontsize=18)
#         fig.text(
#             xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)
#     h,_ = ax[0].get_legend_handles_labels()
#     ax[0].legend(
#         h,['LTM','STM'],
#         loc='upper left',ncol=1,frameon=False).set_title(None)
#     sns.despine(offset=10,trim=True)
#     plt.tight_layout()
#     plt.subplots_adjust(left=0.1,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
#     # plt.margins(0,0)
#     plt.savefig(
#         os.path.join(figPath,'glm_viseff_%s.tif'%(corr_tag)))
#     plt.show(block=True)
#     plt.close('all')

#
v_rt_simi['simi']='v'
v_avg2cate['simi']='v w/o c'
s_resid_simi['simi']='v w/o s'
s_resid_simi_v['simi']='v w/o c&s'
simi_all = pd.concat(
    [v_rt_simi,v_avg2cate,s_resid_simi,s_resid_simi_v],
    ignore_index=True,axis=0)
for act_n in range(0,8):
    simi_all.loc[simi_all['layer']==activation_names[act_n],'layer']=act_n+1
mpl.rcParams.update({'font.size': 21})
for k,dat in enumerate(
        [v_rt_simi,v_avg2cate,s_resid_simi,s_resid_simi_v]):

    fig,ax = plt.subplots(
        1,4,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    n = 0
    out_label_list = []


    if k==0:
        eff_label = 'Visual similarity'
        figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.55,1
    elif k==1:
        eff_label = 'Visual similarity (w/o categorical similarity)'
        figN,xlab_pos,x_pos,y_pos = '(B)',0.05,0.55,1
    elif k==2:
        eff_label = 'Visual similarity (w/o semantic similarity)'
        figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.55,1
    else:
        eff_label = 'Visual similarity (w/o categorical and semantic similarity)'
        figN,xlab_pos,x_pos,y_pos = '(D)',0.05,0.55,1

    for sizeN in sizeList:
        ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

        if n==0 or n==4 or n==8 or n==12:
            leg_tag = True
        else:
            leg_tag = False
        sns.lineplot(
            data=dat[(dat['corr']=='mean')&(dat['setsize']==sizeN)],
            x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
            style_order=['exp1b','exp2'],markers=True,dashes=False,
            palette=clist,linewidth=3,markersize=10,err_style='bars',
            errorbar=('se',0),legend=leg_tag,ax=ax[n])
        ax[n].set_ylabel(ylabel='Beta')
        ax[n].set_yticks(np.arange(-0.1,0.25,0.125))

        ax[n].set_xticks(activation_names,labels=range(1,9))
        ax[n].set_xlabel(xlabel='Layer')

        # y_major_locator = MultipleLocator(0.15)
        # ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title('MSS%d'%(sizeN),fontsize=18)

        y_gap = 0.01
        y_sig = -0.1-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp']==exp_tag)&
                           (dat['setsize']==sizeN)&
                           (dat['corr']=='mean')]
            X = np.array(
                [dat_cond.loc[(dat_cond['layer']==x_name),
                'coeff'].values for x_name in activation_names])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,tail=1,
                n_permutations=n_permutations,out_type='indices')
            # print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<p_crit):
                        sig_x = ['conv_%d'%(layerN+1) \
                                     if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        if n==0 or n==4 or n==8 or n==12:
            h,_ = ax[n].get_legend_handles_labels()
            ax[n].legend(
                h,['LTM','STM'],
                loc='upper left',ncol=1,frameon=False,fontsize=20).set_title(None)
        n += 1
    fig.text(
        x_pos,y_pos,eff_label,ha='center',va='top',
        color='k',fontdict={'weight':'bold'})
    fig.text(
        xlab_pos,y_pos,figN,ha='center',va='top',color='k')
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.099,right=1,top=0.88,bottom=0.15)
    # # plt.margins(0,0)
    plt.savefig(
        os.path.join(figPath,'glm_viseff_mean%d.tif'%(k)))
    plt.show(block=True)
    plt.close('all')


#
#
#
mpl.rcParams.update({'font.size': 21})
for k,dat in enumerate(
        [s_rt_w2v,s_resid_w2v,s_rt_w2v_2avg,s_resid_2cate_s]):

    if k==0:
        eff_label = 'Semantic similarity'
        figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.55,1
        fig_w,fig_h,col_n = 9,6,1
    elif k==1:
        eff_label = 'Semantic similarity (w/o visual similarity)'
        figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.55,1
        fig_w,fig_h,col_n = 18,6,4
    elif k==2:
        eff_label = 'Semantic similarity (w/o categorical similarity)'
        figN,xlab_pos,x_pos,y_pos = '(B)',0.05,0.55,1
        fig_w,fig_h,col_n = 9,6,1
    else:
        eff_label = 'Semantic similarity (w/o categorical and visual similarity)'
        figN,xlab_pos,x_pos,y_pos = '(D)',0.05,0.55,1
        fig_w,fig_h,col_n = 18,6,4

    fig,ax = plt.subplots(
        1,col_n,sharex=True,sharey=True,figsize=(fig_w,fig_h))
    n = 0
    out_label_list = []

    if k == 0 or k == 2:
        ax.axhline(0,color='black',lw=1,linestyle='dashed')
        sns.lineplot(
            data=dat[(dat['corr'] == 'mean')],
            x='setsize', y='coeff', hue='exp', style='exp', hue_order=exp_tags,
            style_order=['exp1b', 'exp2'], markers=True, dashes=False,
            palette=clist, linewidth=3, markersize=10, err_style='bars',
            errorbar=('se', 0), legend=True, ax=ax)
        ax.set(ylim=(-0.1,0.31))
        ax.set_yticks(np.arange(-0.1,0.31,0.1))
        ax.set_xticks([1,2,4,8],labels=[1,2,4,8])
        ax.set_xlabel(xlabel='MSS')
        ax.set_ylabel(ylabel='Beta')
        if k==2:
            ax.set_ylabel(ylabel=None)

        y_gap = 0.01
        y_sig = -0.06 - y_gap
        y_fsig = 0.265 + y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp'] == exp_tag) &
                           (dat['corr'] == 'mean')]
            X = np.array(
                [dat_cond.loc[(dat_cond['setsize'] == x_name),
                'coeff'].values for x_name in sizeList])
            X = np.transpose(X, (1, 0))

            t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,tail=1,
                n_permutations=n_permutations,out_type='indices')
            print(clusters)
            print(p_values)

            if (len(clusters) != 0):
                for pN in range(len(p_values)):
                    if (p_values[pN] < p_crit):
                        sig_x = [sizeList[sizeN] for sizeN in list(clusters[pN][0])]
                        if exp_tag == 'exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax.scatter(
                            sig_x, [y_sig] * len(sig_x), c=lcolor,
                            s=10, marker=lstyle)
        if k==0 or k==2:
            h,_ = ax.get_legend_handles_labels()
            ax.legend(
                h,['LTM','STM'],
                loc='upper left',ncol=1,frameon=False,fontsize=20).set_title(None)
    else:
        ax = ax.ravel()
        n = 0
        for sizeN in sizeList:
            if n==0 or n==4 or n==8 or n==12:
                leg_tag = True
            else:
                leg_tag = False

            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            sns.lineplot(
                data=dat[(dat['corr']=='mean')&(dat['setsize']==sizeN)],
                x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
                style_order=['exp1b','exp2'],markers=True,dashes=False,
                palette=clist,linewidth=3,markersize=10,err_style='bars',
                errorbar=('se',0),legend=leg_tag,ax=ax[n])
            ax[n].set_ylabel(ylabel=None)
            ax[n].set(ylim=(-0.1,0.31))
            ax[n].set_yticks(np.arange(-0.1,0.31,0.1))
            ax[n].set_title('MSS%d' % (sizeN), fontsize=16)
            ax[n].set_ylabel(ylabel='Beta')
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')

            y_gap = 0.01
            y_sig = -0.06 - y_gap
            y_fsig = 0.265 + y_gap
            for exp_tag in exp_tags:
                y_sig += y_gap
                dat_cond = dat[(dat['exp'] == exp_tag) &
                               (dat['setsize'] == sizeN) &
                               (dat['corr'] == 'mean')]
                X = np.array(
                    [dat_cond.loc[(dat_cond['layer'] == x_name),
                    'coeff'].values for x_name in activation_names])
                X = np.transpose(X, (1, 0))

                t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                    X,n_jobs=None,threshold=t_thresh,adjacency=None,tail=1,
                    n_permutations=n_permutations,out_type='indices')
                # print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
                print(clusters)
                print(p_values)

                if (len(clusters) != 0):
                    for pN in range(len(p_values)):
                        if (p_values[pN] < p_crit):
                            sig_x = ['conv_%d' % (layerN + 1) \
                                         if (layerN + 1) < 6 else 'fc_%d' % (layerN + 1) \
                                     for layerN in list(clusters[pN][0])]
                            if exp_tag == 'exp1b':
                                lcolor = clist[0]
                                lstyle = 'o'
                            else:
                                lcolor = clist[1]
                                lstyle = 'x'

                            ax[n].scatter(
                                sig_x, [y_sig] * len(sig_x), c=lcolor,
                                s=10, marker=lstyle)
            if n==0 or n==4 or n==8 or n==12:
                h,_ = ax[n].get_legend_handles_labels()
                ax[n].legend(
                    h,['LTM','STM'],
                    loc='upper left',ncol=1,
                    frameon=False,fontsize=20).set_title(None)
            n += 1

        # y_major_locator = MultipleLocator(0.15)
        # ax[n].yaxis.set_major_locator(y_major_locator)
    fig.text(
        x_pos,y_pos,eff_label,ha='center',va='top',
        color='k',fontdict={'weight':'bold'})
    fig.text(
        xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    if k==1 or k==3:
        plt.subplots_adjust(left=0.08,right=1,top=0.85,bottom=0.15)
    # # plt.margins(0,0)
    plt.savefig(
        os.path.join(figPath,'glm_w2veff_mean%d.tif'%(k)))
    plt.show(block=True)
    plt.close('all')


mpl.rcParams.update({'font.size': 21})
for k,dat in enumerate(
        [c_rt_2cate,c_resid_cate,s_resid_cate,vs_resid_cate]):

    if k==0:
        eff_label = 'Categorical Similarity'
        figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.55,1
        fig_w,fig_h,col_n = 9,6,1
    elif k==1:
        eff_label = 'Categorical Similarity (w/o visual similarity)'
        figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.55,1
        fig_w,fig_h,col_n = 18,6,4
    elif k==2:
        eff_label = 'Categorical Similarity (w/o semantic similarity)'
        figN,xlab_pos,x_pos,y_pos = '(B)',0.05,0.55,1
        fig_w,fig_h,col_n = 9,6,1
    else:
        eff_label = 'Categorical Similarity (w/o semantic and visual similarity)'
        figN,xlab_pos,x_pos,y_pos = '(D)',0.05,0.55,1
        fig_w,fig_h,col_n = 18,6,4

    fig,ax = plt.subplots(
        1,col_n,sharex=True,sharey=True,figsize=(fig_w,fig_h))
    n = 0
    out_label_list = []

    if k == 0 or k == 2:
        ax.axhline(0,color='black',lw=1,linestyle='dashed')
        sns.lineplot(
            data=dat[(dat['corr'] == 'mean')],
            x='setsize', y='coeff', hue='exp', style='exp', hue_order=exp_tags,
            style_order=['exp1b', 'exp2'], markers=True, dashes=False,
            palette=clist, linewidth=3, markersize=10, err_style='bars',
            errorbar=('se', 0), legend=True, ax=ax)
        ax.set_ylabel(ylabel='Beta')
        ax.set(ylim=(-0.1,0.35))
        ax.set_yticks(np.arange(-0.1,0.31,0.1))
        ax.set_xticks([1,2,4,8],labels=[1,2,4,8])
        ax.set_xlabel(xlabel='MSS')
        h,_ = ax.get_legend_handles_labels()
        ax.legend(
            h,['LTM','STM'],
            loc='upper left',ncol=1,frameon=False).set_title(None)
        y_gap = 0.01
        y_sig = -0.06 - y_gap
        y_fsig = 0.265 + y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_cond = dat[(dat['exp'] == exp_tag) &
                           (dat['corr'] == 'mean')]
            X = np.array(
                [dat_cond.loc[(dat_cond['setsize'] == x_name),
                'coeff'].values for x_name in sizeList])
            X = np.transpose(X, (1, 0))

            t_clust,clusters,p_values,H0 = permutation_cluster_1samp_test(
                X,n_jobs=None,threshold=t_thresh,adjacency=None,tail=1,
                n_permutations=n_permutations,out_type='indices')
            print(clusters)
            print(p_values)

            if (len(clusters) != 0):
                for pN in range(len(p_values)):
                    if (p_values[pN] < p_crit):
                        sig_x = [sizeList[sizeN] for sizeN in list(clusters[pN][0])]
                        if exp_tag == 'exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'

                        ax.scatter(
                            sig_x, [y_sig] * len(sig_x), c=lcolor,
                            s=10, marker=lstyle)
    else:
        ax = ax.ravel()
        n = 0
        for sizeN in sizeList:
            if n==0 or n==4 or n==8 or n==12:
                leg_tag = True
            else:
                leg_tag = False
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            sns.lineplot(
                data=dat[(dat['corr']=='mean')&(dat['setsize']==sizeN)],
                x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
                style_order=['exp1b','exp2'],markers=True,dashes=False,
                palette=clist,linewidth=3,markersize=10,err_style='bars',
                errorbar=('se',0),legend=leg_tag,ax=ax[n])
            ax[n].set_ylabel(ylabel=None)
            ax[n].set(ylim=(-0.1,0.35))
            ax[n].set_yticks(np.arange(-0.1,0.31,0.1))
            ax[n].set_title('MSS%d' % (sizeN), fontsize=20)
            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')

            y_gap = 0.01
            y_sig = -0.06 - y_gap
            y_fsig = 0.265 + y_gap
            for exp_tag in exp_tags:
                y_sig += y_gap
                dat_cond = dat[(dat['exp'] == exp_tag) &
                               (dat['setsize'] == sizeN) &
                               (dat['corr'] == 'mean')]
                X = np.array(
                    [dat_cond.loc[(dat_cond['layer'] == x_name),
                    'coeff'].values for x_name in activation_names])
                X = np.transpose(X, (1, 0))

                t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
                    X,n_jobs=None,threshold=t_thresh,adjacency=None,tail=1,
                    n_permutations=n_permutations,out_type='indices')
                # print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
                print(clusters)
                print(p_values)

                if (len(clusters) != 0):
                    for pN in range(len(p_values)):
                        if (p_values[pN] < p_crit):
                            sig_x = ['conv_%d' % (layerN + 1) \
                                         if (layerN + 1) < 6 else 'fc_%d' % (layerN + 1) \
                                     for layerN in list(clusters[pN][0])]
                            if exp_tag == 'exp1b':
                                lcolor = clist[0]
                                lstyle = 'o'
                            else:
                                lcolor = clist[1]
                                lstyle = 'x'

                            ax[n].scatter(
                                sig_x, [y_sig] * len(sig_x), c=lcolor,
                                s=10, marker=lstyle)
            if n==0 or n==4 or n==8 or n==12:
                h,_ = ax[n].get_legend_handles_labels()
                ax[n].legend(
                    h,['LTM','STM'],
                    loc='upper left',ncol=1,frameon=False,fontsize=20).set_title(None)
            n += 1

        # y_major_locator = MultipleLocator(0.15)
        # ax[n].yaxis.set_major_locator(y_major_locator)

    fig.text(
        x_pos,y_pos,eff_label,ha='center',va='top',
        color='k',fontdict={'weight':'bold'})
    fig.text(
        xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=19)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    if k==1 or k==3:
        plt.subplots_adjust(left=0.08,right=1,top=0.86,bottom=0.15)
    # # plt.margins(0,0)
    plt.savefig(
        os.path.join(figPath,'glm_cateeff_mean%d.tif'%(k)))
    plt.show(block=True)
    plt.close('all')


# mpl.rcParams.update({'font.size':16})
# for corr_tag in ['mean','max']:
#
#     fig,ax = plt.subplots(
#         2,4,sharex=True,sharey=True,figsize=(19,14))
#     ax = ax.ravel()
#     n = 0
#     for k,dat in enumerate(
#             [v_rt_simi_3layers,v_avg2cate_3layers,
#              s_resid_simi_3layers,s_resid_simi_v_3layers]):
#         if k==0:
#             eff_label = 'Visual Similarity'
#             figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.3,0.98
#         elif k==1:
#             eff_label = 'Visual Similarity (w/o categorical similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(B)',0.55,0.77,0.98
#         elif k==2:
#             eff_label = 'Visual Similarity (w/o semantic similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.3,0.51
#         else:
#             eff_label = 'Visual Similarity (w/o categorical and semantic similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(D)',0.55,0.76,0.51
#
#
#         for exp_tag in exp_tags:
#             if exp_tag=='exp1b':
#                 exp_title = 'LTM'
#             else:
#                 exp_title = 'STM'
#
#             if n==0:
#                 leg_tag = True
#             else:
#                 leg_tag = False
#
#             sns.lineplot(
#                 data=dat[(dat['corr']==corr_tag)&(dat['exp']==exp_tag)],
#                 x='setsize',y='coeff',hue='layer',
#                 hue_order=['early','late','fc_8'],style='layer',
#                 markers=True,dashes=False,palette=Diona[0:3],
#                 linewidth=2,markersize=12,err_style='bars',
#                 errorbar=('se',1),err_kws={'capsize':10},
#                 legend=leg_tag,ax=ax[n])
#             ax[n].set_xticks(sizeList,labels=sizeList)
#             ax[n].set_xlabel('Memory Set Size')
#             ax[n].set_ylabel('Beta')
#             y_major_locator = MultipleLocator(0.1)
#             ax[n].yaxis.set_major_locator(y_major_locator)
#             ax[n].set_title(exp_title)
#
#             n += 1
#
#         fig.text(
#             x_pos,y_pos,eff_label,ha='center',va='top',color='k',fontsize=18)
#         fig.text(
#             xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)
#
#     h,_ = ax[0].get_legend_handles_labels()
#     ax[0].legend(
#         h,['Early','Late','FC 8'],
#         loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)
#
#     sns.despine(offset=10,trim=True)
#     plt.subplots_adjust(left=0.05,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
#     plt.savefig(
#         os.path.join(figPath,'aov_viseff_%s.tif'%(corr_tag)))
#     plt.show(block=True)
#     plt.close('all')

# mpl.rcParams.update({'font.size':20})
# for corr_tag in ['mean','max']:
#
#     fig,ax = plt.subplots(
#         2,4,sharex=True,sharey=True,figsize=(19,14))
#     ax = ax.ravel()
#     n = 0
#     for k,dat in enumerate(
#             [v_rt_simi_3layers,v_avg2cate_3layers,
#              s_resid_simi_3layers,s_resid_simi_v_3layers]):
#         if k==0:
#             eff_label = 'Visual Similarity'
#             figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.3,0.98
#         elif k==1:
#             eff_label = 'Visual Similarity (w/o categorical similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(B)',0.55,0.77,0.98
#         elif k==2:
#             eff_label = 'Visual Similarity (w/o semantic similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.3,0.51
#         else:
#             eff_label = 'Visual Similarity (w/o categorical and semantic similarity)'
#             figN,xlab_pos,x_pos,y_pos = '(D)',0.55,0.76,0.51
#
#
#         for exp_tag in exp_tags:
#             if exp_tag=='exp1b':
#                 exp_title = 'LTM'
#             else:
#                 exp_title = 'STM'
#
#             if n==0:
#                 leg_tag = True
#             else:
#                 leg_tag = False
#
#             sns.barplot(
#                 data=dat[(dat['corr']==corr_tag)&(dat['exp']==exp_tag)],
#                 x='setsize',y='coeff',hue='layer',
#                 hue_order=['early','late','fc_8'],palette=Diona[0:3],
#                 errorbar='se',capsize=0.15,errcolor='grey',legend=leg_tag,
#                 ax=ax[n])
#             ax[n].set_xlabel('Memory Set Size')
#             ax[n].set_ylabel('Beta')
#             y_major_locator = MultipleLocator(0.1)
#             ax[n].yaxis.set_major_locator(y_major_locator)
#             ax[n].set_title(exp_title)
#
#             n += 1
#
#         fig.text(
#             x_pos,y_pos,eff_label,ha='center',va='top',color='k',fontsize=18)
#         fig.text(
#             xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)
#
#     h,_ = ax[0].get_legend_handles_labels()
#     ax[0].legend(
#         h,['Early','Late','FC 8'],
#         loc='upper left',ncol=1,fontsize=18,frameon=False).set_title(None)
#
#     sns.despine(offset=10,trim=True)
#     plt.subplots_adjust(left=0.05,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
#     plt.savefig(
#         os.path.join(figPath,'aov_bar_viseff_%s.tif'%(corr_tag)))
#     plt.show(block=True)
#     plt.close('all')


# categorical similarity
mpl.rcParams.update({'font.size':20})
for corr_tag in ['mean','max']:
    print(corr_tag)

    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(21,10))
    ax = ax.ravel()
    n = 0
    out_label_list = []

    for k,dat in enumerate(
            [c_resid_cate,vs_resid_cate]):
        # if k==0:
        #     eff_label = 'semantic Similarity'
        #     figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.25,0.98
        # elif k==1:
        #     eff_label = 'semantic Similarity (w/o visual similarity)'
        #     figN,xlab_pos,x_pos,y_pos = '(B)',0.55,0.75,0.98
        # elif k==2:
        #     eff_label = 'semantic Similarity (w/o categorical similarity)'
        #     figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.25,0.51
        # else:
        #     eff_label = 'semantic Similarity (w/o categorical and visual similarity)'
        #     figN,xlab_pos,x_pos,y_pos = '(D)',0.55,0.75,0.51
        if k==0:
            eff_label = 'Categorical Similarity (w/o visual similarity)'
            figN,xlab_pos,x_pos,y_pos = '(A)',0.25,0.5,0.98
        elif k==1:
            eff_label = 'Categorical Similarity (w/o visual and sematic similarity)'
            figN,xlab_pos,x_pos,y_pos = '(C)',0.25,0.5,0.51

        for sizeN in sizeList:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(
                data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
                x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
                style_order=['exp1b','exp2'],markers=True,dashes=False,
                palette=clist,linewidth=2,markersize=10,err_style='bars',
                errorbar=('se',0),legend=leg_tag,ax=ax[n])

            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Beta')
            # y_major_locator = MultipleLocator(0.15)
            # ax[n].yaxis.set_major_locator(y_major_locator)
            ax[n].set_title('MSS%d'%(sizeN),fontsize=18)

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
                print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
                print(clusters)
                print(p_values)

                if (len(clusters)!=0):
                    for pN in range(len(p_values)):
                        if (p_values[pN]<p_crit):
                            sig_x = ['conv_%d'%(layerN+1) \
                                         if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        fig.text(
            x_pos,y_pos,eff_label,ha='center',va='top',color='k',fontsize=18)
        fig.text(
            xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],
        loc='upper left',ncol=1,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1,right=1,top=0.9,bottom=0.1,hspace=0.3,wspace=0.1)
    # plt.margins(0,0)
    plt.savefig(
        os.path.join(figPath,'glm_cateeff_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')

# semantic similarity
mpl.rcParams.update({'font.size':16})
for corr_tag in ['mean','max']:
    print(corr_tag)

    fig,ax = plt.subplots(
        2,4,sharex=True,sharey=True,figsize=(21,10))
    ax = ax.ravel()
    n = 0
    out_label_list = []

    for k,dat in enumerate(
            [s_resid_w2v,s_resid_2cate_s]):
        # if k==0:
        #     eff_label = 'semantic Similarity'
        #     figN,xlab_pos,x_pos,y_pos = '(A)',0.05,0.25,0.98
        # elif k==1:
        #     eff_label = 'semantic Similarity (w/o visual similarity)'
        #     figN,xlab_pos,x_pos,y_pos = '(B)',0.55,0.75,0.98
        # elif k==2:
        #     eff_label = 'semantic Similarity (w/o categorical similarity)'
        #     figN,xlab_pos,x_pos,y_pos = '(C)',0.05,0.25,0.51
        # else:
        #     eff_label = 'semantic Similarity (w/o categorical and visual similarity)'
        #     figN,xlab_pos,x_pos,y_pos = '(D)',0.55,0.75,0.51
        if k==0:
            eff_label = 'semantic Similarity (w/o visual similarity)'
            figN,xlab_pos,x_pos,y_pos = '(A)',0.25,0.5,0.98
        elif k==1:
            eff_label = 'semantic Similarity (w/o categorical and visual similarity)'
            figN,xlab_pos,x_pos,y_pos = '(C)',0.25,0.5,0.51

        for sizeN in sizeList:
            ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

            if n==0:
                leg_tag = True
            else:
                leg_tag = False
            sns.lineplot(
                data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
                x='layer',y='coeff',hue='exp',style='exp',hue_order=exp_tags,
                style_order=['exp1b','exp2'],markers=True,dashes=False,
                palette=clist,linewidth=2,markersize=10,err_style='bars',
                errorbar=('se',0),legend=leg_tag,ax=ax[n])

            ax[n].set_xticks(activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Beta')
            # y_major_locator = MultipleLocator(0.15)
            # ax[n].yaxis.set_major_locator(y_major_locator)
            ax[n].set_title('MSS%d'%(sizeN),fontsize=14)

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
                print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
                print(clusters)
                print(p_values)

                if (len(clusters)!=0):
                    for pN in range(len(p_values)):
                        if (p_values[pN]<p_crit):
                            sig_x = ['conv_%d'%(layerN+1) \
                                         if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        fig.text(
            x_pos,y_pos,eff_label,ha='center',va='top',color='k',fontsize=18)
        fig.text(
            xlab_pos,y_pos,figN,ha='center',va='top',color='k',fontsize=18)
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],
        loc='upper left',ncol=1,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1,right=1,top=0.9,
                        bottom=0.1,hspace=0.3,wspace=0.1)
    # plt.margins(0,0)
    plt.savefig(
        os.path.join(figPath,'glm_w2veff_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')


# s_tag = 'v'
# dat_list = [v,v_c,v_s,v_cs]
# s_tag = 's'
# dat_list = [s,s_v,s_c,s_cv]
# s_tag = 'c'
# dat_list = [c,c_v,c_s,c_vs]
dat_list = [v_cs,s_cv,c_vs]
mpl.rcParams.update({'font.size':20})
for corr_tag in ['mean']:

    fig,ax = plt.subplots(
        1,3,sharex=True,sharey=True,figsize=(16,6))
    ax = ax.ravel()

    for n,dat in enumerate(dat_list):
        if n == 0:
            leg_tag = True
        else:
            leg_tag = False

        # sns.lineplot(
        #     data=dat[(dat['corr']==corr_tag)],
        #     x='setsize',y='coeff',hue='exp',
        #     hue_order=['exp1b','exp2'],style='exp',
        #     markers=True,dashes=False,palette=clist,
        #     linewidth=2,markersize=12,err_style='bars',
        #     errorbar=('se',1),err_kws={'capsize':10},
        #     legend=leg_tag,ax=ax[n])
        sns.barplot(
            data=dat[(dat['corr']==corr_tag)],
            x='setsize',y='coeff',hue='exp',
            hue_order=['exp1b','exp2'],palette=clist,
            errorbar='se',capsize=0.15,errcolor='grey',
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(['1','2','4','8'],labels=sizeList)
        ax[n].set_xlabel('Memory Set Size')
        ax[n].set_ylabel('Beta')
        ax[n].set_ylim(-0.1,0.25)
        y_major_locator = MultipleLocator(0.1)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(
            dat.loc[(dat['subj']==1)&
                    (dat['corr']==corr_tag)&
                    (dat['setsize']==1),'eff'].values[0])
    h, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h, ['LTM','STM'],loc='lower right',ncol=1,labelcolor=None,
        fontsize=18,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'bplt_layer6_3_simi_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')


v_cs_6 = v_cs.groupby(
    ['corr','exp','subj','eff'])['coeff'].agg(np.mean).reset_index()
s_cv_6 = s_cv.groupby(
    ['corr','exp','subj','eff'])['coeff'].agg(np.mean).reset_index()
c_vs_6 = c_vs.groupby(
    ['corr','exp','subj','eff'])['coeff'].agg(np.mean).reset_index()
dat = pd.concat([v_cs_6,s_cv_6,c_vs_6],axis=0,ignore_index=True)
mpl.rcParams.update({'font.size':24})
for corr_tag in ['mean']:

    fig,ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(12,9))
    leg_tag = True

    sns.barplot(
        data=dat[(dat['corr']==corr_tag)],
        x='eff',y='coeff',hue='exp',
        hue_order=['exp1b','exp2'],palette=clist,
        errorbar='se',capsize=0.15,errcolor='grey',
        legend=leg_tag,ax=ax)
    ax.set_xlabel('Similarity',fontdict={'weight':'bold'})
    ax.set_ylabel('Beta',fontdict={'weight':'bold'})
    # ax.set_ylim(-0.05,0.21)
    y_major_locator = MultipleLocator(0.06)
    ax.yaxis.set_major_locator(y_major_locator)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(
        h, ['LTM','STM'],loc='upper right',ncol=1,labelcolor=None,
        fontsize=22,frameon=False).set_title(None)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'bplt_6_simi_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')