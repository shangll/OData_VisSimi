#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2025.2.14
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


res_output = set_filepath(rootPath,'res_all')
save_tag = 1
p_crit = 0.05
tail = 0
t_thresh = None
n_permutations = 1000
seedN = 95
Diona = ['#393A53','#DD9E5B','#D58784','#B69480','#EDCAC6']
Kirara = ["#355D73","#8DC0C8","#D5C7AC","#EAC772","#69A94E"]
activation_names = [
    'conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']
cateList = ['within','between']
corr_tags = ['mean','max']
exp_tags = ['exp1b','exp2']
clist = ['#FFBA00','grey']
lw = 5
mkr_s = 20

# simi_w2v = pd.read_csv(
#     os.path.join(res_output,'expAll_simi_w2v.csv'),sep=',')
# simi_img = pd.read_csv(
#     os.path.join(res_output,'expAll_simi_img.csv'),sep=',')
# simi_all = simi_img.copy(deep=True)
# simi_all[['w2v_mean','w2v_max']] = 0,0
# for exp_tag in exp_tags:
#     for name in activation_names:
#         simi_all.loc[
#             (simi_all['exp']==exp_tag)&
#             (simi_all['layer']==name),
#             ['w2v_mean','w2v_max']] = simi_w2v.loc[
#             (simi_w2v['exp']==exp_tag),
#             ['w2v_mean','w2v_max']].values
# simi_max = simi_all.copy(deep=True)
# simi_all.drop(['simi_max','w2v_max'],axis=1,inplace=True)
# simi_all['corr'] = 'mean'
# simi_all.rename(
#     columns={'simi_mean':'v',
#              'w2v_mean':'s'},inplace=True)
# simi_max.drop(['simi_mean','w2v_mean'],axis=1,inplace=True)
# simi_max['corr'] = 'max'
# simi_max.rename(
#     columns={'simi_max':'v',
#              'w2v_max':'s'},inplace=True)
# simi_all = pd.concat(
#     [simi_all,simi_max],axis=0,ignore_index=True)
# simi_all['c'] = 1
# simi_all.loc[simi_all['cond']=='between','c'] = -1
# simi_all.to_csv(
#     os.path.join(res_output,'simi_all.csv'),sep=',',
#     mode='w',header=True,index=False)
simi_all = pd.read_csv(
    os.path.join(res_output,'simi_all.csv'),sep=',')

# GLM (rt ~ similarity)
var_tags = ['v','s','c']
glm_simi,glm_coeff,glm_subj,glm_size,\
    glm_corr,glm_exp,glm_layer,glm_r2,\
    glm_aic,glm_dev = \
    [],[],[],[],[],[],[],[],[],[]
for corr_tag in corr_tags:
    # each experiment
    for exp_tag in exp_tags:
        # each similarity
        for var in var_tags:
            # each layer
            for name in activation_names:
                # each MSS
                for sizeN in sizeList:
                    exp = simi_all[
                        (simi_all['corr']==corr_tag)&
                        (simi_all['exp']==exp_tag)&
                        (simi_all['layer']==name)&
                        (simi_all['setsize']==sizeN)]
                    exp_subj = list(set(exp['subj']))

                    # each subject
                    for subj in exp_subj:
                        exp_simi_indv = exp[
                            (exp['subj']==subj)].copy()

                        # normalization (Z-score)
                        exp_simi_indv['rt_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'rt'])
                        exp_simi_indv['simi_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,var])

                        # GLM fit
                        y = exp_simi_indv['rt_Z']
                        X = exp_simi_indv['simi_Z']
                        X = sm.add_constant(X)
                        model = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        glm_simi.append(var)
                        glm_coeff.append(model.params[1])
                        glm_subj.append(subj)
                        glm_size.append(sizeN)
                        glm_corr.append(corr_tag)
                        glm_exp.append(exp_tag)
                        glm_layer.append(name)
                        glm_r2.append(
                            model.pseudo_rsquared(kind='cs'))
                        glm_dev.append(
                            model.deviance)
                        glm_aic.append(
                            model.aic)
glm_data = pd.DataFrame(
    {'exp':glm_exp,'subj':glm_subj,'setsize':glm_size,
     'simi':glm_simi,'coeff':glm_coeff,'r2':glm_r2,
     'dev':glm_dev,'aic':glm_aic,'layer':glm_layer,
     'corr':glm_corr})
#
# 1st step
# GLM (rt ~ similarity)
glm_simi,glm_coeff,glm_subj,glm_size,\
    glm_corr,glm_exp,glm_r2,glm_layer,\
    glm_aic,glm_dev = \
    [],[],[],[],[],[],[],[],[],[]
for corr_tag in corr_tags:
    # each experiment
    for exp_tag in exp_tags:
        # each similarity
        for var in var_tags:
            for semi in var_tags:
                if var!=semi:
                    # each layer
                    for name in activation_names:
                        # each MSS
                        for sizeN in sizeList:
                            exp = simi_all[
                                (simi_all['corr']==corr_tag)&
                                (simi_all['exp']==exp_tag)&
                                (simi_all['layer']==name)&
                                (simi_all['setsize']==sizeN)]
                            exp_subj = list(set(exp['subj']))

                            # each subject
                            for subj in exp_subj:
                                exp_simi_indv = exp[
                                    (exp['subj']==subj)].copy()

                                # normalization (Z-score)
                                exp_simi_indv['rt_Z'] = preprocessing.scale(
                                    exp_simi_indv.loc[:,'rt'])
                                exp_simi_indv['simi_Z'] = preprocessing.scale(
                                    exp_simi_indv.loc[:,var])
                                exp_simi_indv['semi_Z'] = preprocessing.scale(
                                    exp_simi_indv.loc[:,semi])

                                # GLM fit
                                # 1st step
                                y = exp_simi_indv['rt_Z']
                                X = exp_simi_indv['semi_Z']
                                X = sm.add_constant(X)
                                model = sm.GLM(
                                    y,X,family=sm.families.Gaussian()).fit()

                                # 2nd step
                                y = np.array(list(model.resid_response))
                                X = exp_simi_indv['simi_Z']
                                X = sm.add_constant(X)
                                model2 = sm.GLM(
                                    y,X,family=sm.families.Gaussian()).fit()

                                glm_simi.append('%s w/o %s'%(var,semi))
                                glm_coeff.append(model2.params[1])
                                glm_subj.append(subj)
                                glm_size.append(sizeN)
                                glm_corr.append(corr_tag)
                                glm_exp.append(exp_tag)
                                glm_layer.append(name)
                                glm_r2.append(
                                    model2.pseudo_rsquared(kind='cs'))
                                glm_dev.append(
                                    model2.deviance)
                                glm_aic.append(
                                    model2.aic)
semi_1var_data = pd.DataFrame(
    {'exp':glm_exp,'subj':glm_subj,'setsize':glm_size,
     'simi':glm_simi,'coeff':glm_coeff,'r2':glm_r2,
     'dev':glm_dev,'aic':glm_aic,'layer':glm_layer,
     'corr':glm_corr})

#
# 2 steps
# GLM (rt ~ similarity)
glm_simi,glm_coeff,glm_subj,glm_size,\
    glm_corr,glm_exp,glm_r2,glm_layer,\
    glm_aic,glm_dev,glm_cond = \
    [],[],[],[],[],[],[],[],[],[],[]
for corr_tag in corr_tags:
    # each experiment
    for exp_tag in exp_tags:
        # each similarity
        for var in var_tags:
            if var=='v':
                semi = 's'
                semi_simi = 'v w/o c&s'
            elif var=='s':
                semi = 'v'
                semi_simi = 's w/o v&c'
            # each layer
            for name in activation_names:
                # each MSS
                for sizeN in sizeList:
                    for cond in cateList:
                        exp = simi_all[
                            (simi_all['corr']==corr_tag)&
                            (simi_all['exp']==exp_tag)&
                            (simi_all['layer']==name)&
                            (simi_all['setsize']==sizeN)&
                            (simi_all['cond']==cond)]
                        exp_subj = list(set(exp['subj']))

                        # each subject
                        for subj in exp_subj:
                            exp_simi_indv = exp[
                                (exp['subj']==subj)].copy()

                            # normalization (Z-score)
                            exp_simi_indv['rt_Z'] = preprocessing.scale(
                                exp_simi_indv.loc[:,'rt'])
                            exp_simi_indv['simi_Z'] = preprocessing.scale(
                                exp_simi_indv.loc[:,var])
                            exp_simi_indv['semi_Z'] = preprocessing.scale(
                                exp_simi_indv.loc[:,semi])

                            # GLM fit
                            # 1st step
                            y = exp_simi_indv['rt_Z']
                            X = exp_simi_indv['semi_Z']
                            X = sm.add_constant(X)
                            model = sm.GLM(
                                y,X,family=sm.families.Gaussian()).fit()

                            # 2nd step
                            y = np.array(list(model.resid_response))
                            X = exp_simi_indv['simi_Z']
                            X = sm.add_constant(X)
                            model2 = sm.GLM(
                                y,X,family=sm.families.Gaussian()).fit()

                            glm_simi.append(semi_simi)
                            glm_cond.append(cond)
                            glm_coeff.append(model2.params[1])
                            glm_subj.append(subj)
                            glm_size.append(sizeN)
                            glm_corr.append(corr_tag)
                            glm_exp.append(exp_tag)
                            glm_layer.append(name)
                            glm_r2.append(
                                model2.pseudo_rsquared(kind='cs'))
                            glm_dev.append(
                                model2.deviance)
                            glm_aic.append(
                                model2.aic)
step_data_cond = pd.DataFrame(
    {'exp':glm_exp,'subj':glm_subj,'setsize':glm_size,
     'simi':glm_simi,'coeff':glm_coeff,'r2':glm_r2,
     'dev':glm_dev,'aic':glm_aic,'layer':glm_layer,
     'corr':glm_corr,'cond':glm_cond})
step_data = step_data_cond.groupby(
    ['exp','subj','setsize','simi','layer','corr'])[
    ['coeff','r2','dev','aic']].agg('mean').reset_index()

#
# GLM (rt ~ similarity - semi-partial)
glm_simi,glm_coeff,glm_subj,glm_size,\
    glm_corr,glm_exp,glm_r2,glm_layer,\
    glm_aic,glm_dev = \
    [],[],[],[],[],[],[],[],[],[]
for corr_tag in corr_tags:
    # each experiment
    for exp_tag in exp_tags:
        # each similarity
        # for var in var_tags:
        for var in ['c']:
            if var=='v':
                semi1 = 'c'
                semi2 = 's'
            elif var=='s':
                semi1 = 'v'
                semi2 = 'c'
            else:
                semi1 = 'v'
                semi2 = 's'
            # each layer
            for name in activation_names:
                # each MSS
                for sizeN in sizeList:
                    exp = simi_all[
                        (simi_all['corr']==corr_tag)&
                        (simi_all['exp']==exp_tag)&
                        (simi_all['layer']==name)&
                        (simi_all['setsize']==sizeN)]
                    exp_subj = list(set(exp['subj']))

                    # each subject
                    for subj in exp_subj:
                        exp_simi_indv = exp[
                            (exp['subj']==subj)].copy()

                        # normalization (Z-score)
                        exp_simi_indv['rt_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'rt'])
                        exp_simi_indv['simi_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,var])
                        exp_simi_indv['semi1_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,semi1])
                        exp_simi_indv['semi2_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,semi2])

                        # GLM fit
                        # 1st step
                        y = exp_simi_indv['rt_Z']
                        X = exp_simi_indv['semi1_Z']
                        X = sm.add_constant(X)
                        model = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        # 2nd step
                        y = np.array(list(model.resid_response))
                        X = exp_simi_indv['semi2_Z']
                        X = sm.add_constant(X)
                        model2 = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        # 3rd step
                        y = np.array(list(model2.resid_response))
                        X = exp_simi_indv['simi_Z']
                        X = sm.add_constant(X)
                        model3 = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        glm_simi.append(
                            '%s w/o %s&%s'%(var,semi1,semi2))
                        glm_coeff.append(model3.params[1])
                        glm_subj.append(subj)
                        glm_size.append(sizeN)
                        glm_corr.append(corr_tag)
                        glm_exp.append(exp_tag)
                        glm_layer.append(name)
                        glm_r2.append(
                            model3.pseudo_rsquared(kind='cs'))
                        glm_dev.append(
                            model3.deviance)
                        glm_aic.append(
                            model3.aic)
semi_data = pd.DataFrame(
    {'exp':glm_exp,'subj':glm_subj,'setsize':glm_size,
     'simi':glm_simi,'coeff':glm_coeff,'r2':glm_r2,
     'dev':glm_dev,'aic':glm_aic,'layer':glm_layer,
     'corr':glm_corr})

glm_data = pd.concat(
    [glm_data,semi_1var_data,step_data,semi_data],
    axis=0,ignore_index=True)
# glm_data = pd.concat(
#     [glm_data,semi_1var_data,semi_data],
#     axis=0,ignore_index=True)
if save_tag==1:
    glm_data.to_csv(
        os.path.join(res_output,'glm_data.csv'),
        sep=',',mode='w',header=True,index=False)

#
glm_data = pd.read_csv(
    os.path.join(res_output,'glm_data.csv'),sep=',')
#
# Plot
#
# rt ~ similarity
simi_list = ['v','v w/o c','s w/o v','c w/o v',
             'v w/o s',
             'v w/o c&s','s w/o v&c','c w/o v&s']
mpl.rcParams.update({'font.size':27})
for corr_tag in corr_tags:
    for simi_tag in simi_list:
        fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(21,9))
        ax = ax.ravel()
        n = 0
        for sizeN in sizeList:
            ax[n].axhline(
                0,color='black',lw=1,linestyle='dashed')
            if n==0:
                leg_tag = True
            else:
                leg_tag = False

            sns.lineplot(
                data=glm_data[
                    (glm_data['corr']==corr_tag)&
                    (glm_data['simi']==simi_tag)&
                    (glm_data['setsize']==sizeN)],
                x='layer',y='coeff',hue='exp',
                style='exp',hue_order=exp_tags,
                style_order=['exp1b','exp2'],
                markers=True,dashes=False,
                palette=clist,linewidth=lw,markersize=mkr_s,
                err_style='bars',errorbar=('se',0),
                legend=leg_tag,ax=ax[n])

            ax[n].set_xticks(
                activation_names,labels=range(1,9))
            ax[n].set_xlabel(xlabel='Layer')
            ax[n].set_ylabel(ylabel='Beta')
            ax[n].set_ylim(-0.12,0.32)
            y_major_locator = MultipleLocator(0.1)
            ax[n].yaxis.set_major_locator(y_major_locator)
            ax[n].set_title(
                'MSS%d'%(sizeN),fontsize=25,fontweight='bold')

            y_gap = 0.01
            y_sig = -0.1-y_gap
            y_fsig = 0.265+y_gap
            for exp_tag in exp_tags:
                y_sig += y_gap
                dat_exp = glm_data[
                    (glm_data['exp']==exp_tag)&
                    (glm_data['simi']==simi_tag)&
                    (glm_data['setsize']==sizeN)&
                    (glm_data['corr']==corr_tag)]
                X = np.array(
                    [dat_exp.loc[(dat_exp['layer']==x_name),
                    'coeff'].values for x_name in activation_names])
                X = np.transpose(X,(1,0))

                t_clust,clusters,p_values,H0 = \
                    permutation_cluster_1samp_test(
                        X,n_jobs=None,threshold=t_thresh,
                        adjacency=None,seed=seedN,
                        n_permutations=n_permutations,
                        out_type='indices')
                print('%s MSS %d %s'%(corr_tag,sizeN,exp_tag))
                print(clusters)
                print(p_values)

                if (len(clusters)!=0):
                    for pN in range(len(p_values)):
                        if (p_values[pN]<p_crit):
                            sig_x = ['conv_%d'%(layerN+1) \
                                         if (layerN+1)<6 \
                                         else 'fc_%d'%(layerN+1)\
                                     for layerN in \
                                     list(clusters[pN][0])]
                            if exp_tag=='exp1b':
                                lcolor = clist[0]
                                lstyle = 'o'
                            else:
                                lcolor = clist[1]
                                lstyle = 'x'
                            ax[n].scatter(
                                sig_x,[y_sig]*len(sig_x),
                                c=lcolor,s=mkr_s*2,marker=lstyle)
            n += 1
        h,_ = ax[0].get_legend_handles_labels()
        ax[0].legend(
            h,['LTM','STM'],
            loc='upper left',ncol=1,frameon=False).set_title(None)
        sns.despine(offset=10,trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1,right=1,top=0.89,
                            bottom=0.15,hspace=0.3,wspace=0.1)
        # plt.margins(0,0)
        x_pos,y_pos = 0.04, 0.97
        if simi_tag=='v':
            semi = ''
            # plt.suptitle(
            #     'Visual',
            #     fontweight='bold')
            # figN = '(A)'
            # plt.suptitle(
            #     simi_tag.upper(),
            #     fontsize=26,fontweight='bold')
        elif simi_tag in ['v w/o c','s w/o v','c w/o v']:
            semi = '_semi1'
            figN = '(B)'
            plt.suptitle(
                simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-1].upper(),
                fontweight='bold')
        elif simi_tag in ['v w/o s']:
            semi = '_semi2'
            figN = '(C)'
            plt.suptitle(
                simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-1].upper(),
                fontweight='bold')
        elif simi_tag in ['v w/o c&s']:
            semi = '_semi'
            plt.suptitle(
                'Visual - Unique',
                fontweight='bold')
        else:
            semi = '_semi'
            figN = '(D)'
            plt.suptitle(
                simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-3:].upper(),
                fontweight='bold')
        # fig.text(
        #     x_pos,y_pos,figN,ha='center',
        #     va='top',color='k',fontsize=24,fontweight='bold')
        plt.savefig(
            os.path.join(
                figPath,'%s_glm_%s'%(
                    corr_tag,(simi_tag[0]+semi))))
        plt.show(block=True)
        plt.close('all')
#
mpl.rcParams.update({'font.size':27})
for corr_tag in corr_tags:
    for simi_tag in ['s','c','s w/o c','c w/o s']:
        fig,ax = plt.subplots(
            1,1,sharex=True,sharey=True,figsize=(12,9))
        ax.axhline(
            0,color='black',lw=1,linestyle='dashed')
        leg_tag = True

        sns.lineplot(
            data=glm_data[
                (glm_data['corr']==corr_tag)&
                (glm_data['simi']==simi_tag)&
                (glm_data['layer']=='conv_3')],
            x='setsize',y='coeff',hue='exp',
            style='exp',hue_order=exp_tags,
            style_order=['exp1b','exp2'],
            markers=True,dashes=False,
            palette=clist,linewidth=lw,markersize=mkr_s,
            err_style='bars',errorbar=('se',0),
            legend=leg_tag,ax=ax)

        ax.set_xticks(sizeList)
        ax.set_xlabel(xlabel='MSS')
        ax.set_ylabel(ylabel='Beta')
        ax.set_ylim(-0.12,0.32)
        y_major_locator = MultipleLocator(0.1)
        ax.yaxis.set_major_locator(y_major_locator)

        y_gap = 0.01
        y_sig = -0.1-y_gap
        y_fsig = 0.265+y_gap
        for exp_tag in exp_tags:
            y_sig += y_gap
            dat_exp = glm_data[
                (glm_data['exp']==exp_tag)&
                (glm_data['simi']==simi_tag)&
                (glm_data['layer']=='conv_1')&
                (glm_data['corr']==corr_tag)]
            X = np.array(
                [dat_exp.loc[(dat_exp['setsize']==x_name),
                'coeff'].values for x_name in sizeList])
            X = np.transpose(X,(1,0))

            t_clust,clusters,p_values,H0 = \
                permutation_cluster_1samp_test(
                    X,n_jobs=None,threshold=t_thresh,
                    adjacency=None,seed=seedN,
                    n_permutations=n_permutations,
                    out_type='indices')
            print(clusters)
            print(p_values)

            if (len(clusters)!=0):
                for pN in range(len(p_values)):
                    if (p_values[pN]<p_crit):
                        sig_x = [sizeList[sizeN] for sizeN in \
                                 list(clusters[pN][0])]
                        if exp_tag=='exp1b':
                            lcolor = clist[0]
                            lstyle = 'o'
                        else:
                            lcolor = clist[1]
                            lstyle = 'x'
                        ax.scatter(
                            sig_x,[y_sig]*len(sig_x),
                            c=lcolor,s=mkr_s*2,marker=lstyle)
        h,_ = ax.get_legend_handles_labels()
        ax.legend(
            h,['LTM','STM'],
            loc='upper left',ncol=1,frameon=False).set_title(None)
        sns.despine(offset=10,trim=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.15,right=1,top=0.89,
                            bottom=0.15,hspace=0.3,wspace=0.1)
        # plt.margins(0,0)
        x_pos,y_pos = 0.04,0.97
        if simi_tag in ['s','c']:
            figN = '(A)'
            semi = ''
            plt.suptitle(
                simi_tag.upper(),fontsize=26,fontweight='bold')
        else:
            figN = '(C)'
            semi = '_semi2'
            plt.suptitle(
                simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-1].upper(),
                fontsize=26,fontweight='bold')
        fig.text(
            x_pos,y_pos,figN,ha='center',
            va='top',color='k',fontsize=24,
            fontweight='bold')
        plt.savefig(
            os.path.join(
                figPath,'%s_glm_%s'%(
                    corr_tag,((simi_tag[0]+semi)))))
        plt.show(block=True)
        plt.close('all')
#
#
#
glm_data_6 = glm_data[
    (glm_data['layer']=='fc_6')].reset_index(drop=True)
simi_tagAll = ['v','s','c','v w/o c&s','s w/o v&c','c w/o v&s']
mpl.rcParams.update({'font.size':26})
for corr_tag in corr_tags:
    fig,ax = plt.subplots(
        2,3,sharex=True,sharey=True,
        figsize=(21,16))
    ax = ax.ravel()
    for indx,simi_tag in enumerate(simi_tagAll):
        ax[indx].axhline(
            0,color='black',lw=1,linestyle='dashed')
        if indx==0:
            leg_tag = True
        else:
            leg_tag = False
        dat = glm_data_6[
            (glm_data_6['corr']==corr_tag)&
            (glm_data_6['simi']==simi_tag)]
        sns.barplot(
            data=dat,
            x='setsize',y='coeff',hue='exp',
            hue_order=exp_tags,palette=clist,
            errorbar='se',capsize=0.15,errcolor='grey',
            legend=leg_tag,ax=ax[indx])
        ax[indx].set_xlabel(xlabel='Memory Set Size')
        ax[indx].set_ylabel(ylabel='Beta')
        ax[indx].set_ylim(-0.12,0.35)
        y_major_locator = MultipleLocator(0.1)
        ax[indx].yaxis.set_major_locator(y_major_locator)
        y_top = 0.985
        y_bott = 0.515
        if indx==0:
            figN = 'A'
            ax[indx].set_title(
                'Visual',fontsize=26,fontweight='bold')
            fig.text(
                0.025,y_top,figN,ha='center',
                va='top',color='k',fontsize=28,
                fontweight='bold')
        elif indx==1:
            figN = 'B'
            ax[indx].set_title(
                'Semantic',
                fontsize=26,fontweight='bold')
            fig.text(
                0.395,y_top,figN,ha='center',
                va='top',color='k',fontsize=28,
                fontweight='bold')
        elif indx==2:
            figN = 'C'
            ax[indx].set_title(
                'Categorical',
                fontsize=26,fontweight='bold')
            fig.text(
                0.7,y_top,figN,ha='center',
                va='top',color='k',fontsize=28,
                fontweight='bold')
        elif indx==3:
            figN = 'D'
            ax[indx].set_title(
                'Visual - Unique',
                fontsize=26,fontweight='bold')
            fig.text(
                0.025,y_bott,figN,ha='center',
                va='top',color='k',fontsize=28,
                fontweight='bold')
        elif indx==4:
            figN = 'E'
            ax[indx].set_title(
                'Semantic - Unique',
                fontsize=26,fontweight='bold')
            fig.text(
                0.395,y_bott,figN,ha='center',
                va='top',color='k',fontsize=28,
                fontweight='bold')
        else:
            figN = 'F'
            ax[indx].set_title(
                'Categorical - Unique',
                fontsize=26,fontweight='bold')
            fig.text(
                0.7,y_bott,figN,ha='center',
                va='top',color='k',fontsize=28,
                fontweight='bold')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['LTM','STM'],
        loc='upper left',ncol=1,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15,right=1,top=0.89,
    #                     bottom=0.15,hspace=0.3,wspace=0.1)
    # plt.margins(0,0)
    plt.savefig(
        os.path.join(
            figPath,'%s_glm_6'%(
                corr_tag)))
    plt.show(block=True)
    plt.close('all')
# mpl.rcParams.update({'font.size':24})
# for corr_tag in corr_tags:
#     for simi_tagAll in [['v','v w/o c','v w/o s','v w/o c&s'],
#                         ['s','s w/o v','s w/o c','s w/o v&c'],
#                         ['c','c w/o v','c w/o s','c w/o v&s']]:
#         fig,ax = plt.subplots(
#             2,2,sharex=True,sharey=True,
#             figsize=(20,20))
#         ax = ax.ravel()
#         for indx,simi_tag in enumerate(simi_tagAll):
#             ax[indx].axhline(
#                 0,color='black',lw=1,linestyle='dashed')
#             if indx==0:
#                 leg_tag = True
#             else:
#                 leg_tag = False
#             dat = glm_data_6[
#                 (glm_data_6['corr']==corr_tag)&
#                 (glm_data_6['simi']==simi_tag)]
#             sns.barplot(
#                 data=dat,
#                 x='setsize',y='coeff',hue='exp',
#                 hue_order=exp_tags,palette=clist,
#                 errorbar='se',capsize=0.15,errcolor='grey',
#                 legend=leg_tag,ax=ax[indx])
#             # sns.lineplot(
#             #     data=dat,
#             #     x='setsize',y='coeff',hue='exp',
#             #     style='exp',hue_order=exp_tags,
#             #     style_order=['exp1b','exp2'],
#             #     markers=True,dashes=False,
#             #     palette=clist,linewidth=lw,markersize=mkr_s,
#             #     err_style='bars',errorbar=('se',1),
#             #     err_kws={'capsize':20},
#             #     legend=leg_tag,ax=ax[indx])
#             # ax[indx].set_xticks(sizeList,labels=sizeList)
#             ax[indx].set_xlabel(xlabel='MSS')
#             ax[indx].set_ylabel(ylabel='Beta')
#             ax[indx].set_ylim(-0.12,0.35)
#             y_major_locator = MultipleLocator(0.1)
#             ax[indx].yaxis.set_major_locator(y_major_locator)
#             if indx==0:
#                 figN = '(A)'
#                 ax[indx].set_title(
#                     simi_tag.upper(),fontsize=26,fontweight='bold')
#                 fig.text(
#                     0.02,0.99,figN,ha='center',
#                     va='top',color='k',fontsize=24,
#                     fontweight='bold')
#             elif indx==1:
#                 figN = '(B)'
#                 ax[indx].set_title(
#                     simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-1].upper(),
#                     fontsize=26,fontweight='bold')
#                 fig.text(
#                     0.545,0.99,figN,ha='center',
#                     va='top',color='k',fontsize=24,
#                     fontweight='bold')
#             elif indx==2:
#                 figN = '(C)'
#                 ax[indx].set_title(
#                     simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-1].upper(),
#                     fontsize=26,fontweight='bold')
#                 fig.text(
#                     0.02,0.52,figN,ha='center',
#                     va='top',color='k',fontsize=24,
#                     fontweight='bold')
#             else:
#                 figN = '(D)'
#                 ax[indx].set_title(
#                     simi_tag[0].upper()+simi_tag[1:6]+simi_tag[-3:].upper(),
#                     fontsize=26,fontweight='bold')
#                 fig.text(
#                     0.545,0.52,figN,ha='center',
#                     va='top',color='k',fontsize=24,
#                     fontweight='bold')
#             # y_gap = 0.01
#             # y_sig = -0.1-y_gap
#             # y_fsig = 0.265+y_gap
#             # for exp_tag in exp_tags:
#             #     y_sig += y_gap
#             #     dat_exp = dat[
#             #         (dat['exp']==exp_tag)]
#             #     X = np.array(
#             #         [dat_exp.loc[(dat_exp['setsize']==x_name),
#             #         'coeff'].values for x_name in sizeList])
#             #     X = np.transpose(X,(1,0))
#             #
#             #     t_clust,clusters,p_values,H0 = \
#             #         permutation_cluster_1samp_test(
#             #             X,n_jobs=None,threshold=t_thresh,
#             #             adjacency=None,seed=seedN,
#             #             n_permutations=n_permutations,
#             #             out_type='indices')
#             #     print(clusters)
#             #     print(p_values)
#             #
#             #     if (len(clusters)!=0):
#             #         for pN in range(len(p_values)):
#             #             if (p_values[pN]<p_crit):
#             #                 sig_x = [sizeList[sizeN] for sizeN in \
#             #                          list(clusters[pN][0])]
#             #                 if exp_tag=='exp1b':
#             #                     lcolor = clist[0]
#             #                     lstyle = 'o'
#             #                 else:
#             #                     lcolor = clist[1]
#             #                     lstyle = 'x'
#             #                 ax[indx].scatter(
#             #                     sig_x,[y_sig]*len(sig_x),
#             #                     c=lcolor,s=mkr_s*3,marker=lstyle)
#         h,_ = ax[0].get_legend_handles_labels()
#         ax[0].legend(
#             h,['LTM','STM'],
#             loc='upper left',ncol=1,frameon=False).set_title(None)
#         sns.despine(offset=10,trim=True)
#         plt.tight_layout()
#         # plt.subplots_adjust(left=0.15,right=1,top=0.89,
#         #                     bottom=0.15,hspace=0.3,wspace=0.1)
#         # plt.margins(0,0)
#         plt.savefig(
#             os.path.join(
#                 figPath,'%s_glm6_%s'%(
#                     corr_tag,simi_tag[0])))
#         plt.show(block=True)
#         plt.close('all')
#
# ANOVA
semi_list = ['v w/o c&s','s w/o v&c','c w/o v&s']
dat6 = glm_data[
    (glm_data['layer']=='fc_6')&
    (glm_data['simi'].isin(semi_list))]
# plt_dat_v = dat6[dat6['simi']=='v w/o c&s']
# plt_dat_v['simi'] = 'v w/o c&s'
# plt_dat_s = dat6[dat6['simi']=='s w/o v&c']
# plt_dat_s['simi'] = 's w/o v&c'
# plt_dat_c = dat6[dat6['simi']=='c w/o v&s']
# plt_dat_c['simi'] = 'c w/o v&s'
# plt_dat = pd.concat(
#     [plt_dat_v,plt_dat_s,plt_dat_c],
#     axis=0,ignore_index=True)
mpl.rcParams.update({'font.size':24})
for corr_tag in corr_tags:
    fig,ax = plt.subplots(
        1,3,sharex=True,sharey=True,figsize=(18,6))
    ax = ax.ravel()
    for n,var in enumerate(semi_list):
        if n==0:
            leg_tag = True
        else:
            leg_tag = False

        sns.barplot(
            data=dat6[dat6['simi']==var],
            x='exp',y='coeff',hue='exp',
            hue_order=exp_tags,palette=clist,
            errorbar='se',capsize=0.15,errcolor='grey',
            legend=leg_tag,ax=ax[n])
        ax[n].set_xticks(exp_tags,labels=[])
        ax[n].set_xlabel('Task')
        ax[n].set_ylabel('Beta')
        # ax.set_ylim(-0.05,0.21)
        y_major_locator = MultipleLocator(0.05)
        ax[n].yaxis.set_major_locator(y_major_locator)

        if n==0:
            figN = 'A'
            ax[n].set_title(
                'Visual - Unique',fontsize=23,
                fontweight='bold')
            fig.text(
                0.025,y_top,figN,ha='center',
                va='top',color='k',
                fontweight='bold')
        elif n==1:
            figN = 'B'
            ax[n].set_title(
                'Semantic - Unique',fontsize=23,
                fontweight='bold')
            fig.text(
                0.395,y_top,figN,ha='center',
                va='top',color='k',
                fontweight='bold')
        elif n==2:
            figN = 'C'
            ax[n].set_title(
                'Categorical - Unique',fontsize=23,
                fontweight='bold')
            fig.text(
                0.7,y_top,figN,ha='center',
                va='top',color='k',
                fontweight='bold')

    h, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h, ['LTM','STM'],loc='best',ncol=1,labelcolor=None,
        frameon=False).set_title(None)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'bplt_6_simi_%s.tif'%(corr_tag)))
    plt.show(block=True)
    plt.close('all')