#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2024.03.
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

simi_raw = pd.read_csv(
    os.path.join(alex_output,'expAll_simi_raw.csv'),sep=',')
col_names = ['sgpt_mean','sgpt_max']
exp1b_w2v = pd.read_csv(
    os.path.join(w2v_output,'exp1b_simi_w2v.csv'),sep=',')
exp1b_w2v['exp'] = 'exp1b'
exp2_w2v = pd.read_csv(
    os.path.join(w2v_output,'exp2_simi_w2v.csv'),sep=',')
exp2_w2v['exp'] = 'exp2'
simi_raw['w2v_mean'] = exp1b_w2v['sgpt_mean'].to_list()*8+\
                       exp2_w2v['sgpt_mean'].to_list()*8
simi_raw['w2v_max'] = exp1b_w2v['sgpt_max'].to_list()*8+\
                      exp2_w2v['sgpt_max'].to_list()*8
simi_raw['cate_trans'] = np.where(
    simi_raw['cate']=='within',1,-1)
simi_raw.to_csv(
    os.path.join(w2v_output,'expAll_simi.csv'),
    sep=',',mode='w',header=True,index=False)

# print(stats.pointbiserialr(simi_raw['cate_trans'],simi_raw['w2v_mean']))
# print(stats.pointbiserialr(simi_raw['cate_trans'],simi_raw['w2v_max']))

w2v_cate = simi_raw[simi_raw['layer']=='conv_1']
print(round(w2v_cate.loc[w2v_cate['cate']=='within','w2v_mean'].mean(),3))
print(round(w2v_cate.loc[w2v_cate['cate']=='between','w2v_mean'].mean(),3))
s_w = w2v_cate.loc[w2v_cate['cate']=='within','w2v_mean'].values
s_b = w2v_cate.loc[w2v_cate['cate']=='between','w2v_mean'].values

# w2v_cate.to_csv(
#     os.path.join(w2v_output,'w2v_cate.csv'),
#     sep=',',mode='w',header=True,index=False)

# from scipy.stats import ttest_ind
# def statistic(x,y,axis):
#     return np.mean(x,axis=axis)-np.mean(y,axis=axis)
#
#
#
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

# t_result = ttest_ind(s_w,s_b)
# t_result.statistic
permu_t(s_w,s_b)


for name in activation_names:
    print(name)
    i_w = simi_raw.loc[(simi_raw['layer']==name)&
                       (simi_raw['cate']=='within'),'simi_mean'].values
    i_b = simi_raw.loc[(simi_raw['layer']==name)&
                       (simi_raw['cate']=='between'),'simi_mean'].values

    print(round(
        simi_raw.loc[(simi_raw['layer']==name)&
                     (simi_raw['cate']=='within'),
        'simi_mean'].mean(),3))
    print(round(
        simi_raw.loc[(simi_raw['layer']==name)&
                     (simi_raw['cate']=='between'),
        'simi_mean'].mean(),3))

    permu_t(i_w, i_b)

    r,p = stats.spearmanr(
        simi_raw.loc[(simi_raw['layer']==name),'simi_mean'].values,
        w2v_cate['w2v_mean'].values)


    def statistic(x, y):
        return stats.spearmanr(x, y).statistic
    res = stats.permutation_test(
        (simi_raw.loc[(simi_raw['layer']==name),'simi_mean'].values,
         w2v_cate['w2v_mean'].values),statistic,vectorized=False,
        permutation_type='pairings',alternative='two-sided')
    r,p = res.statistic,res.pvalue
    print('correlation: %0.3f, %0.3f'%(r,p))



# res = permutation_test(
#     (s_w,s_b),statistic,permutation_type='independent',vectorized=True,
#     n_resamples=n_permutations,alternative='two-sided')
# print('permu: %0.3f, %0.3f' % (res.statistic,res.pvalue))

import pingouin as pg

img_cate = simi_raw.groupby(
    ['exp','subj','cate','layer'])['simi_mean'].agg(np.mean).reset_index()
w2v_cate_mean = w2v_cate.groupby(
    ['exp','subj','cate'])['w2v_mean'].agg(np.mean).reset_index()
for name in activation_names:
    print(name)
    i_w = img_cate.loc[(img_cate['layer'] == name) &
                       (img_cate['cate'] == 'within'), 'simi_mean'].values
    i_b = img_cate.loc[(img_cate['layer'] == name) &
                       (img_cate['cate'] == 'between'), 'simi_mean'].values

    print(round(
        img_cate.loc[(img_cate['layer']==name)&
                     (img_cate['cate']=='within'),
        'simi_mean'].mean(),3))
    print(round(
        img_cate.loc[(img_cate['layer']==name)&
                     (img_cate['cate']=='between'),
        'simi_mean'].mean(),3))

    t,p = stats.ttest_ind(i_w,i_b)
    print('t = %0.3f, p = %0.3f' %(t,p))
    t,p = stats.ttest_rel(i_w,i_b)
    print('t = %0.3f, p = %0.3f'%(t,p))
    r,p = stats.spearmanr(
        img_cate.loc[(img_cate['layer']==name),'simi_mean'].values,
        w2v_cate_mean['w2v_mean'].values)
    print('correlation: %0.3f, %0.3f'%(r,p))

# img_cate.to_csv(
#     os.path.join(w2v_output,'img_cate.csv'),
#     sep=',',mode='w',header=True,index=False)






#
fig,ax = plt.subplots(
            1,4,sharex=True,sharey=True,figsize=(19,6))
ax = ax.ravel()
n = 0
for n,sizeN in enumerate(sizeList):
    ax[n].axhline(0,color='black',lw=1,linestyle='dashed')

    sns.lineplot(data=simi_raw[(simi_raw['setsize']==sizeN)],
                 x='layer',y='w2v_mean',hue='exp',style='exp',
                 hue_order=exp_tags,style_order=['exp1b','exp2'],
                 markers=True,dashes=False,
                 linewidth=2,markersize=10,err_style='bars',
                 errorbar=('se',0),ax=ax[n])
plt.tight_layout()
plt.show(block=True)
plt.close('all')
#
fig,ax = plt.subplots(
            1,1,sharex=True,sharey=True,figsize=(19,6))
ax.axhline(0,color='black',lw=1,linestyle='dashed')

sns.lineplot(data=pd.concat([exp1b_w2v,exp2_w2v],axis=0),
             x='setsize',y='sgpt_mean',hue='exp',style='exp',
             hue_order=exp_tags,style_order=['exp1b','exp2'],
             markers=True,dashes=False,
             linewidth=2,markersize=10,err_style='bars',
             errorbar=('se',0),ax=ax)
plt.tight_layout()
plt.show(block=True)
plt.close('all')

#
#
#
# GLM
#
#
#

# GLM (RT ~ semantic)

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
                    exp_simi_indv['w2v_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

                    # GLM fit
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv['w2v_Z']
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y,X,family=sm.families.Gaussian()).fit()

                    glm_cond.append('intc')
                    glm_coeff.append(model.params[0])
                    glm_cond.append('w2v')
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
glm_data = glm_data[glm_data['cond']=='w2v']

glm_ealy = (glm_data.loc[
               (glm_data['cond']=='w2v')&
               (glm_data['layer']=='conv_1'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='w2v')&
    (glm_data['layer']=='conv_2'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='w2v')&
    (glm_data['layer']=='conv_3'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='w2v')&
    (glm_data['layer']=='conv_4'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='w2v')&
    (glm_data['layer']=='conv_5'),'coeff'].values)/5
glm_late = (glm_data.loc[
               (glm_data['cond']=='w2v')&
               (glm_data['layer']=='fc_6'),'coeff'].values+glm_data.loc[
    (glm_data['cond']=='w2v')&
    (glm_data['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='w2v')]
glm_1 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='w2v')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='w2v')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(w2v_output,'glm_rt-w2v.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(w2v_output,'glm_rt-w2v_3layers.csv'),
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
                        sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        os.path.join(figPath,'glm_rt-w2v_MSS_%s.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

# GLM (resid RT ~ w2v + semi(Similarity))

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
                    exp_simi_indv['w2v_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

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
                    X = exp_simi_indv['w2v_Z']
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
glm_data = glm_data[glm_data['cond']=='cate']

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
glm_1 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='cate')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='cate')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(w2v_output,'glm_resid-w2v.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(w2v_output,'glm_resid-w2v_3layers.csv'),
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
        sns.lineplot(data=dat[(dat['corr']==corr_tag)&(dat['setsize']==sizeN)],
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
                        sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        os.path.join(figPath,'w2v_resid_cate_MSS_%s.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

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
        ax[n].text(0.2,0.23,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_w2v_resid_cate_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')


# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (resid RT ~ Similarity + semi(semantic))

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
                    exp_simi_indv['w2v_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

                    # GLM fit
                    # 1st step
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv['w2v_Z']
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
glm_data = glm_data[glm_data['cond']=='simi']

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

glm_data.to_csv(os.path.join(w2v_output,'glm_resid-simi.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(w2v_output,'glm_resid-simi_3layers.csv'),
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
                        sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        os.path.join(figPath,'w2v_resid_simi_MSS_%s.tif'%corr_tag))
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
        y_major_locator = MultipleLocator(0.125)
        ax[n].yaxis.set_major_locator(y_major_locator)
        ax[n].set_title(exp_title)
        ax[n].text(0.3,0.28,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_w2v_resid_simi_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')



# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (rt ~ w2v) within/between

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
                        exp_simi_indv['w2v_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

                        # GLM fit
                        y = exp_simi_indv['rt_Z']
                        X = exp_simi_indv['w2v_Z']
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
glm_data_avg = glm_data[(glm_data['cate']=='within')&
                        (glm_data['cond']=='simi')].copy(deep=True)
glm_data_avg.drop(labels=['cate','coeff'],axis=1,inplace=True)
glm_data_avg['coeff'] = (glm_data.loc[
                             (glm_data['cate']=='within')&
                             (glm_data['cond']=='simi'),'coeff'].values\
                         +glm_data.loc[
                             (glm_data['cate']=='between')&
                             (glm_data['cond']=='simi'),'coeff'].values)/2
final_col = ['exp','subj','block','cate','setsize',
             'rt','layer','simi_mean','simi_max',
             'resid_mean','resid_max']
simi_data = simi_raw[final_col]
glm_data = glm_data[(glm_data['cond']!='intc')]


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
glm_8 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&(glm_data_avg['cond']=='simi')]
glm_1 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&
    (glm_data_avg['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&
    (glm_data_avg['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_avg_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(w2v_output,'glm_rt-s-2cate.csv'),
                sep=',',mode='w',header=True,index=False)
glm_data_avg.to_csv(os.path.join(w2v_output,'glm_rt-s-2avg.csv'),
                    sep=',',mode='w',header=True,index=False)
glm_avg_meanLayer.to_csv(
    os.path.join(w2v_output,'glm_rt-s-2avg_3layers.csv'),
    sep=',',mode='w',header=True,index=False)

# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (resid RT ~ similarity +semi(w2v)) within/between

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
                        exp_simi_indv['w2v_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

                        # GLM fit
                        # 1st step
                        y = exp_simi_indv['rt_Z']
                        X = exp_simi_indv['w2v_Z']
                        X = sm.add_constant(X)
                        model = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        # 2nd step
                        y = np.array(list(model.resid_response))
                        X = exp_simi_indv['simi_Z']
                        X = sm.add_constant(X)
                        model2 = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        simi_raw.loc[
                            (simi_raw['exp']==exp_tag)&
                            (simi_raw['layer']==name)&
                            (simi_raw['setsize']==n)&
                            (simi_raw['cate']==cate)&
                            (simi_raw['subj']==k),'resid_%s'%corr_tag] = \
                            list(model2.resid_response)

                        glm_cond.append('intc')
                        glm_coeff.append(model2.params[0])
                        glm_cond.append('simi')
                        glm_coeff.append(model2.params[1])
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
glm_data_avg = glm_data[(glm_data['cate']=='within')&
                        (glm_data['cond']=='simi')].copy(deep=True)
glm_data_avg.drop(labels=['cate','coeff'],axis=1,inplace=True)
glm_data_avg['coeff'] = (glm_data.loc[
                             (glm_data['cate']=='within')&
                             (glm_data['cond']=='simi'),'coeff'].values+\
                         glm_data.loc[
                             (glm_data['cate']=='between')&
                             (glm_data['cond']=='simi'),'coeff'].values)/2
final_col = ['exp','subj','block','cate','setsize',
             'rt','layer','simi_mean','simi_max',
             'resid_mean','resid_max']
simi_data = simi_raw[final_col]
glm_data = glm_data[(glm_data['cond']!='intc')]



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
               (glm_data_avg['layer']=='conv_1'),'coeff'].values+glm_data_avg.loc[
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
glm_8 = glm_data_avg[(glm_data_avg['layer']=='fc_8')&(glm_data_avg['cond']=='simi')]
glm_1 = glm_data_avg[(glm_data_avg['layer']=='fc_8')&(glm_data_avg['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data_avg[(glm_data_avg['layer']=='fc_8')&(glm_data_avg['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_avg_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(w2v_output,'glm_resid-2cate_v.csv'),
                sep=',',mode='w',header=True,index=False)
glm_data_avg.to_csv(os.path.join(w2v_output,'glm_resid-2avg_v.csv'),
                    sep=',',mode='w',header=True,index=False)
glm_avg_meanLayer.to_csv(os.path.join(w2v_output,'glm_resid-2avg_3layers_v.csv'),
                         sep=',',mode='w',header=True,index=False)

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
                            sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        h,['Category','within','between','Task','LTM','STM'],loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
            os.path.join(figPath,'%s_w2v_resid-2cate_MSS.tif'%corr_tag))
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
        ax[n].text(0.1,0.22,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_w2v_avg2cate_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')

# plot average
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
                        sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        os.path.join(figPath,'%s_w2v_resid-2avg_MSS.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')

# --- * --- * --- * --- * --- * --- * --- * --- * ---

# GLM (resid RT ~ w2v + semi(simi)) within/between

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
                        exp_simi_indv['w2v_Z'] = preprocessing.scale(
                            exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

                        # GLM fit
                        # 1st step
                        y = exp_simi_indv['rt_Z']
                        X = exp_simi_indv['simi_Z']
                        X = sm.add_constant(X)
                        model = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        # 2nd step
                        y = np.array(list(model.resid_response))
                        X = exp_simi_indv['w2v_Z']
                        X = sm.add_constant(X)
                        model2 = sm.GLM(
                            y,X,family=sm.families.Gaussian()).fit()

                        simi_raw.loc[
                            (simi_raw['exp']==exp_tag)&
                            (simi_raw['layer']==name)&
                            (simi_raw['setsize']==n)&
                            (simi_raw['cate']==cate)&
                            (simi_raw['subj']==k),'resid_%s'%corr_tag] = \
                            list(model2.resid_response)

                        glm_cond.append('intc')
                        glm_coeff.append(model2.params[0])
                        glm_cond.append('simi')
                        glm_coeff.append(model2.params[1])
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
glm_data_avg = glm_data[(glm_data['cate']=='within')&
                        (glm_data['cond']=='simi')].copy(deep=True)
glm_data_avg.drop(labels=['cate','coeff'],axis=1,inplace=True)
glm_data_avg['coeff'] = (glm_data.loc[
                             (glm_data['cate']=='within')&
                             (glm_data['cond']=='simi'),'coeff'].values+\
                         glm_data.loc[
    (glm_data['cate']=='between')&
    (glm_data['cond']=='simi'),'coeff'].values)/2
final_col = ['exp','subj','block','cate','setsize',
             'rt','layer','simi_mean','simi_max',
             'resid_mean','resid_max']
simi_data = simi_raw[final_col]
glm_data = glm_data[(glm_data['cond']!='intc')]



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
glm_1 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data[(glm_data['layer']=='fc_8')&(glm_data['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_ealy = (glm_data_avg.loc[
               (glm_data_avg['cond']=='simi')&
               (glm_data_avg['layer']=='conv_1'),'coeff'].values+glm_data_avg.loc[
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
               (glm_data_avg['layer']=='fc_6'),'coeff'].values+\
            glm_data_avg.loc[
                (glm_data_avg['cond']=='simi')&
                (glm_data_avg['layer']=='fc_7'),'coeff'].values)/2
glm_8 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&(glm_data_avg['cond']=='simi')]
glm_1 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&
    (glm_data_avg['cond']=='simi')].copy(deep=True)
glm_1['layer'] = 'early'
glm_1['coeff'] = glm_ealy
glm_6 = glm_data_avg[
    (glm_data_avg['layer']=='fc_8')&
    (glm_data_avg['cond']=='simi')].copy(deep=True)
glm_6['layer'] = 'late'
glm_6['coeff'] = glm_late
glm_avg_meanLayer = pd.concat([glm_1,glm_6,glm_8],axis=0,ignore_index=True)

glm_data.to_csv(os.path.join(w2v_output,'glm_resid-2cate_s.csv'),
                sep=',',mode='w',header=True,index=False)
glm_data_avg.to_csv(os.path.join(w2v_output,'glm_resid-2avg_s.csv'),
                    sep=',',mode='w',header=True,index=False)
glm_avg_meanLayer.to_csv(
    os.path.join(w2v_output,'glm_resid-2avg_3layers_s.csv'),
    sep=',',mode='w',header=True,index=False)

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
                            sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        h,['Category','within','between','Task','LTM','STM'],loc='upper left',ncol=2,
        fontsize=12,frameon=False).set_title(None)
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
            os.path.join(figPath,'%s_w2v_resid-2cate_s_MSS.tif'%corr_tag))
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
        ax[n].text(0.1,0.12,figN,ha='center',va='top',color='k')
    h,_ = ax[0].get_legend_handles_labels()
    ax[0].legend(
        h,['Early','Late','FC 8'],
        loc='upper left',ncol=1,fontsize=16,frameon=False).set_title(None)

    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figPath,'aov_w2v_avg2cate_s_%s.tif'% corr_tag))
    plt.show(block=True)
    plt.close('all')

# plot average
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
                        sig_x = ['conv_%d'%(layerN+1) if (layerN+1)<6 else 'fc_%d'%(layerN+1)\
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
        os.path.join(figPath,'%s_w2v_resid-2avg_s_MSS.tif'%corr_tag))
    plt.show(block=True)
    plt.close('all')


# GLM (resid RT ~ cate + semi(w2v))

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
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'cate_trans'])
                    exp_simi_indv['w2v_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:,'w2v_%s'%corr_tag])

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
                    X = exp_simi_indv['w2v_Z']
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
glm_data = glm_data[glm_data['cond']=='simi']

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

glm_data.to_csv(os.path.join(w2v_output,'glm_resid-c.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(w2v_output,'glm_resid-c_3layers.csv'),
                     sep=',',mode='w',header=True,index=False)



# GLM (resid RT ~ category; semi(visual+w2v))
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
                        exp_simi_indv.loc[:, 'rt'])
                    exp_simi_indv['simi_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:, 'simi_%s' % corr_tag])
                    exp_simi_indv['cate_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:, 'cate_trans'])
                    exp_simi_indv['w2v_Z'] = preprocessing.scale(
                        exp_simi_indv.loc[:, 'w2v_%s' % corr_tag])

                    # GLM fit
                    # 1st step
                    y = exp_simi_indv['rt_Z']
                    X = exp_simi_indv[['simi_Z','w2v_Z']]
                    X = sm.add_constant(X)
                    model = sm.GLM(
                        y, X, family=sm.families.Gaussian()).fit()

                    # 2nd step
                    y = np.array(list(model.resid_response))
                    # X = model.predict()-exp_simi_indv['cate_Z'].values
                    X = exp_simi_indv['cate_Z']
                    X = sm.add_constant(X)
                    model2 = sm.GLM(
                        y, X, family=sm.families.Gaussian()).fit()

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
glm_data = glm_data[glm_data['cond']=='simi']

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

glm_data.to_csv(os.path.join(w2v_output,'glm_resid-c_vs.csv'),
                sep=',',mode='w',header=True,index=False)
glm_meanLayer.to_csv(os.path.join(w2v_output,'glm_resid-c_vs_3layers.csv'),
                     sep=',',mode='w',header=True,index=False)
