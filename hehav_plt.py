#!/usr/bin/env python
#-*-coding:utf-8 -*-

# ch.4
# linlin.shang@donders.ru.nl


from config import figPath

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

model = LinearRegression()
model2 = LinearRegression()
fit2 = LinearRegression()


# --- --- --- Set Global Parameters --- --- --- #

exp_tags = ['exp1b','exp2']
cate_list = ['within','between']
size_list = [1,2,4,8]
filePath = os.getcwd()
exp1b = pd.read_csv(
    os.path.join(filePath,'exp1b_mean_subjs.csv'),sep=',')
exp1b['exp'] = 'exp1b'
exp2 = pd.read_csv(
    os.path.join(filePath,'exp2_mean_subjs.csv'),sep=',')
exp2['exp'] = 'exp2'
exp_all = pd.concat([exp1b,exp2],axis=0,ignore_index=True)
exp_all = exp_all[exp_all['cond']!='target']

count = 0
dfCoeff = pd.DataFrame()
for exp_tag in exp_tags:
    exp_data = exp_all[exp_all['exp']==exp_tag]
    subj_list = list(set(exp_data['subj'].tolist()))

    for n in subj_list:
        for cate_tag in cate_list:
            dfCoeff.loc[count,'cond'] = cate_tag
            dfCoeff.loc[count,'exp'] = exp_tag
            dfCoeff.loc[count,'subj'] = n

            subj_data = exp_data[
                (exp_data['cond']==cate_tag)&
                (exp_data['subj']==n)&
                (exp_data['setsize']!=8)]

            x = subj_data['setsize'].values
            y = subj_data['rt'].values
            model.fit(x.reshape(-1,1),y)
            exp_all.loc[
                (exp_all['exp']==exp_tag)&
                (exp_all['cond']==cate_tag)&
                (exp_all['subj']==n),'lm'] = model.predict(
                np.array(size_list).reshape(-1,1))

            x2 = np.log2(subj_data['setsize'].values)
            model2.fit(x2.reshape(-1,1),y)
            exp_all.loc[
                (exp_all['exp']==exp_tag)&
                (exp_all['cond']==cate_tag)&
                (exp_all['subj']==n),'log'] = model2.predict(
                np.array([0,1,2,3]).reshape(-1,1))

            fit_x2 = np.array([0,1,2,3])
            fit_y = exp_data.loc[
                (exp_data['cond']==cate_tag)&
                (exp_data['subj']==n),'rt'].values
            fit2.fit(fit_x2.reshape(-1,1),fit_y)
            dfCoeff.loc[count,'coeff'] = fit2.coef_[0]

            count += 1

# exp1b = exp_all[exp_all['exp']=='exp1b']
# exp2 = exp_all[exp_all['exp']=='exp2']
# exp1b_mean = exp1b.groupby(
#     ['setsize','cond'])['rt'].agg(np.mean).reset_index()
# exp1b_mean['exp'] = 'exp1b'
# exp2_mean = exp2.groupby(
#     ['setsize','cond'])['rt'].agg(np.mean).reset_index()
# exp2_mean['exp'] = 'exp2'
exp_mean = exp_all.groupby(
    ['exp','setsize','cond'])[['rt','lm','log']].agg(np.mean).reset_index()

#
#
#
mpl.rcParams.update({'font.size':24})
fig,ax = plt.subplots(1,3,figsize=(24,8))
ax = ax.ravel()
clrs = ['black','grey','dodgerblue','lightskyblue','dodgerblue','lightskyblue']
for indx,exp_tag in enumerate(exp_tags):
    if indx==0:
        leg_tag = True
        fig_lab ='(A)'
        title_name = 'LTM'
        y_tic = ['0.5','0.6','0.7','0.8']
        y_lab = 'RT (sec)'
    else:
        leg_tag = False
        fig_lab ='(B)'
        title_name = 'STM'
    sns.scatterplot(
        data=exp_mean[(exp_mean['exp']==exp_tag)],
        x='setsize',y='rt',hue='cond',hue_order=cate_list,
        s=190,style='cond',
        style_order=cate_list,
        palette=['grey','black'],legend=leg_tag,ax=ax[indx])

    sns.lineplot(
        data=exp_mean[exp_mean['exp']==exp_tag],
        x='setsize',y='lm',hue='cond',dashes=True,
        hue_order=cate_list,markers=True,
        linestyle='--',palette='Blues',linewidth=5,
        legend=leg_tag,ax=ax[indx])

    sns.lineplot(
        data=exp_mean[exp_mean['exp']==exp_tag],
        x='setsize',y='log',hue='cond',dashes=False,
        hue_order=cate_list,markers=True,
        linestyle='solid',palette='Blues',linewidth=5,
        legend=leg_tag,ax=ax[indx])

    ax[indx].set_xticks(size_list)
    if indx==1:
        y_tic = []
        y_lab = None

    ax[indx].set_yticks(np.arange(0.5,0.81,0.1),labels=y_tic)
    ax[indx].set_ylim(0.5,0.82)
    ax[indx].set_xlabel('Memory Set Size')
    ax[indx].set_ylabel('RT (sec)')
    ax[indx].set_title(title_name)
    count = 0
    for y_var in ['rt','lm','log']:
        val_list = list(
            exp_mean.loc[
                (exp_mean['exp']==exp_tag)&(exp_mean['setsize']==8),y_var].values)
        x_pos = [8]*len(val_list)
        for num in range(len(val_list)):
            if exp_tag=='exp1b' and count==0:
                ax[indx].annotate(
                    str(val_list[num].round(3)),xy=(x_pos[num],val_list[num]),
                    xytext=(x_pos[num]+0.05,0.557),color=clrs[count],size=21)
            elif exp_tag=='exp1b' and count==2:
                ax[indx].annotate(
                    str(val_list[num].round(3)),xy=(x_pos[num],val_list[num]),
                    xytext=(x_pos[num]+0.05,0.62),color=clrs[count],size=21)
            elif exp_tag=='exp2' and count==4:
                ax[indx].annotate(
                    str(val_list[num].round(3)),xy=(x_pos[num],val_list[num]),
                    xytext=(x_pos[num]+0.05,0.59),color=clrs[count],size=21)
            elif exp_tag=='exp2' and count==5:
                ax[indx].annotate(
                    str(val_list[num].round(3)),xy=(x_pos[num],val_list[num]),
                    xytext=(x_pos[num]+0.05,0.711),color=clrs[count],size=21)
            else:
                ax[indx].annotate(
                    str(val_list[num].round(3)),xy=(x_pos[num],val_list[num]),
                    xytext=(x_pos[num]+0.05,val_list[num]),color=clrs[count],size=21)
            count += 1
    ax[indx].text(0,0.835,fig_lab,ha='center',va='top',color='k')

sns.barplot(
    data=dfCoeff,x='exp',y='coeff',hue='cond',
    hue_order=['within','between'],errorbar='se',
    capsize=0.15,errcolor='grey',errwidth=1.5,
    palette='Blues',legend=True,ax=ax[2])
ax[2].text(-0.68,0.0755,'(C)',ha='center',va='top',color='k')
ax[2].set_yticks(np.arange(0.0,0.071,0.035))
ax[2].set_xlabel('Task')
ax[2].set_ylabel('Coefficients')
ax[2].set_xticks(exp_tags,['LTM','STM'])
ax[2].set_title('Slopes of RTs in Target-Absent Trials',fontsize=22)

h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(
    h,['observed data (within)','observed data (between)',
       'linear (within)','linear (between)',
       'log2 (within)','log2 (between)'],loc='upper left',ncol=1,
    fontsize=20,frameon=False).set_title(None)
h,_ = ax[2].get_legend_handles_labels()
ax[2].legend(
    h,['within','between'],loc='upper left',ncol=1,
    fontsize=20,frameon=False).set_title(None)
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(
        os.path.join(figPath,'behav_fit.tif'))
plt.show(block=True)
plt.close('all')


