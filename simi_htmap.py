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
import seaborn as sns
import matplotlib.pyplot as plt



# file_tag = 'sgpt'
file_tag = 'w2v'
sgpt_filepath = set_filepath(rootPath,'res_%s'%file_tag)
simi_mtrx = pd.read_csv(os.path.join(sgpt_filepath,'img_%s_simi.csv'%file_tag))

dat_matrx = simi_mtrx.copy(deep=True)
name_list = dat_matrx['name'].to_list()
dat_matrx.head()
dat_matrx.drop(labels=['cate','image','name'],axis=1,inplace=True)
dat_matrx.columns = name_list
dat_matrx.index = name_list

sns.heatmap(data=dat_matrx,annot=True,cmap='RdBu_r')
plt.savefig(os.path.join(figPath,'%s_heatmap.png'%file_tag))
plt.show(block=True)
plt.close('all')


