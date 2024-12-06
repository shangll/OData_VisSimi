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
import statsmodels.api as sm
from sklearn import preprocessing

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


alex_output = set_filepath(rootPath,'res_alex')
save_tag = 0
p_crit = 0.05
activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']
cateList = ['within','between']
exp_tags = ['exp1b','exp2']

exp1b_simi = pd.read_csv(
    os.path.join(alex_output,'exp1b_simi.csv'),sep=',')
exp1b_subj = list(set(exp1b_simi['subj']))
exp2_simi = pd.read_csv(
    os.path.join(alex_output,'exp2_simi.csv'),sep=',')
exp2_subj = list(set(exp2_simi['subj']))


















