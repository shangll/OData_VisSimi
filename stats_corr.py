#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2024.01.17
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



alex_output = set_filepath(rootPath,'res_alex')
activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
p_crit = 0.05

simi_raw = pd.read_csv(
    os.path.join(alex_output,'expAll_simi_raw.csv'),sep=',')