#!/usr/bin/env python
#-*-coding:utf-8 -*-

# 2024.02.13


import os
import pandas as pd


filePath = 'U:/Documents/DCC/exp1b'

anim_list = pd.read_csv(os.path.join(filePath,'StimList/animList.csv'),sep=',')
obj_list = pd.read_csv(os.path.join(filePath,'StimList/objList.csv'),sep=',')

allstim_list = pd.concat([anim_list,obj_list],axis=0,ignore_index=True)

allStims = pd.DataFrame()
for subcate_file in allstim_list['stimSubCate'].tolist():
    stim_df = pd.read_csv(os.path.join(filePath,subcate_file),sep=',')
    allStims = pd.concat([allStims,stim_df],axis=0,ignore_index=True)

allStims['image'] = allStims['stimulus'].str.split('/',expand=True)[3]
webPath = 'https://github.com/shangll/OData_VisMemSch/tree/main/ExpProgram/exp1a/'

web_list = []
for indx,img_name in enumerate(allStims['stimulus'].tolist()):
    web_list.append(webPath+img_name)
allStims['link'] = web_list


allStims.to_csv('U:/Documents/DCC/ch4_LSTM/res_alex/stim_full_list.csv',
                sep=',',mode='w',header=True,index=False)