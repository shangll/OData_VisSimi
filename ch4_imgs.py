#!/usr/bin/env python
#-*-coding:utf-8 -*-

from psychopy import monitors,visual,core,event,gui
from PIL import Image

import os,copy,time,random,csv
import pandas as pd

# windows
mon = monitors.Monitor('testMonitor')
scrSize = (800,600)
win = visual.Window(monitor=mon,color=(1,1,1),size=scrSize,fullscr=False,units='deg')

win.mouseVisible = False
timer = core.Clock()
#
# get path
filePath = os.getcwd()
stim_all = pd.read_csv(os.path.join(filePath,'stim_full_list.csv'),sep=',')
ans_list = []

for img in stim_all['link'].tolist():
    img_pres = visual.ImageStim(
    win,image=img,pos=(0.0,0.0),units='deg')
    img_pres.draw()
    win.flip()
    respKey = event.waitKeys(
    keyList=['space'],timeStamped=True,clearEvents=True)
    
#    ans_box=visual.TextBox(
#    window=win,text="input",font_size=18,font_color=[-1,-1,1],
#    color_space='rgb',size=(1.8,.1),pos=(0.0,.5),units='norm')
#    ans_box.draw()
#    
#    ans_list.append(respKey[0][0].lower())

#stim_all['ans'] = ans_list
#stim_all.to_csv(os.path.join(filePath,'stim_full_list.csv'),
#                sep=',',mode='w',header=True,index=False)