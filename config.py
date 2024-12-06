#!/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


show_flg = True
tag_savefile = 1
tag_savefig = 1



def set_filepath(file_path,*path_names):
    for path_name in path_names:
        file_path = os.path.join(file_path, path_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path

rootPath = os.getcwd()
figPath = set_filepath(rootPath,'Figs')
