#!/usr/bin/env python
#-*-coding:utf-8 -*-

# AlexNet
# 2023.11.17
# linlin.shang@donders.ru.nl


from config import set_filepath,rootPath
import os
from PIL import Image
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torchvision import transforms,models


save_tag = 1
p_crit = 0.05
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# 1. loading data
#
alex_output = set_filepath(rootPath,'res_alex')
# load image file into list
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


# 2. AlexNet 5+3
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation_names = ['conv_%d'%k if k<6 else 'fc_%d'%k for k in range(1,9)]
alex_model = models.alexnet(pretrained=True)
model_children = list(alex_model.children())
for name,n,k in zip(activation_names,[0]*5+[2]*3,[1,4,7,9,11,2,5,6]):
    model_children[n][k].register_forward_hook(
        get_activation(name))
#
preprocess = transforms.Compose([
    transforms.Resize(256),transforms.CenterCrop(224),
    transforms.ToTensor(),transforms.Normalize(
        mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
#
layerVectors = {}
for name in activation_names:
    imgVectors = {}
    for imgPath in imgPathList:
        img = Image.open(os.path.join(rootPath,imgPath))
        input_tensor = preprocess(img)
        input_batch = torch.unsqueeze(input_tensor,dim=0)
        with torch.no_grad():
            outputs = alex_model(input_batch)
        imgName = imgPath.split('/')[3]
        imgVectors[imgName] = activation[name]
        img.close()
    layerVectors[name] = imgVectors
    print(activation[name].shape)

def corrMtrx(a,b):
    arr_a = np.ndarray.flatten(a)
    arr_b = np.ndarray.flatten(b)
    return stats.spearmanr(arr_a,arr_b)[0]

def getSimiMatrix(vectors):
    for imgName in imgNameList:
        vectors[imgName] = np.ndarray.flatten(vectors[imgName].detach().numpy()[0])
    vectors_df = pd.DataFrame(vectors)
    r_df = vectors_df.corr(method='spearman')
    return r_df

simi_mtrx = pd.DataFrame()
for name in activation_names:
    print(name)
    corr_dat = getSimiMatrix(layerVectors[name])
    corr_dat['layer'] = name
    simi_mtrx = pd.concat([simi_mtrx,corr_dat],ignore_index=True,axis=0)


# get simi matrix
simi_mtrx = pd.DataFrame()
indx = 0
for name in activation_names:
    print(name)
    old_imgs = []
    for imgName_a in imgNameList:
        old_imgs.append(imgName_a)
        for imgName_b in imgNameList:
            if imgName_b not in old_imgs:
                if imgDF.loc[imgDF['img']==imgName_a,'subCate'].values != \
                        imgDF.loc[imgDF['img']==imgName_b,'subCate'].values:
                    simi_mtrx.loc[indx,'layer'] = name
                    if imgDF.loc[
                        imgDF['img']==imgName_a,'cate'].values[0]==\
                            imgDF.loc[imgDF['img']==imgName_b,'cate'].values[0]:
                        simi_mtrx.loc[indx, 'cate'] = 'within'
                    else:
                        simi_mtrx.loc[indx, 'cate'] = 'between'

                    simi_mtrx.loc[indx, 'r'] = corrMtrx(
                        layerVectors[name][imgName_a].detach().numpy()[0],
                        layerVectors[name][imgName_b].detach().numpy()[0])
            indx += 1

# simi_mtrx = pd.DataFrame()
# indx = 0
# for name in activation_names:
#     print(name)
#     old_imgs = []
#     for imgName_a in imgNameList:
#         old_imgs.append(imgName_a)
#         for imgName_b in imgNameList:
#             if imgName_b not in old_imgs:
#                 if imgDF.loc[imgDF['img']==imgName_a,'subCate'].values != \
#                         imgDF.loc[imgDF['img']==imgName_b,'subCate'].values:
#                     simi_mtrx.loc[indx,'layer'] = name
#                     simi_mtrx.loc[indx,'img_A'] = imgName_a
#                     simi_mtrx.loc[indx,'cate_A'] = imgDF.loc[
#                         imgDF['img']==imgName_a,'cate'].values[0]
#                     simi_mtrx.loc[indx,'subcate_A'] = imgDF.loc[
#                         imgDF['img']==imgName_a,'subCate'].values[0]
#                     simi_mtrx.loc[indx,'img_B'] = imgName_b
#                     simi_mtrx.loc[indx,'cate_B'] = imgDF.loc[
#                         imgDF['img']==imgName_b,'cate'].values[0]
#                     simi_mtrx.loc[indx,'subcate_B'] = imgDF.loc[
#                         imgDF['img']==imgName_b,'subCate'].values[0]
#
#                     simi_mtrx.loc[indx, 'r'] = corrMtrx(
#                         layerVectors[name][imgName_a].detach().numpy()[0],
#                         layerVectors[name][imgName_b].detach().numpy()[0])
#             indx += 1
# simi_mtrx.drop(
#     simi_mtrx[
#         ((simi_mtrx['cate_A']=='Objects')&
#         (simi_mtrx['cate_B']=='Animals'))|
#         (simi_mtrx['r']==1)].index,
#     inplace=True)
if save_tag==1:
    simi_mtrx.to_csv(
        os.path.join(
            alex_output,'img_alex_simi.csv'),mode='w',header=True,index=False)