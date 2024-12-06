#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# Resnet
# 2023.11.15
# linlin.shang@donders.ru.nl


from config import set_filepath,rootPath,figPath
import os
from PIL import Image
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torchvision import transforms,models

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



# load images into data frame
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
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# Rescaling
inputDim = (224,224)
inputDirCNN = "inputImagesCNN"
transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
for imageName in imgPathList:
    I = Image.open(os.path.join(rootPath,imageName))
    newI = transformationForCNNInput(I)
    if "exif" in I.info:
        exif = I.info['exif']
        newI.save(
            os.path.join(
                set_filepath(
                    rootPath,inputDirCNN),
                imageName.split('/')[-1]),exif=exif)
    else:
        newI.save(
            os.path.join(
                set_filepath(
                    rootPath,inputDirCNN),
                imageName.split('/')[-1]))

# Creating the similarity matrix with Resnet18 (no GPU)
class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalize the resized images as expected by resnet18
        # [0.485,0.456,0.406] -> normalized mean value of ImageNet,
        # [0.229,0.224,0.225] std of ImageNet
        self.normalize = transforms.Normalize(
            mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    def getVec(self,img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1,self.numberFeatures,1,1)
        def copyData(m,i,o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0,:,0,0]
    def getFeatureLayer(self):
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        return cnnModel,layer

# generate vectors for all the images in the set
img2vec = Img2VecResnet18()


# Converting images to feature vectors
# 'marine_12.png' -> 'marine_12.jpg'
# 'primate_03.png' -> 'primate_03.jpg'
# 'primate_04.png' -> 'primate_04.jpg'
allVectors = {}
for image in imgNameList:
    I = Image.open(os.path.join("inputImagesCNN",image))
    vec = img2vec.getVec(I)
    allVectors[image] = vec
    I.close()


def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T,v.T)/(
            (np.linalg.norm(v,axis=0).reshape(-1,1))*(
        (np.linalg.norm(v,axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim,columns=keys,index=keys)
    return matrix

similarityMatrix = getSimilarityMatrix(allVectors)
# similarityMatrix.to_csv(
#     os.path.join(rootPath,'simiMtrx.csv'),
#     mode='w',header=True,index=True)

# from numpy.testing import assert_almost_equal
# import pickle
# from tqdm import tqdm
# k = 5  # the number of top similar images to be stored
# similarNames = pd.DataFrame(index=similarityMatrix.index,columns=range(k))
# similarValues = pd.DataFrame(index=similarityMatrix.index,columns=range(k))
# for j in tqdm(range(similarityMatrix.shape[0])):
#     kSimilar = similarityMatrix.iloc[j,:].sort_values(ascending=False).head(k)
#     similarNames.iloc[j,:] = list(kSimilar.index)
#     similarValues.iloc[j,:] = kSimilar.values
#
# similarNames.to_pickle("similarNames.pkl")
# similarValues.to_pickle("similarValues.pkl")


# get similarity measure based on each trial
exp1b = pd.read_csv(
    os.path.join(rootPath,'exp1b_clean.csv'),sep=',')
exp2 = pd.read_csv(
    os.path.join(rootPath,'exp2_clean.csv'),sep=',')
exp1b_clean = exp1b[exp1b['acc']==1].copy(deep=True)
exp1b_clean.reset_index(drop=True,inplace=True)
exp2_clean = exp2[exp2['acc']==1].copy(deep=True)
exp2_clean.reset_index(drop=True,inplace=True)

sizeList = [1,2,4,8]
blockCate = ['Animals','Objects']

# exp.1b
exp1b_clean['BlockN'] = 1
exp1b_subj = list(set(exp1b_clean['subj'].tolist()))
subjs,blocks,targets = [],[],[]
for subj in exp1b_subj:
    h = 1
    for n in sizeList:
        for cate in blockCate:
            exp1b_clean.loc[
                (exp1b_clean['subj']==subj)&
                (exp1b_clean['block']==cate)&
                (exp1b_clean['setsize']==n),'BlockN'] = h

            targs = exp1b_clean.loc[
                (exp1b_clean['subj']==subj)&
                (exp1b_clean['trialType']=='target')&
                (exp1b_clean['block']==cate)&
                (exp1b_clean['setsize']==n),
                'imgName'].tolist()

            targets += targs
            blocks += [h]*len(targs)
            subjs += [subj]*len(targs)

            h += 1
targImg_1b = pd.DataFrame(
    {'subj':subjs,'BlockN':blocks,'target':targets})

for subj in exp1b_subj:
    for n in range(1,9):
        distrImgs = exp1b_clean.loc[
            (exp1b_clean['subj']==subj)&
            (exp1b_clean['BlockN']==n)&
            (exp1b_clean['trialType']=='distractor'),
            'imgName'].tolist()
        targImgs = list(set(
            targImg_1b.loc[
                (targImg_1b['subj']==subj)&
                (targImg_1b['BlockN']==n),'target'].tolist()))
        h = 1
        for targImg in targImgs:
            for distrImg in distrImgs:
                exp1b_clean.loc[
                    (exp1b_clean['subj']==subj)&
                    (exp1b_clean['BlockN']==n)&
                    (exp1b_clean['imgName']==distrImg),
                    'simi_val_%d'%h] = similarityMatrix.loc[
                    targImg,distrImg]
                exp1b_clean.loc[
                    (exp1b_clean['subj']==subj)&
                    (exp1b_clean['BlockN']==n)&
                    (exp1b_clean['imgName']==distrImg),
                    'targ_%d'%h] = targImg
            h += 1
exp1b_clean['simi_max'] = exp1b_clean[
    ['simi_val_1','simi_val_2','simi_val_3','simi_val_4',
     'simi_val_5','simi_val_6','simi_val_7','simi_val_8']].max(axis=1)
# exp1b_clean.to_csv(os.path.join(rootPath,'exp1b_simi.csv'),
#                    mode='w',header=True,index=False)

exp1b_distr = exp1b_clean[
    exp1b_clean['trialType']=='distractor'].copy(deep=True)
exp1b_distr['simi_mean'] = exp1b_distr[
    ['simi_val_1','simi_val_2',
     'simi_val_3','simi_val_4',
     'simi_val_5','simi_val_6',
     'simi_val_7','simi_val_8']].mean(axis=1)
exp1b_distr.reset_index(drop=True,inplace=True)
exp1b_distr_mean = exp1b_distr.groupby(
    ['subj','setsize','cond'])[[
    'rt','simi_mean']].agg(np.mean).reset_index()
# exp1b_distr_mean.to_csv(os.path.join(rootPath,'exp1b_simi_mean.csv'),
#                         mode='w',header=True,index=False)

# exp.2
for n in range(exp2_clean.shape[0]):
    for k in range(1,9):
        targImg = exp2_clean.loc[n,'imgName']
        distrImg = exp2_clean.loc[n,'imgName%d'%k]
        if isinstance(distrImg,float):
            continue
        exp2_clean.loc[n,'simi_val_%d'%k] = similarityMatrix.loc[
            targImg,distrImg]
exp2_clean['simi_max'] = exp2_clean[
    ['simi_val_1','simi_val_2','simi_val_3','simi_val_4',
     'simi_val_5','simi_val_6','simi_val_7','simi_val_8']].max(axis=1)
# exp2_clean.to_csv(os.path.join(rootPath,'exp2_simi.csv'),
#                   mode='w',header=True,index=False)

exp2_distr = exp2_clean[
    exp2_clean['trialType']=='distractor'].copy(deep=True)
exp2_distr['simi_mean'] = exp2_distr[
    ['simi_val_1','simi_val_2',
     'simi_val_3','simi_val_4',
     'simi_val_5','simi_val_6',
     'simi_val_7','simi_val_8']].mean(axis=1)
exp2_distr.reset_index(drop=True,inplace=True)
exp2_distr_mean = exp2_distr.groupby(
    ['subj','setsize','cond'])[[
    'rt','simi_mean']].agg(np.mean).reset_index()
# exp2_distr_mean.to_csv(os.path.join(rootPath,'exp2_simi_mean.csv'),
#                        mode='w',header=True,index=False)

# mean
mpl.rcParams.update({'font.size':20})
fig,ax = plt.subplots(
    2,2,sharex=True,figsize=(18,12))
ax = ax.ravel()
n = 0
for exp in [exp1b_distr_mean,exp2_distr_mean]:
    if n==0:
        leg_tag = True
    else:
        leg_tag = False
    sns.lineplot(data=exp,x='setsize',y='rt',
                 hue='cond',hue_order=['within','between'],
                 style='cond',markers=['^','o'],dashes=False,
                 palette='Blues',linewidth=2,markersize=10,
                 err_style="bars",errorbar=("se",1),legend=leg_tag,
                 ax=ax[0+n])
    ax[0+n].set_xticks([1,2,4,8],labels=[1,2,4,8])
    ax[0+n].set_xlabel(xlabel='Memory Set Size')
    ax[0+n].set_ylabel(ylabel='RT (sec)')
    ax[0+n].set_ylim(ymin=0.49,ymax=0.75)
    y_major_locator = MultipleLocator(0.1)
    ax[0+n].yaxis.set_major_locator(y_major_locator)

    sns.lineplot(data=exp,x='setsize',y='simi_mean',
                 hue='cond',hue_order=['within','between'],
                 style='cond',markers=['^','o'],dashes=False,
                 palette='Blues',linewidth=2,markersize=10,
                 err_style="bars",errorbar=("se",1),
                 legend=False,ax=ax[1+n])
    ax[1+n].set_ylabel(ylabel='Visual Similarity ')
    ax[1+n].set_ylim(ymin=0.51,ymax=0.58)
    y_major_locator = MultipleLocator(0.03)
    ax[1+n].yaxis.set_major_locator(y_major_locator)

    n += 2
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['Within-Category','Between-Category'],
             loc='upper left',ncol=2,fontsize=15,
             frameon=False).set_title(None)
sns.despine(offset=15,trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figPath,'simi_rt.png'))
plt.show(block=True)


# correlation
for k in ['within','between']:
    print('%s-category condition:'%k)
    for n in sizeList:
        x = exp1b_distr_mean.loc[
            (exp1b_distr_mean['setsize']==n)&
            (exp1b_distr_mean['cond']==k),'rt']
        y = exp1b_distr_mean.loc[
            (exp1b_distr_mean['setsize']==n)&
            (exp1b_distr_mean['cond']==k),'simi_mean']
        r,p = stats.pearsonr(x,y)
        print('MSS=%d'%n)
        print('r = %0.3f,p = %0.3f'%(r,p))


exp1b_distr_allmean = exp1b_distr.groupby(
    ['subj'])[['rt','simi_mean']].agg(np.mean).reset_index()
exp2_distr_allmean = exp2_distr.groupby(
    ['subj'])[['rt','simi_mean']].agg(np.mean).reset_index()
r,p = stats.pearsonr(exp1b_distr_allmean.rt,exp1b_distr_allmean.simi_mean)
print('r = %0.3f,p = %0.3f'%(r,p))
r,p = stats.pearsonr(exp2_distr_allmean.rt,exp2_distr_allmean.simi_mean)
print('r = %0.3f,p = %0.3f'%(r,p))


# trial-level


























