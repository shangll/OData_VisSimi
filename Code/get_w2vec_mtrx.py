#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2:
# AlexNet
# 2023.11.17
# linlin.shang@donders.ru.nl



import gensim.downloader as api
from gensim.models import Word2Vec

word_vectors = api.load('glove-wiki-gigaword-300')
# word_vectors = api.load('word2vec-google-news-300')
# word_vectors = api.load('fasttext-wiki-news-subwords-300')
# wiki_corpus = api.load('wiki-english-20171001')
# word_vectors = Word2Vec(wiki_corpus)


from config import set_filepath,figPath
import os
import pandas as pd

filepath = set_filepath(os.getcwd(),'res_w2v')
data_file = pd.read_csv(os.path.join(filepath,'stim_full_list.csv'))
img_list = data_file['image'].tolist()
word_list = list(set(data_file['name'].tolist()))

# miss_words = []
# for word in word_list:
#     if word not in word_vectors.index_to_key:
#         miss_words.append(word)

for row_img in img_list:
    for col_img in img_list:
        row_name = data_file.loc[
            data_file['image']==row_img,'name'].values[0]
        col_name = data_file.loc[
            data_file['image']==col_img,'name'].values[0]

        cosine_simi = word_vectors.similarity(row_name,col_name)

        data_file.loc[data_file['image']==row_img,col_img] = cosine_simi

data_file.to_csv(os.path.join(filepath,'img_w2v_simi.csv'),
                 sep=',',mode='w',header=True,index=False)
# data_file = pd.read_csv(os.path.join(filepath,'img_w2v_simi.csv'))
#
import copy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

corr_words = copy.copy(word_list)
# for word in miss_words:
#     corr_words.remove(word)

corr_mtrx = pd.DataFrame(columns=corr_words,index=corr_words)
for rname in corr_words:
    for cname in corr_words:
        corr_mtrx.loc[rname,cname] = word_vectors.similarity(rname,cname)
corr_mtrx = corr_mtrx.astype('float32')
mpl.rcParams.update({'font.size':8})
plt.figure(figsize=(18,15))
sns.heatmap(
    data=corr_mtrx,linewidth=.2,cmap='RdBu_r',
    xticklabels=corr_words,yticklabels=corr_words)
plt.tight_layout()
plt.savefig(os.path.join(figPath,'w2v_heatmap_glov.tif'))
plt.close('all')

#
anim = list(
    set(data_file.loc[
            data_file['cate']=='Animals','name'].tolist()))
obj = list(
    set(data_file.loc[
            data_file['cate']=='Objects','name'].tolist()))
# for word in miss_words:
#     if word in anim:
#         anim.remove(word)
#     else:
#         obj.remove(word)
anim_vec = word_vectors[anim]
obj_vec = word_vectors[obj]
from sklearn.manifold import TSNE
import numpy as np
random_state=0
def plotTsne2D(wVectors1,wVectors2,wlist1,wlist2):
    clist = ['tomato','dodgerblue']
    tsne = TSNE(n_components=2,n_iter=10000,perplexity=20)

    np.set_printoptions(suppress=True)
    T1 = tsne.fit_transform(wVectors1)
    T2 = tsne.fit_transform(wVectors2)

    plt.figure(figsize=(16,9))
    plt.scatter(T1[:,0],T1[:,1],marker='o',c=clist[0])
    plt.scatter(T2[:,0],T2[:,1],marker='x',c=clist[1])

    for label,x,y in zip(wlist1,T1[:,0],T1[:,1]):
        plt.annotate(
            label,xy=(x,y),xytext=(0,0),textcoords='offset points')
    for label,x,y in zip(wlist2,T2[:,0],T2[:,1]):
        plt.annotate(
            label,xy=(x,y),xytext=(0,0),textcoords='offset points')
    # for xy_loc,list_name in zip([wVectors1,wVectors2],[wlist1,wlist2]):
    #     for n in range(len(list_name)):
    #         plt.annotate(
    #             list_name[n],xy=(xy_loc[n,0],xy_loc[n,1]),
    #             xytext=(
    #             xy_loc[n,0],xy_loc[n,1]),textcoords='offset points')
    sns.despine(offset=10,trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figPath,'w2v_tsne_glov.tif'))
    plt.show(block=True)
    plt.close('all')

# data_file = pd.read_csv(os.path.join(filepath,'img_w2v_simi.csv'))
plotTsne2D(anim_vec,obj_vec,anim,obj)



#
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(anim_vec)
anim_vec = pca.transform(anim_vec)

pca.fit(obj_vec)
obj_vec = pca.transform(obj_vec)

plt.figure(figsize=(21,12))
plt.scatter(anim_vec[:,0],anim_vec[:,1],marker='o')
plt.scatter(obj_vec[:,0],obj_vec[:,1],marker='x')

for xy_loc,list_name in zip([anim_vec,obj_vec],[anim,obj]):
    for n in range(len(list_name)):
        plt.annotate(
            list_name[n],xy=(xy_loc[n,0],xy_loc[n,1]),
            xytext=(-10,10),textcoords='offset points')
sns.despine(offset=10,trim=True)
plt.tight_layout()
plt.savefig(os.path.join(figPath,'w2v_pca_glov.tif'))
plt.show(block=True)
plt.close('all')
#

