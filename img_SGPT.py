#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 1b+2
# 2024.02.22


import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine

# Get our models - The package will take care of downloading the models automatically
# For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
tokenizer = AutoTokenizer.from_pretrained('Muennighoff/SGPT-125M-weightedmean-nli-bitfit')
model = AutoModel.from_pretrained('Muennighoff/SGPT-125M-weightedmean-nli-bitfit')
# Deactivate Dropout (There is no dropout in the above models so it makes
# no difference here but other SGPT models may have dropout)
model.eval()

filepath = os.path.join(os.getcwd(),'res_sgpt')
data_file = pd.read_csv(os.path.join(filepath,'stim_full_list.csv'))
img_list = data_file['image'].to_list()

# Tokenize input texts
texts = list(set(data_file['name'].to_list()))
batch_tokens = tokenizer(texts,padding=True,truncation=True,return_tensors='pt')

# Get the embeddings
with torch.no_grad():
    # Get hidden state of shape [bs, seq_len, hid_dim]
    last_hidden_state = model(
        **batch_tokens,output_hidden_states=True,return_dict=True).last_hidden_state

# Get weights of shape [bs, seq_len, hid_dim]
weights = (
    torch.arange(start=1,end=last_hidden_state.shape[1]+1)
    .unsqueeze(0)
    .unsqueeze(-1)
    .expand(last_hidden_state.size())
    .float().to(last_hidden_state.device)
)

# Get attn mask of shape [bs, seq_len, hid_dim]
input_mask_expanded = (
    batch_tokens['attention_mask']
    .unsqueeze(-1)
    .expand(last_hidden_state.size())
    .float()
)

# Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
sum_embeddings = torch.sum(last_hidden_state*input_mask_expanded*weights,dim=1)
sum_mask = torch.sum(input_mask_expanded*weights,dim=1)

embeddings = sum_embeddings/sum_mask

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
for row_img in img_list:
    for col_img in img_list:
        row_name = data_file.loc[data_file['image']==row_img,'name'].values[0]
        col_name = data_file.loc[data_file['image']==col_img,'name'].values[0]
        row_indx = texts.index(row_name)
        col_indx = texts.index(col_name)

        cosine_simi = 1-cosine(embeddings[row_indx],embeddings[col_indx])

        data_file.loc[data_file['image']==row_img,col_img] = cosine_simi
data_file.to_csv(os.path.join(filepath,'img_sgpt_simi.csv'),
                 sep=',',mode='w',header=True,index=False)



# cosine_sim_0_1 = 1-cosine(embeddings[0],embeddings[1])
# cosine_sim_0_2 = 1-cosine(embeddings[0],embeddings[2])
# cosine_sim_0_3 = 1-cosine(embeddings[0],embeddings[3])
#
# print("Cosine similarity between \"%s\" and \"%s\" is: %.3f"%(texts[0],texts[1],cosine_sim_0_1))
# print("Cosine similarity between \"%s\" and \"%s\" is: %.3f"%(texts[0],texts[2],cosine_sim_0_2))
# print("Cosine similarity between \"%s\" and \"%s\" is: %.3f"%(texts[0],texts[3],cosine_sim_0_3))