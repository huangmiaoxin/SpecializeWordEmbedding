import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy
import sys
import math
import time

class skip_gram_model(nn.Module):
    def __init__(self, emb_size, emb_dim):
        super(skip_gram_model, self).__init__()
        self.emb_size=emb_size #embedding size
        self.emb_dim=emb_dim #embedding dimension
        self.syn0=nn.Embedding(emb_size, emb_dim, sparse=True)
        self.syn1=nn.Embedding(emb_size, emb_dim, sparse=True)
        #self.activate_fun=F.logsigmoid
        #init weight data
        init_range = 0.5/self.emb_dim #refer to source code of word2vec
        self.syn0.weight.data.uniform_(-init_range, init_range)
        self.syn1.weight.data.uniform_(-0,0)
        
    def forward(self, center_word, context_word, neg_sampling_words):
        emb_center_w=self.syn0(center_word)
        emb_context_w=self.syn1(context_word)
        y=torch.mul(emb_center_w, emb_context_w).squeeze()
        y=torch.sum(y, dim=1)
        y=F.logsigmoid(y)
        emb_neg_sampling_ws=self.syn1(neg_sampling_words)
        neg_y=torch.bmm(emb_neg_sampling_ws, emb_center_w.unsqueeze(2)).squeeze()
        neg_y=F.logsigmoid(-1*neg_y)
        return -1*(torch.sum(y)+torch.sum(neg_y))
    
    def save_model(self, file_path, idx2word, use_cuda):
        fo=open(file_path, 'w')
        if use_cuda:
            model_result=self.syn0.weight.cpu().data.numpy()
        else:
            model_result=self.syn0.weight.data.numpy()
        fo.write('%d %d\n' % (len(idx2word), self.emb_dim))
        for idx, word in enumerate(idx2word):
            emb_result=model_result[idx]
            emb_result=' '.join([str(f) for f in emb_result])
            fo.write('%s %s\n' %(word, emb_result))
        fo.close()
       