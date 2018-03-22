import json
import numpy as np
from scipy import stats
import sys

def relatedness_dataset(f_path):
    #MEN_dataset_natural_form_full dataset
    fi=open(f_path)#MEN_dataset_natural_form_full
    #
    MEN_pairs=[]
    line_cnt=0
    for line in open(f_path):
        if line_cnt==0:
            fi.readline()#skip first line
            line_cnt+=1
            continue
        tmp=fi.readline().split(' ')
        line_cnt+=1
        #print(line_cnt, tmp)
        pair=[tmp[0], tmp[1], float(tmp[2])]
        MEN_pairs.append(pair)
    fi.close()
    
    return MEN_pairs

def similarity_dataset(f_path):
    #get SimLex999 dataset
    fi=open(f_path)#SimLex-999.txt
    #
    SimLex_pairs=[]
    line_cnt=0
    for line in open(f_path):
        if line_cnt==0:
            fi.readline()#skip first line
            line_cnt+=1
            continue
        tmp=fi.readline().split('\t')
        line_cnt+=1
        #print(line_cnt, tmp)
        pair=[tmp[0], tmp[1], float(tmp[3])]
        SimLex_pairs.append(pair)
    fi.close()
    
    return SimLex_pairs

def load_embedding(f_path):
    #get raw embedding from model
    fi=open(f_path)
    embed_dict={}
    line_cnt=0
    for line in open(f_path):
        if line_cnt==0:
            word_embedding=fi.readline()
            line_cnt+=1
            continue
        word_embedding=fi.readline().split(' ')
        word=word_embedding[0]
        embedding=np.array([float(f) for f in word_embedding[1:]])
        #
        embed_dict[word]=embedding
   
        line_cnt+=1
    fi.close()
    
    return embed_dict

def word_cosine(word1,word2,dict):
    #transform word to vector, and then compute cosine of included angle between vectors
    v1=dict[word1]#vector 1 of word 1
    v2=dict[word2]#vector 2 of word 2
    return v1.dot(v2)/np.sqrt(v1.dot(v1))/np.sqrt(v2.dot(v2))

def compute_pearson_coeff(embedding_dict, pairs):
    predicts=[]
    labels=[]
    for pair in pairs:
        w1=pair[0]#word 1
        w2=pair[1]#word 2
        label=pair[2]#relatedness strength
        if w1 in embedding_dict and w2 in embedding_dict:
            predicts.append(word_cosine(w1, w2, embedding_dict))
            labels.append(label)
    pearson_coeff, junk=stats.pearsonr(predicts, labels)
    return pearson_coeff
        
if __name__=='__main__':
    MEN_pairs=relatedness_dataset('./for_evaluation/MEN_dataset_natural_form_full')
    SimLex_pairs=similarity_dataset('./for_evaluation/SimLex-999.txt')
    raw_embedding_dict=load_embedding(sys.argv[1])
    relatedness_embedding_dict=load_embedding(sys.argv[2])
    similarity_embedding_dict=load_embedding(sys.argv[3])
    
    #relatedness
    raw_MEN_coeff=compute_pearson_coeff(raw_embedding_dict, MEN_pairs)
    relatedness_MEN_coeff=compute_pearson_coeff(relatedness_embedding_dict, MEN_pairs)
    #similarity
    raw_SimLex_coeff=compute_pearson_coeff(raw_embedding_dict, SimLex_pairs)
    similarity_SimLex_coeff=compute_pearson_coeff(similarity_embedding_dict, SimLex_pairs)
    
    print('relatedness evaluation: raw embedding(%.3f) vs specialize for relatedness embedding(%.3f)'%(raw_MEN_coeff, relatedness_MEN_coeff))
    print('similarity evaluation: raw embedding(%.3f) vs specialize for similarity embedding(%.3f)'%(raw_SimLex_coeff, similarity_SimLex_coeff))
    