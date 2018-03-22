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
from skip_gram_model import skip_gram_model
from Vocab import Vocab

numpy.random.seed(2333)

def train(vocab, #
          embedding_dimention, 
          batch_size, 
          window_size, 
          neg_sample_num, 
          use_cuda, 
          learning_rate, 
          iteration_num, 
          fi_path, 
          fo_path,
          thesaurus_dict={}
         ):
    skip_gram=skip_gram_model(len(vocab.idx2word), embedding_dimention)
    if use_cuda:
        skip_gram.cuda()
    optimizer = optim.SGD(skip_gram.parameters(), lr=learning_rate)
    
    
    estimated_all_pairs_count=iteration_num*vocab.all_words_count*2*window_size-vocab.all_lines_count*2*window_size
    estimated_all_batches_count=int(estimated_all_pairs_count/batch_size)
    batches_count=0
    early_stop_flag=0
    
    for iter in range(iteration_num):
        fi=open(fi_path)
        pairs=[]
        for line in fi:
            line=line.strip().split(' ')
            #word 2 index
            words_idx=[vocab.word2idx[w.lower()] if w.lower() in vocab.word2idx else vocab.word2idx['unk'] for w in line]

            '''            
            for i, w in enumerate(line):
                try:
                    words_idx.append(vocab.word2idx[w])
                except:
                    print(i,w)
         '''   
            for i, center_word in enumerate(words_idx):
                pairs_len=len(pairs)
                #if center word in thesaurus, append pairs list
                if vocab.idx2word[center_word] in thesaurus_dict:
                    words_list=thesaurus_dict[vocab.idx2word[center_word]]
                    for word in words_list:
                        if word in vocab.word2idx:
                            pairs.append((center_word, vocab.word2idx[word]))
                    estimated_all_pairs_count+=len(pairs)-pairs_len
                    estimated_all_batches_count=int((estimated_all_pairs_count)/batch_size)
                #
                for j, context_word in enumerate(
                    words_idx[max(i-window_size,0):min(i+window_size,len(words_idx))]):
                    if i==j:
                        continue
                    pairs.append((center_word, context_word))
                    #enough word pairs to train
                    if len(pairs)>batch_size:#enough data for training
                        #train
                        pairs_batch=pairs
                        #for pair in pairs:
                        #    pairs_batch.append(pairs.popleft())
                        
                        center_words=[pair[0] for pair in pairs_batch]
                        context_words=[pair[1] for pair in pairs_batch]
                        neg_sampling_words=vocab.get_neg_sampling_result(pairs_batch, neg_sample_num)
                        #variable
                        center_words=Variable(torch.LongTensor(center_words))
                        context_words=Variable(torch.LongTensor(context_words))
                        neg_sampling_words=Variable(torch.LongTensor(neg_sampling_words))
                        if use_cuda:
                            center_words=center_words.cuda()
                            context_words=context_words.cuda()
                            neg_sampling_words=neg_sampling_words.cuda()
                            
                        optimizer.zero_grad()
                        loss=skip_gram.forward(center_words, context_words, neg_sampling_words)
                        loss.backward()
                        optimizer.step()

                        pairs=[]; #empty pairs list
                        
                        batches_count+=1
                        '''
                        # change learning rate gradually
                        if (batches_count*batch_size)%100000==0:
                            lr=learning_rate*(1.0-0.9*batches_count/estimated_all_batches_count)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        '''        
                        # change learning rate gradually, print processing progress per 1/100
                        if batches_count%int(estimated_all_batches_count/100)==0:
                            lr=max(learning_rate*0.2, 
                                   learning_rate*(1.0-0.9*batches_count/estimated_all_batches_count))
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                            print(batches_count,'/',estimated_all_batches_count, 
                                  ':%d%%' %(int(100*batches_count/estimated_all_batches_count)), 
                                  'lr: %.3f'%(lr), 'loss: %.3f'%(loss.data[0]))
                        #
        fi.close()
    #save
    skip_gram.save_model(fo_path, vocab.idx2word, use_cuda)
    #return 
    if use_cuda:
        model_result=skip_gram.syn0.weight.cpu().data.numpy()
    else:
        model_result=skip_gram.syn0.weight.data.numpy()
    return model_result

#main
if __name__=='__main__':
    if len(sys.argv)!=3 and len(sys.argv)!=4:
        print('command format: python specializing_word_embedding.py [iuput_corpus_file] [output_embedding_file] [thesaurus_file]')
        
    t1=time.time()#for compute running time
    use_cuda=torch.cuda.is_available()
    fi_path=sys.argv[1]
    fo_path=sys.argv[2]
    vocab=Vocab(fi_path, filter_count=4)
    if len(sys.argv)==4:
        thesaurus_path=sys.argv[3]
        thesaurus_dict=json.loads(open(thesaurus_path,'r').read())
        model_result=train(vocab, 100, 512, 5, 5, use_cuda, 0.02, 1, fi_path, fo_path, thesaurus_dict)#training
    if len(sys.argv)==3:
        model_result=train(vocab, 100, 512, 5, 5, use_cuda, 0.02, 1, fi_path, fo_path)
        
    print('running time: %3.2f mins'%((time.time()-t1)/60))#print running time