import json
import numpy
import sys
import math
import time


numpy.random.seed(2333)

class Vocab:
    def __init__(self, file_path, filter_count):
        self.vocab_file=file_path
        self.all_words_count=0
        self.all_lines_count=0
        #
        self.build_vocab(filter_count)
        self.init_neg_sample_tabel()
        
        
    def build_vocab(self, filter_count):
        fi=open(self.vocab_file)
        word_count={'unk':filter_count+1}
        for line in open(self.vocab_file):
            sent=fi.readline()
            sent=sent.split(' ')
            self.all_lines_count+=1
            for w in sent:
                self.all_words_count+=1
                w=w.lower()
                if w in word_count:
                    word_count[w]+=1
                else:
                    word_count[w]=1
        fi.close()
        #filter
        tmp_word_count=word_count.copy()
        for w in tmp_word_count:
            if word_count[w]<=filter_count:
                word_count.pop(w)
        #vocab
        word2idx={}#unknow
        idx2word=[]
        for w in word_count:
            word2idx[w]=len(idx2word)
            idx2word.append(w)
        #
        self.word_count=word_count
        self.word2idx=word2idx
        self.idx2word=idx2word

        print('build vocabulary with ', len(self.word2idx), ' words')
        
    def init_neg_sample_tabel(self):
        table_size=int(1e8)
        neg_sample_table=numpy.zeros(table_size, dtype=numpy.uint32)
        
        vocab_size=len(self.idx2word)
        power=0.75
        for_norm=sum([math.pow(self.word_count[w],power) for w in self.word_count])
        
        print('initializing negative sample tabel...')
        pro_sum=0 #sum of probability
        i=0
        for w in self.word2idx:
            pro_sum+=float(math.pow(self.word_count[w],power))/for_norm
            while i<table_size and float(i)/table_size<pro_sum:
                neg_sample_table[i]=self.word2idx[w]
                i+=1
        #
        self.neg_sample_table=neg_sample_table
        
        print('initialized!')
    #
    def get_neg_sampling_result(self, center_words, count):
        neg_result=numpy.random.choice(self.neg_sample_table, size=(len(center_words), count)).tolist()
        return neg_result