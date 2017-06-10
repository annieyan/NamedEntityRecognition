from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import sklearn 
import string
from collections import deque
from itertools import islice
import collections
import math
import argparse
import time
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import matplotlib.pyplot as plt
import itertools
import sys
import scipy
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import *
import cPickle as pickle


# import utils 
from utils import *
from feature_generator import *



# GLOABLS
STOP_token = '_STOP_'
UNK_token = '_UNK_' 
START_token = '_START_'


class perceptron:

    def __init__(self,fea_gen,sentences,test_sentences,output_name1,output_name2):
        self.model = None
        
        self.iter_num = 50
        # feature generation object
        self.fea_gen = fea_gen
        self.netags_list = self.fea_gen.netags_list
        self.netags_list_len = len(self.netags_list)
        self.fea_vector_len = self.fea_gen.fea_vector_len

        with open(output_name1,'r') as fp:
            self.all_features_train_dict = pickle.load(fp)

        with open(output_name2,'r') as fp2:
            self.all_features_test_dict = pickle.load(fp2)

        self.sentences = sentences
        self.testing_sentences = test_sentences
        self.learned_weights = np.empty([1,self.fea_vector_len])

        self.testing_true_lables = list()
        self.testing_predict_lables = list()
  
        



    def uni_NEtagging(self):
        sentences = self.sentences
        netags_list = self.netags_list 
        count = 0
        # tag_fea_pair = dict()
        true_tag_fea_pair = dict()
        N = self.netags_list_len
        print("begin create_uni_feature for true values")
        print("number of sentences:",len(sentences))
        for i in xrange(0,len(sentences)):
            sent = sentences[i]
            # print sent
            sent_len = len(sent)
            true_lables = list()
            true_tag_index = list()
            for line in sent:
                true_lables.append(line[3])
                true_tag_index.append(netags_list.index(line[3]))

            print("calculating true_fea_vector for sentence:",i)
            true_fea_vector = self.create_uni_feature(i,true_tag_index)
            true_tag_fea_pair[i]=true_tag_index,true_fea_vector


        # random initialization of weight vector
        weights = np.random.rand(self.fea_vector_len)
        ave_weights = weights
        for i in xrange(0,self.iter_num):
            print('training iteration:',i)
            
            # iterate sentences 
            for j in xrange(0,len(sentences)):
                #print("loading sentence :",j)
                sent = sentences[j]
                sent_len = len(sent)
                estimated_lables = list()   
                t0 = time.time()
                tag_index = self.uni_tagging_decode(weights,j)
                t1 = time.time()
                #print("------uni_tagging_decode-------",t1-t0)
                true_tag_index = true_tag_fea_pair[j][0]
                true_fea_vector = true_tag_fea_pair[j][1]
                estimated_fea_vector = self.create_uni_feature(j,tag_index)
        
                t2 = time.time()
                # do weight update
                if (tag_index!= true_tag_index):
                    #print("update weights")
                    weights= weights.reshape((1,self.fea_vector_len))
                    new_weights = weights + true_fea_vector - estimated_fea_vector
                    # assert(new_weights == weights)
                    ave_weights = ave_weights + new_weights
                    weights = new_weights
                    count = count +1.0
                t3 = time.time()
                #print("---------update weight----------",t3-t2)
        
        ave_weights = ave_weights/count
        self.learned_weights = ave_weights
        return ave_weights



    '''
    for unary feature, get argmax tag sequence
    '''
    def uni_tagging_decode(self,weights,j):
        # print ('------begin uni_tagging_decode-------------')
        fea_matrix = self.all_features_train_dict[j]
        sent_len = len(self.sentences[j])
        weights_reshape = weights.transpose()
        score = np.empty((sent_len * self.netags_list_len))
        #---------------------------------------------------
        # estimated_tagseq_index = list()

        # # fea_matrix = np.reshape([sent_len,self.netags_list_len])
        # fea_vec_temp = scipy.sparse.csr_matrix((len(self.netags_list),self.fea_vector_len), dtype=np.int8)
        # for line in self.sentences[j]:  
        #     score2 = np.empty([self.netags_list_len])

        #     for i in xrange(0,len(self.netags_list)):
        #     # fea_vector in np sparse matirx
        #         # print((
        #         #     self.fea_gen.feature_vector_gen(line,self.netags_list[i])))
        #         fea_vec_temp[i,:]= scipy.sparse.csr_matrix(\
        #              self.fea_gen.feature_vector_gen(line,self.netags_list[i]))
                
        #         # print fea_vec_temp[i,:].shape
                
        #         # print weights.shape
        #     score2= fea_vec_temp.dot(weights_reshape)
        #     estimated_tagseq_index.append(np.argmax(score2))
        # print('estimated sequence from old code',estimated_tagseq_index)


        #-------------------------------------------
        score= fea_matrix.dot(weights_reshape)
        score_reshape = score.reshape([sent_len, self.netags_list_len])
        tag_index = np.argmax(score_reshape, axis = 1)
        tag_index_reshape = list(tag_index.flat)
        return tag_index_reshape




    '''
    given a sentence, return PHI
    for unary feature only
    '''
    def create_uni_feature(self,i,tag_index):
     
        sent = self.sentences[i]
        fea_matrix = self.all_features_train_dict[i]
        fea_vector = scipy.sparse.csr_matrix((1, self.fea_vector_len), dtype=np.int8)
        sent_len = len(sent)
        for k in xrange(0,sent_len):
            temp_idx = k*self.netags_list_len+tag_index[k]
            fea_vec_temp= fea_matrix[temp_idx,:]
            fea_vector = fea_vector + fea_vec_temp
        return fea_vector



    def perceptron_predict(self):
        print("begin predicting")
        
        sentences = self.testing_sentences
        weights = self.learned_weights
        netags_list = self.netags_list       
        N = len(netags_list)
        print("begin create_uni_feature for true values")
        print("number of sentences:",len(sentences))
        for i in xrange(0,len(sentences)):
            sent = sentences[i]
            sent_len = len(sent)

            true_lables = list()
            estimated_lables = list()
            true_tag_index = list()

            for line in sent:
                true_lables.append(line[3])
                true_tag_index.append(netags_list.index(line[3]))
            estimated_tag_list = self.sent_predict(weights,i)

            self.testing_true_lables.append(true_tag_index)
            self.testing_predict_lables.append(estimated_tag_list)




    '''
        take learned weights, testing data, and predict
        for unary feature, get argmax tag sequence
    '''
    def sent_predict(self,learned_weights,sent_id):
        # print ('------begin uni_tagging_decode-------------')
        weights_reshape = learned_weights.transpose()   
        sent = self.testing_sentences[sent_id]
        fea_matrix = self.all_features_test_dict[sent_id]
        sent_len = len(sent)    
        # for line in sent:
        # score = np.empty((sent_len * self.netags_list_len))
        # fea_matrix = np.reshape([sent_len,self.netags_list_len])
        # fea_vec_temp = scipy.sparse.csr_matrix((len(self.netags_list),self.fea_vector_len), dtype=np.int8)
            
            # for i in xrange(0,len(self.netags_list)):
            # fea_vector in np sparse matirx
                # print(len(
                    # self.fea_gen.feature_vector_gen(line,self.netags_list[i])))
                # fea_vec_temp[i,:]= scipy.sparse.csr_matrix(\
                    # self.fea_gen.feature_vector_gen(line,self.netags_list[i]))
                
                # print fea_vec_temp[i,:].shape
                # print weights.shape
        score= fea_matrix.dot(weights_reshape)
        score_reshape = score.reshape([sent_len,self.netags_list_len])
        tag_index = np.argmax(score_reshape, axis = 1)
        tag_index_reshape = list(tag_index.flat)
        return tag_index_reshape




    def result_report(self):

        y_true = [item for sublist in self.testing_true_lables for item in sublist]
        y_pred = [item for sublist in self.testing_predict_lables for item in sublist]
        target_names = self.netags_list
        accuracy = accuracy_score(y_true, y_pred)
        print("number of iteration:",self.iter_num)
        print("NE tags",self.netags_list)
        print("number of tokens:",len(y_true))
        print("number of features:",self.fea_vector_len)
        print ("accuracy:",accuracy)
        print(classification_report(y_true, y_pred, target_names=target_names))
        with open ('output','w+') as file:
            file.write("number of iteration: %d" % self.iter_num)
            file.write("number of accuracy: %f" % accuracy)

 


    '''
    given a line, generate Big phi (bigram case)
    PHI(xi,y) = sum over k in netag_list (xi,k,yk,yk-1)
    return PHI as summed feature vector of a given line (word, pos, chunk,ne)
    '''
    def create_bigram_feature(self,line):
        netags_list = self.netags_list 
        N = len(netags_list)
        fea_vector = scipy.sparse.csr_matrix((1, self.fea_vector_len), dtype=np.int8)
        
        # empty_line = ""
        # sent_len = len(sent)
        # START of the sentence will be tageed as 'O'. all others will be 0
        # for n in xrange(0,N):
        #     first_bi = START_token, netags_list[n]

        #     fea_vec1= self.fea_gen.feature_vector_gen(empty_line,netags_list[n])
        #     fea_vec2 = self.fea_gen.feature_vector_gen(sent[0],netags_list[n])
        #     tag_fea_pair[first_bi] = fea_vec1.extend(fea_vec2)

        #     last_bi = netags_list[n],STOP_token
        #     fea_vec1= self.fea_gen.feature_vector_gen(sent[sent_len-1],netags_list[n])
        #     fea_vec2 = self.fea_gen.feature_vector_gen(empty_line,netags_list[n])
        #     tag_fea_pair[last_bi] = fea_vec1.extend(fea_vec2)
        for i in xrange(0,N):
            for j in xrange(0,N):
                fea_vec1= self.fea_gen.feature_vector_gen(line,netags_list[i])
                fea_vec2 = self.fea_gen.feature_vector_gen(line,netags_list[j])
                fea_vec1.extend(fea_vec2)
                fea_vector = fea_vector + scipy.sparse.csr_matrix(fea_vec1)                
        return fea_vector
                


  


    '''
    viterbi for bigram
    input: NE tag space of lenth N: vocabulary_tag
          sentences of lenth T: list of tokens
          weights: weights from last round.   np.array
          fea_matrix: feature matrix from any combination of bigram NEtags
    return: best path(best tag sequence, or estimation tages) 
           in list
    '''
    # def viterbi_bi(self,sent,weights):
    #     # probability matrix viterbi[N+1,T], including stop
    #     # backpointer path matric[N+1,T]
    #     # T observations means sent length
        
    #     weights = weights
    #     T = len(sent)
    #     # N states/tags not including stop and start     
    #     N = len(self.netags_list)
    #     # transition prob
    #     weights = weights
    #     prob_mat = np.empty((N+1,T))
    #     #path_mat = np.empty((N+1,T))
    #     netags_list = self.netags_list
    #     # a dict of strings of tags
    #     path= {}
    #     predicted_tags = list()
    #     # initialization, base case
    #     # assume bigram first, ignore start
    #     # trans
    #     for i in xrange(0,N):
    #         # from start to the first tag * emission from first word 
    #         # conditions on first tag
    #         prob_mat[i][0] = trellis[0][i]*self.emit_prob[sent_tokens[0],netags_list[i]]
    #         path[tag_space[i]] = tag_space[i]
    #     for t in xrange(1,T):
    #         newpath = {}
    #         # iterate through states
    #         for n in xrange(0,N):
    #             # previous transition tag n0
    #             prob_mat[n][t], state=max(((prob_mat[n0][t-1])*trellis[n0+1][n]* self.emit_prob[sent_tokens[t],netags_list[n]],n0) for n0 in xrange(0,N))
    #             #path_mat[n][t]= tag_space[n0]
    #             #print('T',t,'N',n)
    #             #print(prob_mat[n][t])
    #             newpath[tag_space[n]] = path[tag_space[state]]+tag_space[n]
    #         path = newpath                
    #     # terminaiton prob_mat[N][T-1] = STOP token:last observation
    #     prob_mat[N][T-1],state = max(((prob_mat[n0][T-1]*trellis[N+1][n0]),n0)for n0 in xrange(0,N))
    #     #final_path = path[tag_space[state]]
    #     #path_mat[N][T-1] = 
    #     predicted_tags = list(path[netags_list[state]])
    #     #print(predicted_tags)
    #     return predicted_tags


