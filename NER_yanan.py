# for nlp hw4, yanan, 2017, March
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
import random

# import utils 
from utils import *
from feature_generator import *

from perceptron import *

UNK_Token = '_UNK_'

class NER:


    def __init__(self,args):
        # input
        self.training_set = args.training_set
        self.dev_set = args.dev_set
        self.unk_threshold = args.threshold


        self.oovcount = 0
        self.oovs = dict()
        self.words = dict()
        self.postags = dict()
        self.chunktags = dict()
        self.netags = dict()

        # self.features_freq = dict()
        self.fea_generator =  None

        # list of NE tags in trainning data
        self.netags_list = list()


        self.sentences = list()
        self.test_sentences = list()
        
        self.all_features_train_dict= dict()
        self.all_features_test_dict = dict()



    def text_processing(self,filename):
        text = ""
        with open(filename,'r+') as f:
            text = f.read()
        return text
        
            
    def load_sentences(self,path):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
        Sentences are a set of tuples
        One sentence is a tuple of tuple of (word, pos, chunk,NER)
        """

        sentences = list()
        sentence = []
        with open(path,'r+') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if len(sentence) > 0:
                        if 'DOCSTART' not in sentence[0][0]:
                            # sentences.add(tuple(sentence))
                            sentences.append(tuple(sentence))
                        sentence = []
                else:
                    word = line.split()
                    assert len(word) >= 2
                    sentence.append(tuple(word))           
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(tuple(sentence))
        return sentences


    def get_sent_of_tokens(self,path):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
        Sentences are a set of tuples
        One sentence is a tuple Token object
        """
        fg = self.fea_generator
        sentences = set()
        sentence = []
        with open(path,'r+') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if len(sentence) > 0:
                        if ('DOCSTART' != sentence[0].get_word):
                        
                            sentences.add(tuple(sentence))
                        sentence = []
                else:
                    word = line.split()
                    assert len(word) >= 2
                    token = Token(line)
                    token.set_feature_vectors(fg.feature_vector_gen(line))
                    sentence.append(token)           
            if len(sentence) > 0:
                if ('DOCSTART' != sentence[0].get_word):
                    sentences.add(tuple(sentence))
        return sentences




    def get_dictionary(self,file_name):
        data_file = open(file_name, "r")
        rawwords={}
            
        for line in data_file:
            line = line.strip()
            if line != "" and not line.startswith("-DOCSTART-"):

                w = re.split("\s",line)

                w0=norm_digits(clean(w[0]).lower())
                rawwords[w0]=rawwords.setdefault(w0,0)+1
                w1=escape(w[1])
                self.postags[w1]=self.postags.setdefault(w1,0)+1
                # w2=w[2].replace('-','_')
                w2=w[2]
                self.chunktags[w2]=self.chunktags.setdefault(w2,0)+1
                # w3=w[3].replace('-','_')
                w3=w[3]
                self.netags[w3]=self.netags.setdefault(w3,0)+1
        self.oovcount=0
        for w,c in rawwords.iteritems():
            if c>= self.unk_threshold:
                self.words[w]=c
            else:
                self.oovs[w]=c
                self.oovcount+=1
        print "Constructed dictionary:"
        #print rawwords
        #print words
        print str(len(self.words))+" items and "+str(self.oovcount)+" oovs "
        return

    


    def initNER(self):
        # ori_train_set = self.load_sentences(self.training_set)
        # print ori_train_set
        # self.get_dictionary(self.training_set)
        # line_eg = "Sheep NNP I-NP O"
        # feature generator got from training data   
        # fea_vector_eg = self.fea_generator.feature_vector_gen(line_eg,'O')
       

        self.sentences = self.load_sentences(self.training_set)
        self.test_sentences = self.load_sentences(self.dev_set)
        self.fea_generator = fea_gen(self.training_set,self.unk_threshold,self.sentences,self.test_sentences)
        # token_eg = Token(line_eg)
        self.netags_list = list(self.fea_generator.netags.keys())
        print self.netags_list 
        self.all_features_train_dict  = self.fea_generator.compute_feature_matrix(self.sentences)
        self.all_features_test_dict  = self.fea_generator.compute_feature_matrix(self.test_sentences)

        print self.fea_generator.feature_name_list

        ran_n = random.randint(1, 1000)  
        output_name1 = 'all_feature_dict_'+str(ran_n)+self.training_set+'.p'
        output_name2 = 'all_feature_dict_'+str(ran_n)+self.dev_set+'.p'
        with open(output_name1,'wb') as fp:
            pickle.dump(self.all_features_train_dict,fp)

        with open(output_name2,'wb') as fp2:
            pickle.dump(self.all_features_test_dict,fp2)
       
        # do perceptron learn
        print ("begin training")
        model = perceptron(self.fea_generator,self.sentences,self.test_sentences,output_name1,output_name2)
        learned_w = model.uni_NEtagging()
        # print ("learned weights:")
        # # print learned_w
        # np.savetxt('learned_weights',learned_w,delimiter=',')
        print("begin inference")
        model.perceptron_predict()
        model.result_report()     
        return None




def main():
    args = get_args()
    ner = NER(args)
    ner.initNER()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_set",action = "store",
    help = "training set")

    parser.add_argument("dev_set", action="store",
                        help="either dev data or test data. dev for tunning, test can be used only once")
    parser.usage = ("NER_yanan.py [-h] [-n N] training_set dev_set")
    parser.add_argument("-t", "--threshold", action="store", type=int,
                        default=5, metavar='T',
                        help="threshold value for words to be UNK.")

    return parser.parse_args()

if __name__ == "__main__":
    main()


