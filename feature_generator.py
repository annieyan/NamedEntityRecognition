"""

Generate feature space given a training set
Discard features appeared less than 5 times


see: https://github.com/aleju/ner-crf 
     https://github.com/pprett/nut/blob/master/nut/ner/features/en_best_v1.py

- simple features:

Whether to word starts with an uppercase letter
Whether the word is a single capital letter,i.e. A?
Whether the word consists only of capital letters. i.e. AASDSA
Whether the word contains any digit (0-9)
Whether the word contains any punctuation, i.e. . , : ; ( ) [ ] ? !
Whether the word contains only digits
Whether The word contains a hyphen
Whether the word contains an appostophy
Whether the word contains only punctuation


- complex features
The word shape of the word, e.g. XxxxX
The short word shape. E.G. x, X, Xx+
The length of the token
The unigram  of the word, a.k.a identity of the word, lowercased, normalized
The 1-4 prefix of the word, i.e. John becomes Joh.
The 1-4 suffix of the word, i.e. John becomes ohn.
The Part of Speech tag (POS) of the word 
The chunktag of the word

- external features
The https://code.google.com/p/word2vec/ cluster of the word (add -classes flag to the word2vec tool)
The https://github.com/percyliang/brown-cluster of the word
The brown cluster bitchain of the word (i.e. the position of the word's brown cluster in the tree of all brown clusters represented by a string of 1 and 0)
Whether the word is contained in a Gazetteer of person names. 
    The Gazetteer is created by scanning through an annotated corpus and collecting all names (words labeled with PER) that appear more often among 
    the person names than among all words.
The LDA topic (among 100 topics) of a small window (-5, +5 words) around the word. 



"""
from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
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
import cPickle as pickle

from utils import *

# globals
UNK_Token = '_UNK_'

#Regular expressions that express some features of words
#Is the word capitalized?
capitalized = re.compile("[A-Z][a-z]+")
#Is the word a single capital letter?
caplet = re.compile("[A-Z]$")
#The word consists only of capital letters
allcaps = re.compile("[A-Z]+$")
#Tries to capture e.g. McLovin
caps = re.compile("[A-Z]+[a-z]+[A-Z]+[a-z]")
#The word contains a number
contains_num = re.compile("\d+")
#The word contains a hyphen
contains_hyp = re.compile("\-")
#The word contains an appostophy
contains_appos = re.compile("\'")
#The word contains a period
contains_per = re.compile("[A-Za-z]\.$")
#The word is a number
is_num = re.compile("[+-]?\d+(\,\d+)?\.?\d*$")
#The word is an integer number
is_int = re.compile("\d+$")
#The word ends in s
ends_s = re.compile("[A-Za-z]+s$")
#The word ends in ing
ends_ing = re.compile("[A-Za-z]+ing$")
#The word ends in ed
ends_ed = re.compile("[A-Za-z]+ed$")

feature_suffix = ['_cur','_pre','_post']


# all of them have an associated NE Tag suffix: "_XXX"
simple_features=["shape_capitalized","shape_caplet","shape_allcaps",
"shape_caps","shape_containsnum",
		"shape_containshyp","shape_containsappos",
        "shape_containsper","shape_isnum",
		"shape_isint","shape_ends","shape_ended","shape_ending"]

# unigram=xxx_B-LOC, shape=xxx, shapeshort=xxx, tokenlen=x, 
# suff1=x, suff2 = xx, suff3=xxx, pref1=x, pref2=xx, pref3=xxx,
# postag=xx, chunktag=xx


# sliding window
window_size = 1
window = range(-window_size,window_size+1)

'''
take training data as input, generate {feature:frequecy}
features are string values. i.e. 'unigram=one_dash_day', 'chunktag=I-NP',
'''
class fea_gen(object):
    def __init__(self,filename,unkthreshold,sentences,test_sentences):
        self.filename = filename
        self.freq_limit = unkthreshold
        self.testing_sentences = test_sentences
        self.sentences = sentences

        self.features_freq = dict()
        # ordered empty feature list filled with 0
        # self.empty_fea_dict = dict()
        self.feature_name_map = dict()

        self.unigram_set= set()
        self.unigram_set.add(UNK_Token)
        self.postags_set = set()
        self.chunktags_set = set()
        self.shape_set = set()
        self.shapeshort_set = set()
        self.tokenlen_set = set()
        self.suffs_set = set()
        self.prefs_set = set()

        self.netags = dict()

        # feature dict for all sentences in training/dev over all netags
        self.all_features_train_dict = dict()
        self.all_features_test_dict = dict()

        self.feature_dict_init()
        self.create_ordered_fea_dict()
        self.netags_list = list(self.netags.keys())
        self.feature_name_list = list(self.features_freq.keys())
        self.fea_vector_len = len(self.feature_name_list)
        self.feature_key_set = set(self.features_freq.keys())

  


    def create_ordered_fea_dict(self):
        ordered_fea_dict = collections.OrderedDict()  
        key_index = np.arange(len(self.features_freq))
        values = list(self.features_freq.keys())
        # dict feature_name : index
        temp_dict = dict(zip(values,key_index))
        self.feature_name_map = collections.OrderedDict(sorted(temp_dict.items()))
        # ordered_fea_dict = {key:0 for key in list(self.features_freq.keys())}
        # self.empty_fea_dict = collections.OrderedDict(sorted(ordered_fea_dict.items()))
        print("created empty feature dict")

       



    '''
    # initiate simple features in feature frequency dictionary
    #
    # i.e. 'shape_isint': 0, 'shape_containsappos': 0,.....
    '''

    # def simple_feature_init(self):
    #     global simple_features
    #     for fea in simple_features:
    #         self.features_freq[fea] = 0

    '''
       e.g. { 'pref=ga': 106,'unigram=olympic': 8, 'pref=ge': 110,....}
        fill in features frequecy dictionary, discard freq < threshold
    '''
    def feature_dict_init(self):
        #print docno,
        data_file = open(self.filename, "r")

        rawwords={}
        postags ={}
        chunktags = {}
        wordlens = {}
        shapes = {}
        shortshapes={}
        suffs = {}
        prefs = {}
        

        for line in data_file:
            line = line.strip()
            if line != "" and not line.startswith("-DOCSTART-"):

                w = re.split("\s",line)
                w3 = w[3]
                self.netags[w3]=self.netags.setdefault(w3,0)+1
                word_ori = w[0]
                # create simple features:
                for fea in simple_features:
                    fea = fea+"_"+w3
                    self.features_freq[fea] = self.features_freq.setdefault(fea,0)+1

                # create complex features
                w0=norm_digits(clean(w[0]).lower())
                unigram_ne = w0+"_"+w3
                rawwords[unigram_ne]=rawwords.setdefault(unigram_ne,0)+1
                
                # word len
                word_len = str(len(word_ori))+"_"+w3
                wordlens[word_len] = wordlens.setdefault(word_len,0)+1
                
                # shape and shortshape       
                shape = re.sub('[A-Z]', 'X', word_ori)
                shape = re.sub('[a-z]', 'x', shape)
                shape = re.sub('[0-9]', '0', shape)
                shape = re.sub('#', '#', shape)
                shape = re.sub('[^A-Za-z0-9#]', '.', shape)+"_"+w3

                shapeshort = re.sub('X+', 'X', shape)
                shapeshort = re.sub('x+', 'x', shapeshort)
                shapeshort = re.sub('0+', '0', shapeshort)
                shapeshort = re.sub('#+', '#', shapeshort)
                shapeshort = re.sub('\.+', '.', shapeshort)+"_"+w3
                
                shapes[shape] = shapes.setdefault(shape,0)+1
                shortshapes[shapeshort]= shortshapes.setdefault(shapeshort,0)+1
                
                # suff and pref
                if len(w0) > 1:
                    suff1 = w0[-1]+"_"+w3
                    pref1 = w0[:1]+"_"+w3
                    suffs[suff1]= suffs.setdefault(suff1,0)+1
                    prefs[pref1]=prefs.setdefault(pref1,0)+1
                if len(w0) > 2:
                    suff2 = w0[-2]+"_"+w3
                    pref2 = w0[:2]+"_"+w3
                    suffs[suff2]= suffs.setdefault(suff2,0)+1
                    prefs[pref2]=prefs.setdefault(pref2,0)+1
                if len(w0) > 3:
                    suff3 = w0[-3]+"_"+w3
                    pref3 = w0[:3]+"_"+w3
                    suffs[suff3]= suffs.setdefault(suff3,0)+1
                    prefs[pref3]=prefs.setdefault(pref3,0)+1
                    

                # postags
                w1=escape(w[1])+"_"+w3
                postags[w1]=postags.setdefault(w1,0)+1
                w2=w[2]+"_"+w3
                chunktags[w2]=chunktags.setdefault(w2,0)+1

        # discard features < 5 times
        # construct ungram features
        for w,c in rawwords.iteritems():
            if c>=self.freq_limit:
                feature_name = 'unigram='+w
                self.features_freq[feature_name]=c
                self.unigram_set.add(w)
            else:
                feature_name = 'unigram='+UNK_Token+"_"+w3
                self.features_freq[feature_name]=self.features_freq.setdefault(feature_name,0)+c
    # 			oovcount+=1
        print "Constructed unigramt features:"
        
        # construct POSTAG
        for t,c in postags.iteritems():
            if c>=self.freq_limit:
                feature_name = 'postag='+t
                self.features_freq[feature_name]=c
                self.postags_set.add(t)
        # construct chunktag
        for chunk,c in chunktags.iteritems():
            if c>=self.freq_limit:
                feature_name = 'chunktag='+chunk
                self.features_freq[feature_name]=c
                self.chunktags_set.add(chunk)
        
        # construct word len
        for l,c in wordlens.iteritems():
            if c>=self.freq_limit:
                feature_name = 'tokenlen='+str(l)
                self.features_freq[feature_name]=c
                self.tokenlen_set.add(l)
            # construct word len
        for s,c in shapes.iteritems():
            if c>=self.freq_limit:
                feature_name = 'shape='+s
                self.features_freq[feature_name]=c
                self.shape_set.add(s)
                
                # construct word len
        for ss,c in shortshapes.iteritems():
            if c>=self.freq_limit:
                feature_name = 'shapeshort='+ss
                self.features_freq[feature_name]=c
                self.shapeshort_set.add(ss)
                    # construct word len
        for suff,c in suffs.iteritems():
            if c>=self.freq_limit:
                feature_name = 'suff='+suff
                self.features_freq[feature_name]=c
                self.suffs_set.add(suff)
                
        for p,c in prefs.iteritems():
            if c>=self.freq_limit:
                feature_name = 'pref='+p
                self.features_freq[feature_name]=c
                self.prefs_set.add(p)
        
        print "finished feature space construction"
        print "feature space length"
        print(len(self.features_freq))
        print "ne tag list length:"
        print(len(self.netags))
        return



        '''
        take a given line ("Sheep NNP I-NP O"),and a NE tag,
        generate its feature vector
        take training data or testing data
        '''
    def feature_vector_gen(self,line,netag):
        
        fea_vector=[0]*self.fea_vector_len
        fm = self.feature_name_map 
        if line != "":
            t0 = time.time()
            # w = re.split("\s",line)
            w = line
            # w3 = w[3]
            NE_suff = '_'+netag

            # filling the feature dictionary
            word_ori = w[0]
            w0=norm_digits(clean(w[0]).lower())
            # simple features generation 
            if(capitalized.match(word_ori)):
                # if "shape_capitalized"+NE_suff in self.feature_key_set:
                fea_vector[fm["shape_capitalized"+NE_suff]] = 1
            
            if(caplet.match(word_ori)):
                fea_vector[fm["shape_caplet"+NE_suff]] = 1
            if(allcaps.match(word_ori)):
                fea_vector[fm["shape_allcaps"+NE_suff]]= 1
            if(caps.match(word_ori)):
                fea_vector[fm["shape_caps"+NE_suff]] = 1
            if(contains_num.search(word_ori)):
                fea_vector[fm["shape_containsnum"+NE_suff]] = 1
            if(contains_hyp.search(word_ori)):
                fea_vector[fm["shape_containshyp"+NE_suff]]= 1
            if(contains_appos.search(word_ori)):
                fea_vector[fm["shape_containsappos"+NE_suff]] = 1
            if(contains_per.search(word_ori)):
                fea_vector[fm["shape_containsper"+NE_suff]] = 1
            if(is_num.match(word_ori)):
                fea_vector[fm["shape_isnum"+NE_suff]] = 1
            if(is_int.match(word_ori)):
                fea_vector[fm["shape_isint"+NE_suff]] = 1
            if(ends_s.search(word_ori)):
                fea_vector[fm["shape_ends"+NE_suff]] = 1
            if(ends_ing.search(word_ori)):
                fea_vector[fm["shape_ended"+NE_suff]] = 1
            if(ends_ed.search(word_ori)):
                fea_vector[fm["shape_ending"+NE_suff]] = 1

            t1 = time.time()
            #print("simple features time: ", t1-t0)               
            # set unigram feature
            if 'unigram='+w0+NE_suff   in self.feature_key_set:
                
                feature_name = 'unigram='+w0+NE_suff  
                fea_vector[fm[feature_name]] = 1
            elif 'unigram='+UNK_Token+NE_suff in self.feature_key_set:
                feature_name = 'unigram='+UNK_Token+NE_suff
                fea_vector[fm[feature_name]] = 1
                    
            t2 = time.time()
            #print("unigram features time: ", t2-t1)  
                # word len
            word_len = str(len(word_ori))+NE_suff
            if 'tokenlen='+str(word_len) in self.feature_key_set:
                feature_name = 'tokenlen='+str(word_len)
                fea_vector[fm[feature_name]] = 1
                
            t3 = time.time()
            #print("word length features time: ", t3-t2)             
                # shape and shortshape
            
            shape = re.sub('[A-Z]', 'X', word_ori)
            shape = re.sub('[a-z]', 'x', shape)
            shape = re.sub('[0-9]', '0', shape)
            shape = re.sub('#', '#', shape)
            shape = re.sub('[^A-Za-z0-9#]', '.', shape)+NE_suff

            shapeshort = re.sub('X+', 'X', shape)
            shapeshort = re.sub('x+', 'x', shapeshort)
            shapeshort = re.sub('0+', '0', shapeshort)
            shapeshort = re.sub('#+', '#', shapeshort)
            shapeshort = re.sub('\.+', '.', shapeshort)+NE_suff
            
            if 'shape='+shape in self.feature_key_set:
                feature_name = 'shape='+shape
                fea_vector[fm[feature_name]] = 1
                
            if 'shapeshort='+shapeshort  in self.feature_key_set:
                feature_name = 'shapeshort='+shapeshort 
                fea_vector[fm[feature_name]] = 1
    
            t4 = time.time()
            #print("shape features time: ", t4-t3)  
                # suff and pref
            if len(w0) > 1:
                suff1 = w0[-1]+NE_suff
                pref1 = w0[:1]+NE_suff
                if 'suff='+suff1 in self.feature_key_set:
                    feature_name = 'suff='+suff1
                    fea_vector[fm[feature_name]] = 1
                if 'pref='+pref1 in self.feature_key_set:
                    feature_name = 'pref='+pref1
                    fea_vector[fm[feature_name]] = 1
                    
                
            if len(w0) > 2:
                suff2 = w0[-2]+NE_suff
                pref2 = w0[:2]+NE_suff
                if 'suff='+suff2 in self.feature_key_set:
                    feature_name = 'suff='+suff2
                    fea_vector[fm[feature_name]] = 1
                if 'pref='+pref2 in self.feature_key_set:
                    feature_name = 'pref='+pref2
                    fea_vector[fm[feature_name]] = 1
                    
            
            if len(w0) > 3:
                suff3 = w0[-3]+NE_suff
                pref3 = w0[:3]+NE_suff
                if 'suff='+suff3 in self.feature_key_set:
                    feature_name = 'suff='+suff3
                    fea_vector[fm[feature_name]] = 1
                if 'pref='+pref3 in self.feature_key_set:
                    feature_name = 'pref='+pref3
                    fea_vector[fm[feature_name]] = 1               
                
                # postags
            w1=escape(w[1])+NE_suff
            # chunktags
            w2=w[2]+NE_suff
        
            if 'postag='+w1 in self.feature_key_set:
                feature_name = 'postag='+w1
                fea_vector[fm[feature_name]] = 1
                
            if 'chunktag='+w2 in self.feature_key_set:
                feature_name = 'chunktag='+w2
                fea_vector[fm[feature_name]] = 1

            #t5 = time.time()
            #print("prefix suffix postage features time: ", t5-t0)  
        # deal with START and STOP token, take an empty line, 
        # return a 
        elif(line == ""):
            fea_vector = fea_vector            
        return fea_vector



    '''
    for unary feature, get a matrix of features 
    indexed by a dictionary
    dict.key = sentence id
    matrix.shape = [len of sent*len of tag_list, fea_vector]
    [word1-tag1,word1-tag2.....wordn-tag1...wordn-tagn]
    take training or testing data
    '''
    def compute_feature_matrix(self,sentences):
        print ('------begin compute feature matrix-------------')
        print("------feature lenth",self.fea_vector_len)
        sentences = sentences
        netags_list = self.netags_list 
        all_features_dict= dict()
       
        N = len(netags_list)
        print("number of sentences:",len(sentences))
        for i in xrange(0,len(sentences)):
            sent = sentences[i]
            # print sent
            sent_len = len(sent)

            t0 = time.time()
            
            temp_fea_matrix = scipy.sparse.csr_matrix((len(self.netags_list)*sent_len,self.fea_vector_len), dtype=np.int8)
            # loop over all words
            for j in xrange(0,sent_len):
                # [word-tag1,word-tag2....]
                for k in xrange(0,len(self.netags_list)):
            # fea_vector in np sparse matirx
                # print(len(
                    # self.fea_gen.feature_vector_gen(line,self.netags_list[i])))
                    temp_fea_matrix[(j)*N+k,:]= scipy.sparse.csr_matrix(\
                    self.feature_vector_gen(sent[j],self.netags_list[k]))
            
            print("-------feature gen at sent:",i)
            print("------time:",time.time()-t0)
            all_features_dict[i] = temp_fea_matrix
        return all_features_dict
                
    



                
                



