************************************************
Named Entity Recognition using perceptron learning
Data: CoNLL-2003
Algorithm: Structured Perceptron
author : An Yan
date: March, 2017
Python version: 2.7
*************************************************

HOW TO RUN:

example:

python NER_yanan.py -t 5 eng.train.small eng.dev.small

** this will output accuracy and classification report
** I did not use  conlleval.txt script, but sklearn classfication report

****************************
parameters: 

-t: the threshold of discarding features that occur less than K times in the training set.
    default = 1. 
test data set 
dev data set or test data set.

*****************************
Help message, please type


python yanan_lm.py -h

************************************************
