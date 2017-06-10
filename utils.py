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



punc = re.compile("\W")

string_map = {".":"_P_" , ",":"_C_", "'":"_A_", "%":"_PCT_", "-":"_DASH_",
	      "$":"_DOL_", "&":"_AMP_", ":":"_COL_", ";":"_SCOL_", "\\":"_BSL_"
	      , "/":"_SL_", "`":"_QT_", "?":"_Q_", "=":"_EQ_", "*":"_ST_",
	      "!":"_E_", "#":"_HSH_", "@":"_AT_", "(":"_LBR_", ")":"_RBR_"
	      , "\"":"_QT1_"}




def escape(s):
    for val, repl in string_map.iteritems():
        s = s.replace(val,repl)
    return s
        
def clean(s):
    s = escape(s)
        #if re.match("[^A-Za-z0-9_]+",s):
        #	print s
    return punc.sub("_",s)

def printind(s):
    return '_'+str(s).replace("-","m")+'_'


def norm_digits(s):
    return re.sub("\d","0",s)