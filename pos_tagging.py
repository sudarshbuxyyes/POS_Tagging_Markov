# -*- coding: utf-8 -*-
"""POS_tagging.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DhjYe0uktuCpU85SDj5N8mIHyMBaZU9D
"""

import string
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.corpus import words
#from nltk.stem import WordNetLemmatizer
#from nltk import tokenize
from nltk.tag import pos_tag


def preprocessing(str):
    #stopword removal
    Stop = stopwords.words('english')
    #remove punctuation
    no_punc = [char for char in str if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    #add word lemmatization.
    return ' '.join([word for word in no_punc.split() if (word.lower() not in Stop)])

def pos_tagging(s):
    nopunc_sentence = preprocessing(s)
    print(nopunc_sentence)
    tags = pos_tag(nltk.word_tokenize(nopunc_sentence))
    dict_tag = {'NN':[], 'VB':[], 'OTH':[]} #only three categories at the moment, can add all categories coming from pos_tag programmatically.
    for tag in tags:
        if tag[1] in ["NN","NNS","NNP","NNPS"]: #NN-singular noun, NNS- noun plural, NNP-proper noun singular, NNPS- proper noun plural.
            dict_tag['NN'].append(tag[0])
        elif tag[1] == "VB":
            dict_tag['VB'].append(tag[0])
        else:
            dict_tag['OTH'].append(tag[0])
    return dict_tag

string2 = "You should be ashamed of yourself! For slandering Newcastle United"
tags2 = pos_tagging(string2)
print(tags2)