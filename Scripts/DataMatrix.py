# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:18:47 2017

@author: Bharadwaj
"""
import pandas as pd
import csv
import re
import string
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from datetime import datetime

dataFrame = pd.read_csv('../DataSet/export_dashboard_aapl_2016_06_15_14_30_09_Stream.csv')
#getting date in "%Y-%m-%dT%H:%M:%S.%fZ" format
dataFrame["DateTime"] = dataFrame["Date"].map(str) + 'T'+ dataFrame["Hour"] + '00.00Z'

tweetdatalist = list(dataFrame.loc[:, "Tweet content"])
tweettimelist = list(dataFrame.loc[:, "DateTime"])
favlist = list(dataFrame.loc[:, "Favs"])
rtlist = list(dataFrame.loc[:, "RTs"])
followerslist = list(dataFrame.loc[:, "Followers"])

stop_words = list(stopwords.words('english'))
translate_table_for_punctuation = dict((ord(char), None) for char in string.punctuation)  

def build_lexicon(corpus):
    lexicon = set()    
    for doc in corpus:
        #removing links
        doc = re.sub(r'http\S+', '', doc)        
        #remove punctuation
        doc = doc.translate(translate_table_for_punctuation)
        list_of_words_n_doc = doc.split()
        lexicon.update([word for word in list_of_words_n_doc if not word in stop_words])
    return lexicon

#compute term frequency
def tf(term, document):
  return freq(term, document)/float(len(document.split()))

#compute idf
def idf(word, doclist):
    n_docs = len(doclist)
    dfreq = numDocsContaining(word, doclist)
    return np.log(n_docs / 1+dfreq)

def freq(term, document):
  return document.split().count(term)

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount 

vocabulary = build_lexicon(tweetdatalist)
my_idf_vector = [(word,idf(word, tweetdatalist)) for word in vocabulary]

timenormlist = []
favnormlist = []
rtnormlist = []
followersnormlist = []
size = len(tweettimelist)
min_tweet_time =  datetime.strptime(min(tweettimelist), "%Y-%m-%dT%H:%M:%S.%fZ")
max_tweet_time =  datetime.strptime(max(tweettimelist), "%Y-%m-%dT%H:%M:%S.%fZ")
range_tweet_time = (max_tweet_time - min_tweet_time).total_seconds()
max_fav = max(favlist)
max_rt = max(rtlist)
max_followers = max(followerslist)

for i in range(0, size):
    time_diff = (datetime.strptime(tweettimelist[i], "%Y-%m-%dT%H:%M:%S.%fZ") - min_tweet_time).total_seconds()
    timenormlist.append(float(time_diff)/range_tweet_time)
    if favlist[i] == '':
        favnormlist.append(float(0.1)/float(max_fav))
    else:
        favnormlist.append(float(favlist[i])/float(max_fav))
    if rtlist[i] == '':
        rtnormlist.append(float(0.1)/float(max_rt))
    else:
        rtnormlist.append(float(favlist[i])/float(max_rt))
    if followerslist[i] == '':
        followersnormlist.append(float(0.1)/float(max_followers))
    else:
        followersnormlist.append(float(favlist[i])/float(max_followers))    
    
doc_term_matrix = []
j = 0
for doc in tweetdatalist:    
    tf_vector = [tf(word, doc)*idf*timenormlist[j]*rtnormlist[j]*
                 favnormlist[j]*followersnormlist[j] for word, idf in my_idf_vector]
    doc_term_matrix.append(tf_vector)
    j += 1
    print(j)

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    for i in range(0, len(doc_term_matrix)):
        writer.writerows(doc_term_matrix[i])
