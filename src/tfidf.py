# -*- coding: utf-8 -*-

import pandas as pd
import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import gensim
from nltk.tokenize import RegexpTokenizer
from itertools import compress
import datetime


gossip = pd.read_csv('all_documents.csv')

gossip = gossip.drop_duplicates(keep='last')
gossip = gossip.reset_index(drop=True)

corpus = [i for i in gossip['content_split']]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
print tfidf.shape
words = vectorizer.get_feature_names()

text_filtered = []
for i in range(tfidf.shape[0]):
	text_filtered.append([k for k in compress(words, [j >0.1 for j in tfidf[i,].toarray()[0]])])

gossip['words_filtered'] = [";".join(i).encode('utf-8') for i in text_filtered]

gossip.to_csv('all_documents_after_tfidf.csv')

notification.notification('the filtering of tfidf is finished')
