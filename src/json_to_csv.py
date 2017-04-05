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
import glob, os
import csv
import json
import datetime
from utils import notification


tmp_all=[]
tmp_all.append(['title','date','url','n_hate','content_split'])
for data_file in glob.glob("files_contend/*.json"):
    
	print data_file
	json_data=open(data_file).read()
    
	data = json.loads(json_data)
	i = 0
	for post in data:
		tmp=[]
		if u'新聞' in post['title'] and u'Re: [新聞]' not in post['title'] and u'Fw: [新聞]' not in post['title']:
			print post['title'].encode('utf-8')
			try:

				tmp.append(post['title'].encode('utf-8'))
				tmp.append(post['date'])
				tmp.append(post['url'])
				tmp.append(" ".join(jieba.cut(post['content'], cut_all=False)).encode('utf-8'))
				tmp_all.append(tmp)
			except:
				print("except")


with open("all_documents.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(tmp_all)

notification.notification('the transfer of json to csv is ready')
