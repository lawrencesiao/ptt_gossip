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
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.mime.image import MIMEImage 


all_ = pd.read_csv('output_filtered2.csv')

all_['date'] =  pd.to_datetime(all_['date'], format='%Y-%m-%d %H:%M:%S')
all_['score_positive'] = 0
all_['score_hate'] = 0
all_['score_neutral'] = 0

base = max(all_['date']).replace(hour=0, minute=0, second=0, microsecond=0)+ datetime.timedelta(days=1)
n_days = 1+(base - min(all_['date']).replace(hour=0, minute=0, second=0, microsecond=0)).days

date_upper_list = [base - datetime.timedelta(hours=x*6) for x in range(0, n_days * 4-56)]
date_lower_list = [base - datetime.timedelta(hours=x*6) for x in range(56, n_days * 4)]

print len(date_upper_list) == len(date_lower_list)

count=0

for up, low in zip(date_upper_list,date_lower_list):
	count+=1
	print str(up) + ' to ' + str(low)
	print 'finished: ' + str(float(count)/float(len(date_upper_list)))

	gossip = all_[[i & j for i, j in zip(all_['date'] >= low, all_['date'] <= up)]]
	gossip = gossip.reset_index(drop=True)

	if len(gossip) == 0:
		next
	else:
		text_filtered = [i.split(';') for i in gossip['words_filtered']]
		dictionary = corpora.Dictionary(text_filtered)
		corpora1 = [dictionary.doc2bow(text) for text in text_filtered]
		ldamodel = gensim.models.ldamodel.LdaModel(corpora1, num_topics=20, id2word = dictionary, passes=20)

		index = range(20)

		columns = ['counts','all_n_push', 'all_n_hate','all_n_neutral']
		df_ = pd.DataFrame(index=index, columns=columns)
		df_ = df_.fillna(0) # with 0s rather than NaNs

		for j in range(len(gossip)):
		    for i in ldamodel.get_document_topics(corpora1[j]):
		        df_.ix[i[0],['counts']] +=i[1]
		        df_.ix[i[0],['all_n_push']] +=gossip.ix[j,'n_push']*i[1]
		        df_.ix[i[0],['all_n_hate']] +=gossip.ix[j,'n_hate']*i[1]
		        df_.ix[i[0],['all_n_neutral']] +=gossip.ix[j,'n_neutral']*i[1]


		df_['avg_n_push'] = df_['all_n_push']/df_['counts']
		df_['avg_n_hate'] = df_['all_n_hate']/df_['counts']
		df_['avg_n_neutral'] = df_['all_n_neutral']/df_['counts']

		idx_for_test = [i & j for i, j in zip(all_['date'] >= up, all_['date'] <= up + datetime.timedelta(hours=6))]

		corpora2 = [dictionary.doc2bow(text) for text in [i.split(';') for i in all_['words_filtered'][idx_for_test]]]

		score_positive = []
		score_hate = []
		score_neutral = []

		for i in corpora2:
		    tmp_topics = ldamodel.get_document_topics(i)
		    score_positive_tmp = 0
		    score_hate_tmp = 0
		    score_neutral_tmp = 0		
		    for tmp_topic in tmp_topics:
		        score_positive_tmp += df_.ix[tmp_topic[0],'avg_n_push'] * tmp_topic[1]
		        score_hate_tmp += df_.ix[tmp_topic[0],'avg_n_hate'] * tmp_topic[1]
		        score_neutral_tmp += df_.ix[tmp_topic[0],'avg_n_neutral'] * tmp_topic[1]


		    score_positive.append(score_positive_tmp)
		    score_hate.append(score_hate_tmp)
		    score_neutral.append(score_neutral_tmp)

		all_['score_positive'][idx_for_test] = score_positive
		all_['score_hate'][idx_for_test] = score_hate
		all_['score_neutral'][idx_for_test] = score_neutral


all_.to_csv('output_with_score.csv')

notification.notification('the topic features are ready!')




