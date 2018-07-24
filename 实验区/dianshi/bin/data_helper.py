import re
import pandas as pd
import numpy as np
import jieba
import codecs

def ad_detect(discuss):
	return discuss
def splitWord(query,stopwords):#分词返回字符串，停用词表是自己找的删除了一些程度描述词
	if len(query)==0:
		query='空信息'
	wordList = jieba.cut(query)
	num = 0
	result = ''
	for word in wordList:
		word = word.rstrip()
		word = word.rstrip('"')
		if word not in stopwords:
			if num == 0:
				result = word
				num = 1
			else:
				result = result + ' ' + word
	return result
def preprocess(data):
	stopwords = {}
	for line in codecs.open('../data/stop.txt','r','utf-8'):
		stopwords[line.rstrip()]=1
	data['split_word'] = data['content'].map(lambda x:splitWord(x,stopwords))
	return data
def select_cn(discuss):
	discuss = re.sub("[A-Za-z0-9\!\%\[\]\,\。\?\？\、\。\，\；\’\【\】\·\~\！\@\#\￥\%\……\&\*\（\）\》\《\：\“\”\{\}\<\/\>\.\ \;\；]", "", discuss)
	return discuss
if __name__ == '__main__':
	train_df = pd.read_csv('../data/data_train.csv',sep='\t',encoding='gbk')
	test_df = pd.read_csv('../data/data_test.csv',sep='\t',encoding='gbk')
	train_df['content']=train_df['content'].fillna('空') 
	test_df['content']=test_df['content'].fillna('空') 
	train_df['content'] = train_df['content'].apply(select_cn)
	test_df['content'] = test_df['content'].apply(select_cn)
	train_df = preprocess(train_df)
	test_df = preprocess(test_df)
	train_df.to_csv('../data/train.csv',index=None)
	test_df.to_csv('../data/test.csv',index=None)
