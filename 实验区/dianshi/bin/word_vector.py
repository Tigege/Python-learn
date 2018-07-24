import codecs
from gensim.models.word2vec import LineSentence


import gensim
from gensim.models import word2vec
def allget():
	dict1={}
	allword=open('../data/all_fin.csv','w+')
	file=['../data/train.csv','../data/test.csv']
	for filename in file:
		num=0
		with open(filename) as fn:
			if 'train' in filename:
				for line in fn:
					num+=1
					if num==1:
						pass
					else:
						if line.strip().split(',')[4] in dict1:
							pass
						else:
							dict1[line.strip().split(',')[4]]='1'
							allword.write(line.strip().split(',')[4]+'\n')
							allword.flush()
			if 'test' in filename:
				for line in fn:
					num+=1
					if num==1:
						pass
					else:
						if line.strip().split(',')[3] in dict1:
							pass
						else:
							dict1[line.strip().split(',')[3]]='1'
							allword.write(line.strip().split(',')[3]+'\n')
							allword.flush()
	allword.close()
allget()
in_path='../data/all_fin.csv'
model=word2vec.Word2Vec(sentences=LineSentence(in_path), size=300,sg=1,min_count=5,window=10,workers=30)
model.save('../results_gensim_vec/all_fin_train300_model.m')


