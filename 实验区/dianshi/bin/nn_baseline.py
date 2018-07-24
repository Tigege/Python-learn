# -*- coding: utf-8 -*-
import pandas  as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import fasttext
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GlobalMaxPool1D,GRU, Embedding,Bidirectional, Flatten,LSTM, BatchNormalization,Conv1D,MaxPooling1D
from keras.models import Model
from keras.layers import GlobalMaxPooling1D
from keras.layers import *
from keras.layers.convolutional import Convolution1D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras import regularizers
import re
from keras.layers import Input, Concatenate
import numpy as np
import gensim
from keras.utils import np_utils
class Model1:
	def __init__(self):
		self.models = gensim.models.Word2Vec.load('../results_gensim_vec/all_fin_train300_model.m')
	def getVec(self,msg):
		return self.models[msg]
new_model = Model1()
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


maxlen = 30
embed_size = 300
train.split_word.fillna('_na_',inplace=True)
test.split_word.fillna('_na_',inplace=True)
comment_text = np.hstack([train.split_word.values,test.split_word.values])
tok_raw = Tokenizer()
tok_raw.fit_on_texts(comment_text)
train['Discuss_seq'] = tok_raw.texts_to_sequences(train.split_word.values)
word_index = tok_raw.word_index
test['Discuss_seq'] = tok_raw.texts_to_sequences(test.split_word.values)




def get_keras_data(dataset): 
	X={
		'Discuss_seq_pre':pad_sequences(dataset.Discuss_seq,maxlen=maxlen,padding='pre', truncating='pre'),
		'Discuss_seq_post':pad_sequences(dataset.Discuss_seq,maxlen=maxlen,padding='post', truncating='post')
	}
	return X

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def cnn_w2c():
	#Inputs
	comment_seq_pre = Input(shape=[maxlen],name='Discuss_seq_pre')
	comment_seq_post = Input(shape=[maxlen],name='Discuss_seq_post')
	emb_comment1 =Embedding(len(word_index) + 1, embed_size,weights=[embedding_matrix])(comment_seq_pre)
	emb_comment2 =Embedding(len(word_index) + 1, embed_size,weights=[embedding_matrix])(comment_seq_post)
	emb_comment =concatenate([emb_comment1,emb_comment2],axis=1)
	convs1 = []
	filter_sizes = [1,2,3,4,5,6]
	for fsz in filter_sizes:
		l_conv1 = Conv1D(filters=512,kernel_size=fsz,activation='relu')(emb_comment)
		avg_pool1 = GlobalAveragePooling1D()(l_conv1)
		max_pool1 = GlobalMaxPooling1D()(l_conv1)
		x1 = concatenate([avg_pool1, max_pool1])
		convs1.append(x1)
	merge1 =concatenate(convs1,axis=1)
	merge1 = BatchNormalization()(merge1)
	output = Dropout(0.3)(Dense(128,activation='relu')(merge1))#drop 0.3
	output = Dense(3,activation='softmax')(Dropout(0.1)(output))#drop 0.1
	model = Model(inputs=[comment_seq_pre,comment_seq_post],outputs=[output])
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mae", fmeasure])
	return model

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
for word, i in word_index.items():
	try:
		embedding_vector = new_model.getVec(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	except:
		pass
		
X_train =get_keras_data(train)
X_test = get_keras_data(test)
y_train = np_utils.to_categorical(train.label.values)
batch_size = 512
epochs = 15
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)
from keras.models import load_model

callbacks_list = [early_stopping]
pred=[]
for ii in range(5):
	model1 = cnn_w2c()
	model1.summary()
	model1.fit(X_train, y_train,
			validation_split=0.1,
			batch_size=batch_size, 
			epochs=epochs, 
			shuffle = True,
			callbacks=callbacks_list)
	preds = model1.predict(X_test)
	model1.save('../model/model'+str(ii)+'.h5')
	pred.append(preds)
result1=[]
for a,b,c,d,e in zip(pred[0],pred[1],pred[2],pred[3],pred[4]):
	result1.append(np.array(a.tolist()) + np.array(b.tolist())+np.array(c.tolist()) + np.array(d.tolist())+ np.array(e.tolist()))
aa=open('./submission.csv','w+')
for k1,v1 in zip(result1,test.id.values):
	lable=k1.tolist().index(max(k1.tolist()))
	aa.write(str(v1)+','+str(lable)+'\n')
	aa.flush()

