# test 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:20:58 2017

@author: kelly
"""



import csv
import os
import numpy as np
#import random
import sys


import os
import re
import numpy as np
import _pickle as pk
import random


from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional,Flatten,Dot,Add ,  Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.regularizers import l2


#os.chdir("/Users/kelly/Documents/大四/機器學習/Hw5")

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def rate(model, user_id, item_id):
    return model.predict([np.array([user_id]), np.array([item_id])])[0][0]

def build_model(user_num,item_num,latent_dim=666):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    user_vec = Embedding(user_num,latent_dim,embeddings_initializer = 'random_normal', embeddings_regularizer=l2(1e-5))(user_input)
    user_vec = Flatten()(user_vec)

    item_vec = Embedding(item_num,latent_dim,embeddings_initializer = 'random_normal', embeddings_regularizer=l2(1e-5))(item_input)
    item_vec = Flatten()(item_vec)

    user_bias = Embedding(user_num,1,embeddings_initializer = 'zeros', embeddings_regularizer=l2(1e-5))(user_input)
    user_bias = Flatten()(user_bias)

    item_bias = Embedding(item_num,1,embeddings_initializer = 'zeros', embeddings_regularizer=l2(1e-5))(item_input)
    item_bias = Flatten()(item_bias)

    hat = Dot(axes = 1)([user_vec,item_vec])
    hat = Add()([hat,user_bias,item_bias])
    hat = Lambda(lambda x:  K.constant(1.116897661146206) * x + K.constant(3.581712))(hat)
    
    model = Model([user_input,item_input],hat)
    model.compile(loss = 'mse',optimizer = 'adamax', metrics=[rmse])
    return model

#movies = []
#train_ratings =[] 
n_row = 0
users =[]
movies = []
path_1 = sys.argv[1]
text = open(path_1, 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    
    # 第0列沒有資訊
    if n_row != 0:
        user = int(r[1])
        movie = int(r[2])
        #rating = float(r[3])
        users.append(user)
        movies.append(movie)
        #train_ratings.append(rating)
        
    n_row = n_row+1
text.close()

users = np.array(users)
movies = np.array(movies)


test_users = users - 1
test_movies = movies - 1

model = build_model(6040, 3952, 10)
print('Loading model weights...')
model.load_weights('hw5_5_1.h5')
print('Loading model done!!!')

#model = load_model('model/hw5_4_1.h5')
print(model.summary())

p_result = model.predict([test_users,test_movies], verbose=1)

#mean = 3.5817120860388076
#std =  1.116897661146206
#train = ((train_ratings - mean)/std)
#rating = p_result * std + mean

print(max(p_result))
print(min(p_result))
#result = np.around(p_result)
#result = rating.reshape(len(rating),)

result = []
for i in p_result:
    #i = 4.2
    #print(i)
    if i < 1.0  :
        #print('1')
        result.append(1.0)
    
    elif i >= 5:
        result.append(5.0)
        #print('5')
        
    else:
        result.append(i[0])

print(max(result))
print(min(result))

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["TestDataID","Rating"])

for i, v in  enumerate(result):
    s.writerow([(i+1),v])
text.close()
