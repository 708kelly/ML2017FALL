#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:39:20 2018

@author: kelly
"""

from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from sklearn.cluster import KMeans

#import pandas as pd
import numpy as np
import os
import sys

input_img = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# build encoder
encoder = Model(input=input_img, output=encoded)

# build autoencoder
adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()

#os.chdir("/Users/kelly/Documents/大四/機器學習/Hw6")
# load images
train_num = 130000
path_1 = sys.argv[1]
X = np.load(path_1)
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))
x_train = X[:train_num]
x_val = X[train_num:]
x_train.shape, x_val.shape

# train autoencoder
autoencoder.fit(X, X,
                epochs=1,
                batch_size=128,
                shuffle=True)
autoencoder.save('autoencoder.h5')
encoder.save('encoder.h5')

# after training, use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
clf = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


import csv
#from sklearn.metrics.pairwise import cosine_similarity

n_row = 0
#path_1 = sys.argv[1]
index_1 = []
index_2 = []
path_2 = sys.argv[2]
text = open(path_2, 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    
    # 第0列沒有資訊
    if n_row != 0:
        index_1.append(int(r[1]))
        index_2.append(int(r[2]))
        
    n_row = n_row+1
text.close()
answers = []
coss = []
for j in range(len(index_1)):
    first = index_1[j]
    second = index_2[j]
    if clf.labels_[first] == clf.labels_[second]:
        answers.append(1)
    else:
        answers.append(0)
       
#np.linalg.norm(result[19897]-result[107188])
#plt.plot(coss)
filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["ID","Ans"])

for i, v in  enumerate(answers):
    s.writerow([(i),v])
text.close()
