#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:43:35 2017

@author: kelly
"""
import csv
import os
import numpy as np
#import random
import sys


from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import regularizers
#from matplotlib import pyplot as plt
from keras.layers import Dropout
from keras.models import load_model,Model
from keras.utils.vis_utils import plot_model


#import sklearn
#from sklearn.model_selection import train_test_split

#os.chdir("/Users/kelly/Documents/大四/機器學習/Hw3") 
path_1 = sys.argv[1]
model = load_model('hw3_7_1_1.h5')

test_x = []
text = open(path_1, 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
n_line = 0
for l in row:
    if n_line != 0  :
        short = l[1].split(' ')
        short = [float(i) for i in short]
        test_x.append(short)
    # 第0列沒有資訊
    
    n_line = n_line+1
text.close()

test_x = np.array(test_x)
test_x = test_x.reshape(7178,48,48,1)
test_x = np.array(test_x).astype('float32')
test_x = test_x / 255.0


# Final evaluation of the model
p_result = model.predict(test_x, verbose=1)
result = p_result.argmax(axis=-1)
path_2 = sys.argv[2]

filename = path_2
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])

for i, v in  enumerate(result):
    s.writerow([(i),int(v)])
text.close()


