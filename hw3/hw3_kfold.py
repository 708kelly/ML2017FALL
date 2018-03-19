#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:27:18 2017

@author: kelly
"""


# coding: utf-8

# In[17]:

import csv
import os
import numpy as np
import random
import sys


from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Input
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import regularizers
#from matplotlib import pyplot as plt
from keras.layers import Dropout
from keras.models import load_model,Model

#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

"""
def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test
"""
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# Load Data
x_data = []
y_data = []
n_row = 0

path = sys.argv[1]
text = open(path, 'r') 
#text = open(os.path.join(path,'data/train.csv'), 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        short = r[1].split(' ')
        short = [float(i) for i in short]
        x_data.append(short)
        y_data.append(int(r[0]))
        	
    n_row = n_row+1
text.close()

x_data = np.array(x_data).reshape(28709,48,48,1)
y_data = np.array(y_data)

#np.save('y_data.npy',y_data)
#X_train, X_val, y_train, y_val = cross_validate(x_data, y_data)

# normalize inputs from 0-255 and 0.0-1.0
x_data = np.array(x_data).astype('float32')

x_data = x_data / 255.0

# one hot encode outputs
y_data = np.array(y_data)
#y_val = np.array(y_val)
y_data = np_utils.to_categorical(y_data)
#y_val = np_utils.to_categorical(y_val)
num_classes = y_data.shape[1]
print("Data normalized and hot encoded.")




def build_model(epoch):

    '''
    #先定義好框架
    #第一步從input吃起
    '''
    input_img = Input(shape=(48, 48, 1))
    
    block1 = Conv2D(32, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    #block2 = Conv2D(48, (3, 3), activation='relu')(block1)
    #block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block1)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    #block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    #block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block3)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(fc1)
    fc2 = Dropout(0.5)(fc2)
    
    
    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)


    epochs = epoch  # >>> should be 25+
    lrate = 0.001
    decay = lrate/epochs
    opt = SGD(lr=0.006, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    #opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
    
    

## k-fold
k_part = []
k_num = 10
num = len(x_data) // k_num

rlist = range(len(x_data))
for i in range(k_num):
     part = random.sample(rlist,num)
     rlist = [x for x in rlist if x not in part]
     k_part.append(part)
#l_rate = 0.001
#repeat = 10000
#lamda = 0.001
w_option = {}
w_error = {}
alist = range(k_num)
PATIENCE = 10
#Model_name
model_name = 'model/hw3_7_1_1.h5'

#epoch
num_epoch = 100
saveevery = 1
for k in range(2):
    
    #k = 0
    ## Train
    train_list = []
    train = [x for x in alist if x is not k]
    
    for j in train:
        train_list.extend(k_part[j])
    
    train_data = x_data[train_list]
    correct = y_data[train_list]
    train_data = np.concatenate((train_data,train_data),axis = 0)
    correct = np.concatenate((correct,correct),axis = 0)
    ## val
    val_list = k_part[k]
    val_data = x_data[val_list]
    val_correct = y_data [val_list]
    
    # augument
    datagen.fit(train_data)
    imgen_valid = ImageDataGenerator()

    imgen_valid.fit(val_data)
    
    
    pretrain = False
    
    if pretrain == False:
        model = build_model(num_epoch)
        #pretrain = True
    else:
        model = load_model(model_name)
    

    best_metrics = 0.0
    early_stop_counter = 0
    for e in range(num_epoch):
        #shuffle data in every epoch
        # Random shuffle
        X_train, Y_train = _shuffle(train_data,correct)
        print ('#######')
        print ('Epoch ' + str(e+1))
        print ('#######')
        #start_t = time.time()
        #model.fit(X_train, Y_train, validation_data=(val_data, val_correct), nb_epoch=1, batch_size=32)
        result =model.fit_generator(datagen.flow(*_shuffle(train_data, correct), batch_size=40),steps_per_epoch=train_data.shape[0]//(40), validation_data=imgen_valid.flow(*_shuffle(val_data, val_correct)), nb_epoch=1,validation_steps=10)

        #scores = model.evaluate(val_data, val_correct, verbose=0)
        scores = result.history['val_acc']
        print("Accuracy: %.2f%%" % (scores[0]*100))
        #print("Accuracy: %.2f%%" % (scores[1]*100))
        
        if scores[0] >= best_metrics:
            best_metrics = scores[0]
            print ("save best score!! "+str(scores[0]))
            model.save("model/hw3_7_"+str(k+1)+".h5")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        """
        if (e+1) % saveevery == 0:
            model.save(model_name+str(e+1))
            print ('Saved model %s!' %str(e+1))
        """
        if early_stop_counter >= PATIENCE:
            print ('Stop by early stopping')
            print ('Best score: '+str(best_metrics))
            break
    w_error[k] = best_metrics
    
order = sorted(w_error.items(), key = lambda t : t[1],reverse=True)
print("Best-k : ",(order[0][0]+1),"/ score : ",order[0][1])
print(order)    

