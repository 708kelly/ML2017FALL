#Created on Wed Nov 29 23:24:04 2017

#@author: kelly
import os
import re
import numpy as np
import _pickle as pk
import random
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf

#from utils.util import DataManager


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

# build model
def simpleRNN(vocab_size,max_length,cell):
    embedding_dim = 300
    #max_length = 30
    inputs = Input(shape=(max_length,))

    # Embedding layer
    embedding_inputs = Embedding(vocab_size, 
                                 embedding_dim, 
                                 trainable=True)(inputs)
    hidden_size = 300
    # RNN 
    return_sequence = False
    dropout_rate = 0.2
    if cell == 'GRU':
        RNN_cell = GRU(hidden_size, 
                       return_sequences=return_sequence, 
                       dropout=dropout_rate)
    elif cell == 'LSTM':
        RNN_cell = LSTM(hidden_size, 
                        return_sequences=return_sequence, 
                        dropout=dropout_rate)

    RNN_output = RNN_cell(embedding_inputs)

    # DNN layer
    outputs = Dense(hidden_size//2, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
        
    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile( loss='binary_crossentropy', optimizer=adam, metrics=[ 'accuracy',])
    
    return model



x = []
label = []
#n_line = 0
path_1 = sys.argv[1]
text = open(path_1, 'r', encoding='UTF-8') 

lines=text.readlines()
content = [x.strip() for x in lines]
#x=[]
n_line = 0
for l in content:
    if n_line != 0  :
        short = l.split('+++$+++')
        y = int(short[0])
        label.append(y)
        a = short[1].strip()
        #x = a.split(' ')
        x.append(a)
    # 第0列沒有資訊
    n_line = n_line+1
text.close()
x = np.array(x)


# Unlabeled Data
path_2 = sys.argv[2]
text = open(path_2, 'r', encoding='UTF-8') 
data = []
lines=text.readlines()
content = [x.strip() for x in lines]
#x=[]
n_line = 0
for l in content:
    if n_line != 0  :
        #short = l.split(',')
        #y = float(short[0])
        #test_label.append(y)
        a = l.strip()
        #x = a.split(' ')
        data.append(a)
    # 第0列沒有資訊
    n_line = n_line+1
text.close()



MAX_NB_WORDS = 20000

# load tokenizer
path = 'token.pkl'
tokenizer = pk.load(open(path, 'rb'))

maxlen = 30
# to_sequence
sequences = tokenizer.texts_to_sequences(x)
#x_data = np.array(sequences)
x_data = np.array(pad_sequences(sequences, maxlen=maxlen))

# to BOw
"""
bow = tokenizer.texts_to_matrix(x,mode='count')
tfidf = tokenizer.texts_to_matrix(x,mode='tfidf')
"""
# to category
y_data = np.array(label)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)



    
    
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
PATIENCE = 3
#Model_name
model_name = 'model/hw4_1.h5'

#epoch
num_epoch = 100
saveevery = 1

for k in range(1):
    
    #k = 0
    ## Train
    train_list = []
    train = [x for x in alist if x is not k]
    
    for j in train:
        train_list.extend(k_part[j])
    
    train_data = x_data[train_list]
    correct = y_data[train_list]
    #train_data = np.concatenate((train_data,train_data),axis = 0)
    #correct = np.concatenate((correct,correct),axis = 0)
    ## val
    val_list = k_part[k]
    val_data = x_data[val_list]
    val_correct = y_data [val_list]
    
    # augument
    #datagen.fit(train_data)
    #imgen_valid = ImageDataGenerator()

    #imgen_valid.fit(val_data)
    
    
    pretrain = False
    
    if pretrain == False:
        model = simpleRNN(MAX_NB_WORDS,maxlen,'LSTM')
        print (model.summary())
        #pretrain = True
    #else:
     #   model = load_model(model_name)
    
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
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
        #result =model.fit_generator(datagen.flow(*_shuffle(train_data, correct), batch_size=40),steps_per_epoch=train_data.shape[0]//(40), validation_data=imgen_valid.flow(*_shuffle(val_data, val_correct)), nb_epoch=1,validation_steps=10)
        result = model.fit(X_train,Y_train, 
                            validation_data=(val_data, val_correct),
                            epochs=1,
                            batch_size=32)
        #scores = model.evaluate(val_data, val_correct, verbose=0)
        scores = result.history['val_acc']
        train_acc.extend(result.history['acc'])
        val_acc.extend(result.history['val_acc'])
        train_loss.extend(result.history['loss'])
        val_loss.extend(result.history['val_loss'])
        print("Accuracy: %.2f%%" % (scores[0]*100))
        #print("Accuracy: %.2f%%" % (scores[1]*100))
        
        if scores[0] >= best_metrics:
            best_metrics = scores[0]
            print ("save best score!! "+str(scores[0]))
            model.save("model/hw4_LSTM_"+str(k+1)+".h5")
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

                                                
