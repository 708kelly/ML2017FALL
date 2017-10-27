#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:27:41 2017

@author: kelly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:58:10 2017

@author: kelly
"""
import csv 
import numpy as np
#from numpy.linalg import inv
import random
import math
import sys

#import os



def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return np.clip(res, 1e-8, 1-(1e-8))

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
path_x = sys.argv[1]
text = open(path_x, 'r', encoding='big5')
lines=text.readlines()
content = [x.strip() for x in lines]
x=[]
n_line = 0
for l in content:
    if n_line != 0  :
        short = l.split(',')
        short = [float(i) for i in short]
        x.append(short)
    # 第0列沒有資訊
    
    n_line = n_line+1
text.close()


x = np.array(x)
path_y = sys.argv[2]
text_y = open('y_train', 'r', encoding='big5')
#lines = text.read().split('\n,')
y_lines=text_y.readlines()
y_content = [x.strip() for x in y_lines]
y=[]
n_line = 0
for l in y_content:
    if n_line != 0  :
        y.append(float(l))
    # 第0列沒有資訊
    
    n_line = n_line+1
text_y.close()

y = np.array(y)

addlist = (0,1,4,5,15,16,17,18,19,20,21)
# add square term
rep = x
x = np.concatenate((x,rep[:,addlist]**2), axis=1)
#x = np.concatenate((x,rep[:,9:]**3), axis=1)


# 標準化
mean = np.mean(x,axis=0)
std = np.std(x,axis=0)
x = ((x - mean)/std)


# Train 
"""
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

## k-fold
k_part = []
k_num = 10
num = len(x) // k_num

rlist = range(len(x))
for i in range(k_num):
     part = random.sample(rlist,num)
     rlist = [x for x in rlist if x not in part]
     k_part.append(part)
l_rate = 0.001
repeat = 10000
lamda = 0.001
w_option = {}
w_error = {}
alist = range(k_num)


for k in range(k_num):
    
    #k = 0
    ## Train
    train_list = []
    train = [x for x in alist if x is not k]
    
    for j in train:
        train_list.extend(k_part[j])
    
    train_data = x[train_list]
    correct = y[train_list]
    ## val
    val_list = k_part[k]
    val_data = x[val_list]
    val_correct = y [val_list]
    
    batch_size = 100
    train_data_size = len(train_data)
    step_num = int(math.floor(train_data_size / batch_size))
    epoch_num = 1500
    save_param_iter = 50
    
    total_loss = 0.0
    
    w = np.zeros(len(train_data[0]))
    train_data_t = train_data.transpose()
    s_gra = np.zeros(len(train_data[0]))
    
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0

        # Random shuffle
        X_train, Y_train = _shuffle(train_data,correct)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w))
            y_ = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y_)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y_)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y_).reshape((batch_size,1)), axis=0)
            #s_gra += w_grad**2
            #ada = np.sqrt(s_gra)
            #b_grad = np.mean(-1 * (np.squeeze(Y) - y_))

            # SGD updating parameters
            w = w - l_rate * w_grad
    
    
    w_option[k] = w
    # val
    predict = sigmoid(np.dot(val_data,w))
    val_ans = np.around(predict)
    
    result = (np.squeeze(val_ans) == val_correct)
    print('Validation acc = %f' % (float(result.sum()) / len(val_data)))
    
    #error  = -1 * (np.dot(np.squeeze(Y), np.log(y_)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y_)))/ len(val_data)
    error  = (float(result.sum()) / len(val_data))
    w_error[k] = error
 
order = sorted(w_error.items(), key = lambda t : t[1],reverse=True)
best_w = w_option[order[0][0]]
print(w_error[order[0][0]])

# save model
np.save('model_hw2_8.npy',best_w)
"""

# read model
best_w = np.load('model_hw2_6.npy')

test_x = []
n_row = 0
path_test = sys.argv[3]
text = open(path_test ,"r")
        
lines=text.readlines()
content = [x.strip() for x in lines]
test_x=[]
n_line = 0
for l in content:
    if n_line != 0  :
        short = l.split(',')
        short = [float(i) for i in short]
        test_x.append(short)
    # 第0列沒有資訊
    
    n_line = n_line+1
text.close()

test_x = np.array(test_x)

# add square term
#test_rep = test_x
#test_x = np.concatenate((test_x,test_rep[:,9:]**2), axis=1)
#test_x = np.concatenate((test_x,test_rep[:,9:]**3), axis=1)

# add square term
test_rep = test_x
test_x = np.concatenate((test_x,test_rep[:,addlist]**2), axis=1)
#test_x = np.concatenate((test_x,test_rep[:,9:]**3), axis=1)

# normalization

test_x = ((test_x - mean)/std)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    
    a = sigmoid(np.dot(best_w,test_x[i]))
    if a >= 0.5:
        answer = 1
    else:
        answer = 0
    ans.append([str(i+1),answer])
    #ans[i].append(a)

filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
