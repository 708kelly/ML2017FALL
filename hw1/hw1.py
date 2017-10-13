#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:57:57 2017

@author: kelly
"""

import csv 
import numpy as np
import random
import math
import sys

#TrainData
"""
data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])
n_row = 0

var_list =[8,9]
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
# 取前幾個小時
time_range = 9
train_time = 20 * 24 - time_range


for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(train_time):
        x.append([])
        # 18種污染物
        for t in var_list:
            # 連續9小時
            for s in range(time_range):
                x[train_time*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+time_range])
x = np.array(x)
y = np.array(y)



# add square term
rep = x
x = np.concatenate((x,rep[:,9:]**2), axis=1)
#x = np.concatenate((x,rep[:,9:]**3), axis=1)

# normalization
mean = np.mean(x,axis=0)
std = np.std(x,axis=0)
x = ((x - mean)/std)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)




## k-fold
k_part = []
k_num = 3
num = len(x) // k_num

rlist = range(len(x))
for i in range(k_num):
     part = random.sample(rlist,num)
     rlist = [x for x in rlist if x not in part]
     k_part.append(part)

    

l_rate = 1
repeat = 20000

w_option = {}
w_error = {}
alist = range(k_num)
for k in range(k_num):
    
    
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
    
    w = np.zeros(len(train_data[0]))
    train_data_t = train_data.transpose()
    s_gra = np.zeros(len(train_data[0]))
    for i in range(repeat):
        hypo = np.dot(train_data,w)
        loss = hypo - correct
        cost = np.sum(loss**2) / len(train_data)
        cost_a  = math.sqrt(cost)
        gra = np.dot(train_data_t,loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        if i != (repeat-1):
            w = w - l_rate * gra/ada
        #else:
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
        
    w_option[k] = w
    # val
    predict = np.dot(val_data,w)
    val_loss = predict - val_correct
    val_cost = np.sum(val_loss**2) / len(val_data)
    error  = math.sqrt(cost)
    w_error[k] = error
 
order = sorted(w_error.items(), key = lambda t : t[1])
best_w = w_option[order[0][0]]
print(w_error[order[0][0]])
# save model
np.save('model.npy',best_w)
"""
var_list =[8,9]
best_w = np.load('hw1.npy')
time_range = 9

test_x = []
n_row = 0
path = sys.argv[1]
text = open(path ,"r")
row = csv.reader(text , delimiter= ",")
        
for r in row:
    if n_row %18 == 0:
            test_x.append([])
    if n_row % 18 in var_list: 
        for i in range((2+(9-time_range)),11):
            #print(i)
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
test_rep = test_x
test_x = np.concatenate((test_x,test_rep[:,9:]**2), axis=1)
#test_x = np.concatenate((test_x,test_rep[:,9:]**3), axis=1)
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(best_w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
