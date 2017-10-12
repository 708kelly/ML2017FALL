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

best_w = np.load('model.npy')

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
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
