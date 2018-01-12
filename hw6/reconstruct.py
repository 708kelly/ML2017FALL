#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:45:35 2018

@author: kelly
"""

#import os
import numpy as np
import sys

#os.chdir("/Users/kelly/Documents/大四/機器學習/Hw6")

file_name = sys.argv[1]

from skimage import io
#img = io.imread(file_name)
#from skimage import transform

dim = 600
def convert_small(f):
    rgb=io.imread(f)
    return rgb

"""
img_dir='./'+file_name+'/*.jpg'
coll = io.ImageCollection(img_dir)
io.imshow(coll[10])
"""

mat = []
for i in range(415):
    num = str(i )
    tmp = convert_small('./'+file_name+'/'+num+'.jpg')
    tmp = tmp.flatten()
    mat.append(tmp)
#io.imshow(mat[10].reshape(100,100,3))

mat =np.array(mat)

flat_num = dim * dim * 3
#img_f = mat.reshape(mat.shape[0],flat_num) 

X_mean = np.mean(mat,axis = 0)

X = (mat-X_mean).T

U, s, V = np.linalg.svd(X, full_matrices=False)

target = sys.argv[2]
target = io.imread(target)
#target = convert_small(target)
target_f = target.flatten()
T = target_f - X_mean

eigen_dim = 4
eigen_faces = U[:,:eigen_dim]

weights = np.dot(T, eigen_faces)
pics = np.dot( weights,eigen_faces.T)
pics += X_mean
M = pics
M -= np.min(M)
M /= np.max(M)
M = ( M * 255).astype(np.uint8)

recon = pics.reshape(dim,dim,3)
#io.imshow(recon)
io.imsave('reconstruction.jpg',recon.reshape(dim,dim,3), quality=100)


