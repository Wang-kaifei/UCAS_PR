#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:18:19 2021

@author: kaifeiwang
"""
import cv2
import numpy as np
import os

def get_feature(filepath, label):
    files= os.listdir(filepath) #得到文件夹下的所有文件名称
    features = []
    winSize = (8,8)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    winStride = (8,8)
    padding = (8,8)
    for file in files: #遍历文件夹
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            img = cv2.imread(filepath+"/"+file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
            test_hog = hog.compute(img, winStride, padding).reshape((-1,))
            temp = ' '.join([str(label)]  +  [str(i)+':'+str(j) for (i,j) in zip(range(1,len(test_hog)+1), test_hog)])
            features.append(temp)
    return features

def write_feature(features, filepath):
    with open(filepath,'w') as fp:
        for feature in features:
            fp.write(feature + '\n')


if __name__ == '__main__':
    Train_path0 = './data/train/0/'
    Train_path1 = './data/train/1/'
    Test_path0 = './data/test/0/'
    Test_path1 = './data/test/1/'
    train_feature_path = './data/train/feature.txt'
    test_feature_path = './data/test/feature.txt'
    #每一行代表一个特征，其中的第一个数字代表label
    train_feature_0 = get_feature(Train_path0, 0)
    train_feature_1 = get_feature(Train_path1, 1)
    write_feature(train_feature_0 + train_feature_1, train_feature_path)
    test_feature_0 = get_feature(Test_path0, 0)
    test_feature_1 = get_feature(Test_path1, 1)
    write_feature(test_feature_0 + test_feature_1, test_feature_path)