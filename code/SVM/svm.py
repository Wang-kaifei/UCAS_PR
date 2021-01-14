#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:18:19 2021

@author: kaifeiwang
"""

import numpy as np
import os
from libsvm.python.svmutil import *
from libsvm.python.svm import *
from libsvm.python.commonutil import *

def my_svm_model(features_path, model_path, param):
    y, x = svm_read_problem(features_path)
    model = svm_train(y, x, param)
    svm_save_model(model_path, model)

def svm_model_test(features_path, model):
    """测试模型"""
    yt, xt = svm_read_problem(features_path)
    p_label, p_acc, p_val = svm_predict(yt, xt, model)#p_label即为识别的结果
    #return ''.join(str(int(p)) for p in p_label)

if __name__ == '__main__':
    param = '-t 2 -c 4 -g 0.5'
    #param = svm_parameter('-t 2 -c 4 -g 0.5')
    model_path = './data/model'
    train_features_path = './data/train/feature.txt'
    my_svm_model(train_features_path, model_path, param)
    test_features_path = './data/test/feature.txt'
    model = svm_load_model(model_path)
    svm_model_test(test_features_path, model)
