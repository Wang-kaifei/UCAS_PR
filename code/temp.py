# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

import numpy as np
import matplotlib.pyplot as plt


A = [[0.1, 6.8, -3.5, 2.0, 4.1, 3.1, -0.8, 0.9, 5.0, 3.9],
     [1.1, 7.1, -4.1, 2.7, 2.8, 5.0, -1.3, 1.2, 6.4, 4.0],
     [1, 0, 0, 0]]
B = [[7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1],
     [4.2, -4.3, 0.0, 1.6, 1.9, -3.2, -4.0, -6.1, 3.7, -2.2],
     [0, 1, 0, 0]]
C = [[-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9],
     [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1],
     [0, 0, 1, 0]]
D = [[-2.0, -8.9, -4.2, -8.5, -6.7, -0.5, -5.3, -8.7, -7.1, -8.0],
     [-8.4, 0.2, -7.7, -3.2, -4.0, -9.2, -6.7, -6.4, -9.7, -6.3],
     [0, 0, 0, 1]]

def pre_stan(nodes, flag):
    """对点对进行预处理"""
    res = []
    for i in range(0, len(nodes[0])):
        #第二类样本需要取相反数
        newnode = [nodes[0][i], nodes[1][i], 1] if flag else [-nodes[0][i], -nodes[1][i], -1]
        res.append(newnode)
    return res

def pre_mse(nodes, flag):
    """多类MSE预处理"""    
    Y = []
    res = []
    len1 = [0, 8] if flag else [8, 10]  # 训练数据长度为8，测试数据长度为2
    for i in range(0, len(nodes)):
        for j in range(len1[0], len1[1]):
            newnode = [nodes[i][0][j], nodes[i][1][j], 1]
            Y.append(nodes[i][2])
            res.append(newnode)
    return Y, res
    

def batch_per(a, eta, nodes):
    """可变增量批处理修正方法,a是初始向量，eta是步长，nodes是标准化的训练样本"""
    flag = True
    step = 0  #记录迭代轮数
    while flag:
        step += 1
        flag = False
        y = np.zeros(3).reshape(-1,1)
        cnt = 0
        for node in nodes:
            node = np.array([node]).T  #列向量，代表一个点
            score = a.T.dot(node)
            if score[0][0] <= 0:  #找到一个错分样本点
                cnt += 1
                flag = True   #当本次迭代还有错分样本时，需要进行下一次迭代
                y = y + node  #错误累计
        a = a + eta * y
    return a, step


def Ho_Kashyap(a, b, eta, nodes, theta, step_max):
    """Ho_Kashyap求解margin并确定a"""
    Y = np.empty([len(nodes), 3]) #20 * 3每一行是一个训练样本
    for i in range(0, len(nodes)):
        Y[i] = nodes[i]
    step = 0
    for i in range(0, step_max):
        step += 1
        e = Y.dot(a) - b
        ee = 0.5 * (e + np.abs(e))
        b = b + 2 * eta * ee
        a = np.linalg.pinv(Y).dot(b)
        if((np.abs(e) < theta).all()):
            print(step)
            return a, b, sum(np.where(Y.dot(a) > 0, 0, 1)) / len(nodes), np.sum((Y.dot(a)-b) ** 2)
    print("No solution!")
    return a, b, sum(np.where(Y.dot(a) > 0, 0, 1)) / len(nodes), np.sum((Y.dot(a)-b) ** 2)

def mul_MSE(nodes_train, Y_train, lam, nodes_test, Y_test):
    Y_train = np.array(Y_train).T #4行n列，每一列代表一个样本的标签
    Y_test = np.array(Y_test).T
    X_train = np.array(nodes_train).T #3行n列，每一列代表一个样本
    I = np.eye(len(X_train))  # 3*3单位阵
    W = (np.linalg.inv(X_train.dot(X_train.T) + lam * I)).dot(X_train).dot(Y_train.T)
    error_train = 0
    for i in range(0, len(nodes_train)):
        score = (W.T).dot(nodes_train[i])
        if np.argmax(score) != np.argmax(Y_train[:, i]):
            error_train += 1
    error_test = 0
    for i in range(0, len(nodes_test)):
        score = (W.T).dot(nodes_test[i])
        if np.argmax(score) != np.argmax(Y_test[:, i]):
            error_test += 1
    print(1 - error_train / len(nodes_train))
    print(1 - error_test / len(nodes_test))
    return W

if __name__ == "__main__":

#    """MSE 多类扩展方法"""
#    label_train, data_train = pre_mse([A, B, C, D], True)
#    label_test, data_test = pre_mse([A, B, C, D], False)
#    mul_MSE(data_train, label_train, 0.0001, data_test, label_test)
    
    """Ho-Kashyap algorithm"""
    data = pre_stan(A, True) + pre_stan(C, False) #待分类样本
    a_start = np.array([[0, 0, 0]]).T  #初始向量列向量
    b_start = np.array([np.ones([len(data)])]).T  #初始列向量
    theta = np.array([[0.0001] * len(data)]).T
    step_max = 10000
    a_new, b_new, rate, error = Ho_Kashyap(a_start, b_start, 1, data, theta, step_max)
    print(rate, '\n', error)
    x = np.arange(-10, 10, 0.1)
    y = -a_new[2] / a_new[1] - (a_new[0] / a_new[1]) * x
    plt.plot(x, y)
    #样本点
    type1 = plt.scatter(A[0], A[1], alpha=0.6, c = 'green', label='\omiga1')
    type2 = plt.scatter(C[0], C[1], alpha=0.6, c = 'orange')
    plt.legend((type1, type2,), (u"第一类", u"第三类"), loc = 0)
    plt.savefig("/Users/kaifeiwang/Desktop/课程资料/模式识别/作业/assgnment3/3.png",dpi=500,bbox_inches = 'tight')
    plt.show()
    
    data = pre_stan(B, True) + pre_stan(D, False) #待分类样本
    a_new, b_new, rate, error = Ho_Kashyap(a_start, b_start, 1, data, theta, step_max)
    print(rate, '\n', error)
    x = np.arange(-10, 10, 0.1)
    y = -a_new[2] / a_new[1] - (a_new[0] / a_new[1]) * x
    plt.plot(x, y)
    #样本点
    type1 = plt.scatter(B[0], B[1], alpha=0.6, c = 'red', label='\omiga1')
    type2 = plt.scatter(D[0], D[1], alpha=0.6, c = 'grey')
    plt.legend((type1, type2,), (u"第二类", u"第四类"), loc = 0)
    plt.savefig("/Users/kaifeiwang/Desktop/课程资料/模式识别/作业/assgnment3/4.png",dpi=500,bbox_inches = 'tight')
    plt.show()
    
#    """batch perception"""
#    start = np.array([[0.0, 0.0, 0.0]]).T  #初始向量
#    data = pre_stan(A, True) + pre_stan(B, False) #待分类样本
#    a_new, step = batch_per(start, 1, data) #求权向量
#    print("迭代次数：", step)
#    #判别面
#    x = np.arange(-10, 10, 0.1)
#    y = -a_new[2] / a_new[1] - (a_new[0] / a_new[1]) * x
#    plt.plot(x, y)
#    #样本点
#    type1 = plt.scatter(A[0], A[1], alpha=0.6, c = 'green', label='\omiga1')
#    type2 = plt.scatter(B[0], B[1], alpha=0.6, c = 'red')
#    plt.legend((type1, type2,), (u"第一类", u"第二类"), loc = 0)
#    plt.savefig("/Users/kaifeiwang/Desktop/课程资料/模式识别/作业/assgnment3/1.png",dpi=500,bbox_inches = 'tight')
#    plt.show()
#    
#    data = pre_stan(B, True) + pre_stan(C, False) 
#    a_new, step = batch_per(start, 1, data)  #求权向量
#    print("迭代次数：", step)
#    #判别面
#    x = np.arange(-10, 10, 0.1)
#    y = -a_new[2] / a_new[1] - (a_new[0] / a_new[1]) * x
#    plt.plot(x, y)
#    #样本点
#    type1 = plt.scatter(B[0], B[1], alpha=0.6, c = 'red', label='\omiga1')
#    type2 = plt.scatter(C[0], C[1], alpha=0.6, c = 'orange')
#    plt.legend((type1, type2,), (u"第二类", u"第三类"), loc = 0)
#    plt.savefig("/Users/kaifeiwang/Desktop/课程资料/模式识别/作业/assgnment3/2.png",dpi=500,bbox_inches = 'tight')
#    plt.show()















