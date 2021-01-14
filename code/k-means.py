#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:18:19 2021

@author: kaifeiwang
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def build_data():
    num = 200
    cov = [[1, 0], [0, 1]]
    mu1 = [1, -1]
    x1 = np.random.multivariate_normal(mu1, cov, num)
    mu2 = [5.5, -4.5]
    x2 = np.random.multivariate_normal(mu2, cov, num)
    mu3 = [1, 4]
    x3 = np.random.multivariate_normal(mu3, cov, num)
    mu4 = [6, 4.5]
    x4 = np.random.multivariate_normal(mu4, cov, num)
    mu5 = [9, 0]
    x5 = np.random.multivariate_normal(mu5, cov, num)
    
    X= np.concatenate((x1, x2, x3, x4, x5), axis=0)
    plt.scatter(x1[:, 0], x1[:, 1], color='r')
    plt.scatter(x2[:, 0], x2[:, 1], color='b')
    plt.scatter(x3[:, 0], x3[:, 1], color='k')
    plt.scatter(x4[:, 0], x4[:, 1], color='g')
    plt.scatter(x5[:, 0], x5[:, 1], color='m')
    plt.savefig("dots.png", dpi=300)
    plt.show()
    np.save('data', X) 


class K_means:

    def __init__(self, datafile = "", k = 5, random = 1):
        self.k = k
        if datafile == "":
            build_data()
            self.nodes = np.load('data.npy')
        else:
            self.nodes = np.load(datafile)
        self.nodes = np.insert(self.nodes, 2, 0, axis=1)
        self.centers = np.zeros((k, 3)) #类中心
        if random == 1:   #随机在样本点中选择初始类中心
            np.random.shuffle(self.nodes)
            for i in range(0, k):
                self.centers[i][0] = self.nodes[i][0]
                self.centers[i][1] = self.nodes[i][1]
                self.centers[i][2] = i
        elif random == 2: #分别在每一类中选一点作为初始中心
            for i in range(0, k):
                self.centers[i][0] = self.nodes[i * 200][0]
                self.centers[i][1] = self.nodes[i * 200][1]
                self.centers[i][2] = i
        elif random == 3: #中心点在第一类中选择一个，然后选择附近点
            for i in range(0, k):
                self.centers[i][0] = self.nodes[i][0]
                self.centers[i][1] = self.nodes[i][1]
                self.centers[i][2] = i
        elif random == 4: #中心点在第一类中选择一个，然后选择附近点
            self.centers[0][0] = self.nodes[0][0]
            self.centers[0][1] = self.nodes[0][1]
            self.centers[0][2] = 0
            for i in range(1, k):
                self.centers[i][0] = self.centers[i - 1][0] + 0.1
                self.centers[i][1] = self.centers[i - 1][1] + 0.1
                self.centers[i][2] = i

    def iterate(self):
        flag = True
        cnt = 0
        while(flag):
            flag = False
            cnt += 1
            temp_center = np.zeros((self.k, 3))
            for i in range(0, np.size(self.nodes, 0)):
                dis = np.zeros(self.k)
                for j in range(0, np.size(self.centers, 0)):
                    dis[j] = pow((self.nodes[i][0] - self.centers[j][0]), 2) + pow((self.nodes[i][1] - self.centers[j][1]), 2)
                kk = np.argmin(dis)  #分到该类
                self.nodes[i][2] = kk
                temp_center[kk][0] += self.nodes[i][0]
                temp_center[kk][1] += self.nodes[i][1]
                temp_center[kk][2] += 1
            #计算新类中心
            for i in range(0, np.size(self.centers, 0)):
                temp_x = temp_center[i][0] / temp_center[i][2]
                temp_y = temp_center[i][1] / temp_center[i][2]
                if temp_x != self.centers[i][0] or temp_y != self.centers[i][1]:
                    flag = True
                self.centers[i][0] = temp_x
                self.centers[i][1] = temp_y
        print("The iterate has been convergented after %d rounds" %cnt)
        #将样本点按照分类划分
        self.nodes_ps = []
        for i in range(0, self.k):
            self.nodes_ps.append([])
        for i in range(0, np.size(self.nodes, 0)):
            self.nodes_ps[int(self.nodes[i][2])].append(self.nodes[i])
        
    def evaluate(self, base):
        """第一步计算类内方差之和，第二步计算类中心mse"""
        J = 0
        for i in range(0, self.k):
            for j in range(0, np.size(self.nodes_ps[i], 0)):
                J += pow(self.centers[i][0] - self.nodes_ps[i][j][0], 2) + pow(self.centers[i][1] - self.nodes_ps[i][j][1], 2)
        print("J: %f" %J)
        flag = np.zeros(np.size(self.centers, 0))  #标记base中的点是否已选
        for i in range(np.size(self.centers, 0)):
            dis_min = 10000
            temp = -1
            for j in range(np.size(base, 0)):
                dis = pow(base[j][0] - self.centers[i][0], 2) + pow(base[j][1] - self.centers[i][1], 2)
                if  dis < dis_min and flag[j] == 0:
                    temp = j
                    dis_min = dis
            self.centers[i][2] = temp
            flag[temp] = 1
        self.centers = sorted(self.centers, key = lambda x : x[2])
        mse = 0
        for i in range(np.size(self.centers, 0)):
            mse += pow(self.centers[i][0] - base[i][0], 2) + pow(self.centers[i][1] - base[i][1], 2)
            print("center %d : %f  %f"  %(i, self.centers[i][0], self.centers[i][1]))
            print("base %d : %f  %f"  %(i, base[i][0], base[i][1]))
        print("mse: %f" %mse)
    
    def show(self):
        """聚类结果可视化"""
        colors = ['r', 'b', 'k', 'g', 'm']
        for i in range(0, self.k):
            nodes = np.array(self.nodes_ps[i])
            plt.scatter(nodes[:, 0], nodes[:, 1], color = colors[i])
        print(np.shape(self.centers))
        self.centers = np.array(self.centers)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], color = 'y', marker = '+')
        plt.savefig("after4.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    depart = K_means(datafile = "data.npy", random = 4)
    depart.iterate()
    base = np.array([[1, -1], [5.5, -4.5], [1, 4], [6, 4.5], [9, 0]])
    depart.evaluate(base)
    depart.show()








