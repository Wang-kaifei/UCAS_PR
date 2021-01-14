#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:46:47 2020

@author: kaifeiwang
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt



def sigmoid(x):
    """激活函数sigmoid"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """sigmoid求导"""
    return x * (1 - x)

def tanh(x):
    """激活函数tanh"""
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    """tanh求导"""
    return 1 - x ** 2

def rand(a, b):
	return (b - a) * random.random() + a     

def get_sample(fpath):
    """从文件中读取训练样本"""
    samples = []
    with open(fpath, 'r') as f:
        while True:
            line = f.readline().rstrip()
            if not line:
                break
            segs = line.split(' ')
            data = np.array([float(segs[0]), float(segs[1]), float(segs[2])])
            label = segs[3]
            samples.append((data, int(label)))
        f.close()
    return samples


    
class BP_net:
    def __init__(self, ni, nh, no):
        #网络结构
        self.ni = ni
        self.nh = nh
        self.no = no
        #w_i_h:输入层到隐藏层的初始连接权 nh * ni 维，对应每个输入结点到隐藏结点的连接权
        self.w_i_h = self.get_w(nh, ni)
        #w_h_o:隐藏层到输出层的初始连接权 no * nh 维，对应每个隐藏结点到输出结点的连接权
        self.w_h_o = self.get_w(no, nh)
        #输入层
        self.input = np.zeros([ni, 1])
        #隐藏层输入
        self.net_h = np.zeros([nh, 1])
        #隐藏层输出
        self.y = np.zeros([nh, 1])
        #输出层输入
        self.net_o = np.zeros([no, 1])
        #输出层输出
        self.z = np.zeros([no, 1])
        
    def get_w(self, m, n):
        """初始化连接权重"""
        w = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                w[i, j] = rand(-1, 1)
        return w
                
    def forward(self, sample):
        """前向传播"""
        self.input = sample[0].reshape(-1, 1)
        self.net_h = self.w_i_h.dot(self.input)
        self.y = tanh(self.net_h)
        self.net_o = self.w_h_o.dot(self.y)
        self.z = sigmoid(self.net_o)
    
    def backward(self, sample, yita):
        """反向传播"""
        #得到误差信号
        t = np.zeros(self.no).reshape(-1, 1)
        t[sample[1], 0] = 1
        delta_z = (t - self.z) * sigmoid_derivative(self.z) #c * 1维 对应元素相乘，每一个元素代表该输出结点收集到的信号
        delta_h = (self.w_h_o.T).dot(delta_z) * tanh_derivative(self.y) #nh * 1维 每一个元素代表该隐层结点收集到的信号
        #更新隐层到输出层权重
        d2 = yita * delta_z.dot(self.y.T) #c * nh 每一行代表到该输出结点的连接权更新值
        #更新输入层到隐层权重
        d1 = yita * delta_h.dot(self.input.T) #nh * d 每一行代表到该隐层结点的连接权更新值
        return d1, d2  #返回该样本对权重的更新
    
def Single_BP(samples, network:BP_net, max_step = 100000, yita = 0.1, theta = 1e-5): 
    """单样本方式更新权重"""
    d1 = np.zeros_like(network.w_i_h)
    d2 = np.zeros_like(network.w_h_o)
    err = []
    err_next = 10000
    for step in range(0, max_step):
        k = random.randint(0, len(samples) - 1)
        network.forward(samples[k])  #前向传播
        d1, d2 = network.backward(samples[k], yita) #反向传播
#        scale1 = d1 / network.w_i_h #权重更新率
#        scale2 = d2 / network.w_h_o
        network.w_i_h += d1 #更新权重
        network.w_h_o += d2 
        if step % 1000 == 0:
            error = test(samples, network)
            if error > err_next:
                print("??")
                print(error)
                print(err_next)
            err_next = error
            err.append(error)
#        if step % 99 == 0:
#            error1 = test(samples, network)
#        if step % 100 == 0:
#            error2 = test(samples, network)
#            if math.fabs((error2 - error1) / error1) < theta:
#                print("error1", '%0.30f' % error1)
#                print("error2", '%0.30f' % error2)
#                print("step", step + random.randint(0, 100))
#                return network
#        if math.fabs(max(scale1.min(), scale1.max(), key=abs)) < theta and math.fabs(max(scale2.min(), scale2.max(), key=abs)) < theta:
#            error = test(samples, network)
#            print("error", error)
#            print("step", step + 1)
#            return network
    error = test(samples, network)
    print("error", error)
    print("step", max_step)
    return network, err

def Batch_BP(samples, network:BP_net, max_batch = 3333, yita = 0.1, theta = 1e-5):
    """批量更新算法""" 
    err_next = 10000
    err = []
    d1 = np.zeros_like(network.w_i_h)
    d2 = np.zeros_like(network.w_h_o)
    dd1 = np.zeros_like(network.w_i_h)
    dd2 = np.zeros_like(network.w_h_o)
#    error1 = 0
#    error2 = 0
    for batch in range(0, max_batch):
        for k in range(0, len(samples)):
            network.forward(samples[k])  #前向传播
            d1, d2 = network.backward(samples[k], yita) #反向传播
            dd1 += d1 #存储权重
            dd2 += d2 
            
#        scale1 = dd1 / network.w_i_h  #权重更新率
#        scale2 = dd2 / network.w_h_o
        
        network.w_i_h += dd1 #更新权重
        network.w_h_o += dd2 
        
#        if math.fabs(max(scale1.min(), scale1.max(), key=abs)) < theta and math.fabs(max(scale2.min(), scale2.max(), key=abs)) < theta:
#            error = test(samples, network)
#            print("err22123213or", error)
#            print("error", error)
#            print("batch", batch + 1)
#            return network
        dd1 = np.zeros_like(network.w_i_h)
        dd2 = np.zeros_like(network.w_h_o)
        if batch % 33 == 0:
            error = test(samples, network)
            if error > err_next:
                print("??")
                print(error)
                print(err_next)
            err_next = error
            err.append(error)
#        if batch % 9 == 0:
#            error1 = test(samples, network)
#        if batch % 10 == 0:
#            error2 = test(samples, network)
#            if math.fabs((error2 - error1) / error1) <= theta:
#                print("error1", '%0.30f' % error1)
#                print("error2", '%0.30f' % error2)
#                print("step", batch + random.randint(0, 100))
#                return network
    error = test(samples, network)
    print("error", error)
    print("step", max_batch)
    return network, err
   
def test(samples, network : BP_net):
    """测试函数，返回mse和错分样本数"""
    error = 0
#    case = 0
    for sample in samples:
        network.forward(sample)
        #该样本的目标值 c * 1向量
#        pre = np.argmax(network.z)
#        if pre != sample[1]:
#            case += 1
        t = np.zeros(network.no).reshape(-1, 1)
        t[sample[1], 0] = 1
        error += np.sum(np.square(t - network.z)) #误差（目标）函数的导数
    error = error / (2 * len(samples))
    return error
         
#if __name__ == "__main__":
#    fpath = "/Users/kaifeiwang/Desktop/1_data.txt"
#    yy = 0.1
#    hidden = 8
#    network = BP_net(3, hidden, 3)
#    network = Single_BP(get_sample(fpath), network, yita = yy)  
#    
#    network2 = BP_net(3, hidden, 3)
#    network2 = Batch_BP(get_sample(fpath), network2, yita = yy)    
    
if __name__ == "__main__":
    fpath = "/Users/kaifeiwang/Desktop/1_data.txt"
    yy = 0.001
    hidden = 8
    network = BP_net(3, hidden, 3)
    network, y1 = Single_BP(get_sample(fpath), network, yita = yy)  
    
    network2 = BP_net(3, hidden, 3)
    network2, y2 = Batch_BP(get_sample(fpath), network2, yita = yy)
    node1 = []
    node2 = []
    for i in range(0, 100000, 1000):
        node1.append(i)
    for i in range(0, 3333, 33):
        node2.append(i)
    plt.plot(node1, y1, 'r')
    plt.xlabel('Iteration number (Single step)')
    plt.ylabel('Loss function value')
    plt.savefig("/Users/kaifeiwang/Desktop/filename.png", dpi=300)
    plt.show()
    
    plt.plot(node2, y2)
    plt.xlabel('Iteration number (Batch)')
    plt.ylabel('Loss function value')
    plt.savefig("/Users/kaifeiwang/Desktop/filename2.png", dpi=300)
    plt.show()

 
#x = np.arange(0.6, 2, 0.01)
#lenth = len(x)
#y = np.empty(x.shape, dtype = float) 
#
#for i in range(lenth):
#    y[i] = 1 / math.pow(x[i], 5)
#
#plt.scatter([0.6], [12.86], s=25, c='r')
#plt.text(0.6+0.15, 12.86+0.15, '(0.6, 12.86)', ha='center', va='bottom', fontsize=10.5)
#
#plt.xlabel('θ')
#plt.ylabel('$p(\mathcal{D}|θ)$')
#plt.axis([0, 1.5, 0, 14])
#
#plt.plot(x, y)
#plt.savefig("/Users/kaifeiwang/Desktop/filename.png")
#plt.show()

    
    
    
    
    