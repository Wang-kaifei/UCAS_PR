# SVM作业说明
## 题目
从MNIST数据集中任意选择两类，对其进行SVM分类，可调用现有的SVM工具如LIBSVM，展示超参数C以及核函数参数的选择过程。
## 程序说明
首先需要对MNIST数据集进行解码，修改了blog https://blog.csdn.net/panrenlong/article/details/81736754 的代码，选择类别0和1存储。这部分对应代码data_process.py

然后使用OpenCV在相同参数下提取训练集和测试集的HOG特征并分别存储成LIBSVM要求的格式。这部分对应代码build_feature.py
>格式要求为：
>label 1:feature1 2:feature2 3:feature3 ...

最后使用LIBSVM工具训练和测试。

参数选择使用LIBSVM自带的网格参数寻优脚本grid.py。
