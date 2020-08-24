# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:00:24 2020

@author: yy
"""

from collections import Counter
import pandas as pd
import numpy as np

class CART:
    """创建CART树"""
    
    class Node:
        """节点类"""
        def __init__(self, name):
            self.name = name   # 节点名称，即分割特征
            self.splited_value = -1
            self.connections = {}   # 与子节点的链接

        def connect(self, label, node):
            self.connections[label] = node   # 连接子节点，以分割特征的属性值为key，value为子节点
    
    def __init__(self, dataset):
        """初始化树"""
        self.dataset = dataset   # 建树的数据集
        self.root = None   # 根节点

    def cal_gini(self, dataset):
        """
        计算数据集的基尼系数
        输入--
            dataset: 数据集，DataFrame格式
        输出--
            gini: 基尼系数值
        """
        gini = 1
        label_counts = Counter(dataset.iloc[:,-1])
        for label_count in label_counts.values():
            p = float(label_count/len(dataset))**2
            gini -= p
        return gini

    def split_dataset(self, dataset, col_index, split_value):
        """
        按照指定列和指定值对数据集进行分割
        输入--
            dataset: 数据集，DataFrame格式
            col_index: 分割列的索引序号
            split_value: 分割值
        输出--
            splited: 字典形式数据集分割结果，字典key为“left”和“right”两个，表示“小于等于”和“大于”分割值，
                     字典的value为对应条件下的数据子集
        """        
        splited = {}
        splited['left'] = dataset[(dataset.iloc[:,col_index] <= split_value)]   # 小于等于split_value的索引
        splited['right'] = dataset[(dataset.iloc[:,col_index] > split_value)]   # 大于split_value的索引
        return splited

    def get_best_split(self, dataset):
        """
        找到数据集的最佳分割
        输入--
            dataset: 数据集，DataFrame格式
        输出--
            best_col_index: 最佳分割特征的索引
            best_splited_result: 分割后的数据集
        """    
        best_Gini = np.inf   # 初始基尼系数值为正无穷
        best_col_index = -1   # 初始最佳分割特征的索引为-1
        best_splited_value = -1   # 初始最佳分割值为-1
        best_splited_result = None   # 初始最佳分割值为None
        
        for col_index in range(dataset.shape[1]-1):
            sort_index = dataset.iloc[:,col_index].argsort()
   
            for i in range(1, dataset.shape[0]):
            
                if dataset.iloc[sort_index[i-1],col_index] != dataset.iloc[sort_index[i],col_index]:
                        
                    split_value = (dataset.iloc[sort_index[i-1],col_index] + dataset.iloc[sort_index[i],col_index]) / 2    
            
                    splited = self.split_dataset(dataset, col_index, split_value)
             
                    temp_gini = 0
                    for value in splited.values():
                        temp_gini += self.cal_gini(value)
            
                    if temp_gini < best_Gini:
                        best_Gini, best_col_index, best_splited_value, best_splited_result = temp_gini, col_index, split_value, splited

        print("分割后的基尼系数值为{}，最佳分割特征为第{}个特征（即{}），最佳分割值为{}。".format(best_Gini, best_col_index, dataset.columns[best_col_index], best_splited_value))
        
        for key, value in best_splited_result.items():
            temp = value.reset_index(drop=True)
            best_splited_result[key] = temp.drop(columns = dataset.columns[best_col_index])
       
        return best_col_index, best_splited_value, best_splited_result

    def majority_label(self, dataset):
        """
        找到数量较多的标签值，作为叶节点的标签
        输入--
            dataset: 数据集
        输出--
            label: 标签值
        """
        
        label = dataset.iloc[:,-1].value_counts().index[0]
        return label

    def construct_tree(self):
        """首次最佳分割，建立根节点"""
        best_col_index, best_splited_value, best_splited_result = self.get_best_split(self.dataset)
        self.root = self.Node(best_col_index)   # 最佳分割特征的索引作为根节点名
        self.root.splited_value = best_splited_value
        self.__construct(self.root, best_splited_result)   # 继续建立树
    
    def __construct(self, parent_node, splited_result):
        """
        根据父节点及分割子集建立树
        输入--
            parent_node: 父节点
            splited_result: 父节点分割后结果
        输出--
            无
        """
        for key, result in splited_result.items():   # 遍历每个子集
            if len(set(result.iloc[:,-1])) == 1:   # 如果数据子集的标签均为一类值，生成叶节点
                name = result.iloc[0,-1]   # 叶节点名为标签值
                node = self.Node(name)
                parent_node.connect(key, node)   # 叶节点与父节点建立连接
                continue
            elif result.shape[1] == 1:   # 如果数据子集中仅剩标签列，生成叶节点
                name = self.majority_label(result)   # 叶节点名为标签列中数量较多的列值
                node = self.Node(name)
                parent_node.connect(key, node)   # 叶节点与父节点建立连接
                continue
            else:
                best_col_index, best_splited_value, best_splited_result = self.get_best_split(result)   # 进行最佳分割
                node = self.Node(best_col_index)   # 叶节点名为最佳分割特征
                node.splited_value = best_splited_value
                parent_node.connect(key, node)
                self.__construct(node, best_splited_result)   # 以分割节点作为父节点，递归构建树
    
    def print_tree(self, node, tabs):
        """打印决策树"""
        print(tabs + str(node.name))
        for value, child_node in node.connections.items():
            if value == 'left':
                print(tabs + "\t" + "(" + "<=" + str(node.splited_value) + ")")
            else:
                print(tabs + "\t" + "(" + ">" + str(node.splited_value) + ")")
            self.print_tree(child_node, tabs + "\t\t")
    
    def predict(self, test):
        """
        对测试集进行结果预测
        输入--
            test: 测试集，DataFrame格式
        输出--
            predict_labels: 预测结果，Series格式
        """
        labels = []
        for i in range(len(test)):
            sample = test.iloc[i,:-1]
            temp_node = self.root    
    
            label = self.__predict(sample, temp_node)
            labels.append(label)
        
        predict_labels = pd.Series(data=labels, index = test.index)
        return predict_labels
    
    
    def __predict(self, sample, temp_node):
        """
        对一组样本进行预测
        输入--
            sample: 测试集的一行，Series格式
        输出--
            label: 预测结果值
        """
        if temp_node.connections == {}:
            return temp_node.name
        value = sample[temp_node.name]
        if value <= temp_node.splited_value:
            temp_node = temp_node.connections['left']
        else:
            temp_node = temp_node.connections['right']
        label = self.__predict(sample, temp_node)
        return label
        
    def accuracy(self, test):
        '''
        预测准确率
        输入--
            test: 测试集，DataFrame格式
        输出--
            accuracy: 准确率值
        '''
        predict = self.predict(test)
        true = test.iloc[:,-1]
        acc_counts = [1 if i == j else 0 for i, j in zip(predict, true)]
        accuracy = sum(acc_counts) / len(test)
        
        return accuracy

if __name__ == "__main__":
    """CART示例"""
    dataset = pd.DataFrame(data=[[0, 0, 0, 0, 'no'],  # 数据集
                             [0, 0, 0, 1, 'no'],
                             [0, 1, 0, 1, 'yes'],
                             [0, 1, 1, 0, 'yes'],
                             [0, 0, 0, 0, 'no'],
                             [1, 0, 0, 0, 'no'],
                             [1, 0, 0, 1, 'no'],
                             [1, 1, 1, 1, 'yes'],
                             [1, 0, 1, 2, 'yes'],
                             [1, 0, 1, 2, 'yes'],
                             [2, 0, 1, 2, 'yes'],
                             [2, 0, 1, 1, 'yes'],
                             [2, 1, 0, 1, 'yes'],
                             [2, 1, 0, 2, 'yes'],
                             [2, 0, 0, 0, 'no']],
                       columns=['age', 'job', 'house', 'credit', 'result']) 


    tree = CART(dataset)
    tree.construct_tree()
    tree.print_tree(tree.root, "")
    
    test = dataset
    predict = tree.predict(test)
    tree.accuracy(test)
    
        
    


