# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:00:13 2020

@author: yy
"""

# 使用CART实现手写数字分类
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# 读入数据集
digits = load_digits()
X, y = digits.data, digits.target

# 绘制手写数字图样
n=0
fig, axes=plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(15, 6))
for i in range(2):
    for j in range(5):
        axes[i][j].imshow(digits.images[n], cmap=plt.cm.gray)
        axes[i][j].set_title("Number " + str(n))
        n += 1
plt.suptitle("The Image of Hand-writing Numbers")
plt.show()


# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=666)

# 建立决策树
cart = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=666)

# 训练模型
cart.fit(X_train, y_train)

# 模型预测
y_predict = cart.predict(X_test)

# 模型评价
score = accuracy_score(y_test, y_predict)
print('CART准确率: %0.4lf' % score)

# 绘制模型预测的confusion matrix
cart_cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10,10), dpi=80)
sns.heatmap(cart_cm, square=True, annot=True, cbar=False)


# 绘制决策树，查看错误判断节点
dot_data = export_graphviz(cart, out_file=None, 
                           #feature_names=digits.feature_names, 
                           class_names=list(map(str,digits.target_names)), 
                           filled=True)
graph = graphviz.Source(dot_data)
graph.render(r'C:\Users\lenovo\Desktop\cart')
