# 神经网络

Fundamentals: This includes understanding the structure of a neural network, such as layers, weights, biases, and activation functions (sigmoid, tanh, ReLU, etc.)

Training and Optimization: Familiarize yourself with backpropagation and different types of loss functions, like Mean Squared Error (MSE) and Cross-Entropy. Understand various optimization algorithms like Gradient Descent, Stochastic Gradient Descent, RMSprop, and Adam.

Overfitting: Understand the concept of overfitting (where a model performs well on training data but poorly on unseen data) and learn various regularization techniques (dropout, L1/L2 regularization, early stopping, data augmentation) to prevent it.

Implement a Multilayer Perceptron (MLP): Build an MLP, also known as a fully connected network, using PyTorch.

## 1.神经元模型

M-P神经元模型

激活函数

学习率

权重初始化：常数初始化，随机分布初始化，Xavier初始化，He初始化

## 2.感知机与多层网络（MLP）

多层前馈神经网络（Multi-Layer FFN）

前馈并不意味着网络信号不能向后传，而是指网络拓扑结构上不存在环或回路。

## 3.反向传播算法（BP）

LMS（Least Mean Square）算法（BP前身）：将LMS推广到由非线性可微神经元组成的多层前馈网络，就得到BP算法。

梯度下降GD

随机梯度下降SGD

解决过拟合：早停，正则化（dropout，标签平滑，权重衰减）

## 4.全局最小与局部最小

跳出局部最小：启发式算法（模拟退火，遗传算法等）

## 5.其他神经网络

RBF网络

ART网络

SOM网络

级联相关网络

Elman网络

Boltzman机

深度学习（pre-training,fine-tuning）

深度信念网络DBN

权值共享CNN

## 6.深度学习
