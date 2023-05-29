import numpy as np
import pickle
import sys, os
sys.path.append(r'C:\Users\JEJOON YOO\Desktop\prgm\dl_scratch\data')
from mnist import load_mnist

def step_function(x):
    y = x > 0
    return y.astype(np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    const = np.max(x)
    exp_a = np.exp(x-const)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def relu(x):
    y = np.maximum(0, x)
    return y


def init_network():
    network = {}
    network['W1'] = np.random.randn(2, 2)
    network['b1'] = np.random.randn(2, 2)
    network['W2'] = np.random.randn(2, 2)
    network['b2'] = np.random.randn(2, 2)
    network['W3'] = np.random.randn(2, 2)
    network['b3'] = np.random.randn(2, 2)
    
    return network

def forward(network, x, activation):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    l1 = np.dot(x, W1) + b1
    z1 = activation(l1)
    l2 = np.dot(z1, W2) + b2
    z2 = activation(l2)
    l3 = np.dot(z2, W3) + b3
    z3 = activation(l3)

    return z3


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(r"C:\Users\JEJOON YOO\Desktop\prgm\dl_scratch\data\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
