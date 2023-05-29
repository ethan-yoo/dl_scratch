import numpy as np

def AND(x1, x2):
    W = np.array([0.5, 0.5])
    X = np.array([x1, x2])
    b = -0.7
    tmp = np.sum(W*X) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):
    W = np.array([-0.5, -0.5]) # AND 게이트와 가중치(w, b)만 다름
    X = np.array([x1, x2])
    b = 0.7
    tmp = np.sum(W*X) + b
    if tmp <= 0:
        return 0
    else:
        return 1
        
def OR(x1, x2):
    W = np.array([0.5, 0.5]) # AND 게이트와 가중치(w, b)만 다름
    X = np.array([x1, x2])
    b = -0.2
    tmp = np.sum(W*X) + b
    if tmp <= 0:
        return 0
    else:
        return 1
        
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y