import numpy as np
from numpy import ndarray

def fix_vectors(a, b):
    if type(a) == float:
        a = [a]
    if type(b) == float:
        b = [b]
    shortest = min(len(a), len(b))
    a = a[:shortest]
    b = b[:shortest]
    return a, b


def add(a, b):
    a, b = check(a, b)
    return list(np.add(a, b))


def add_abs(a, b):
    return np.absolute(add(a, b))


def subtract(a, b):
    a, b = check(a, b)
    return list(np.subtract(a, b))


def subtract_abs(a, b):
    return np.absolute(subtract(a,b))


def multiply(a, b):
    a, b = check(a, b)
    return list(np.multiply(a, b))


def protectedDiv(a, b):
    x = []
    a, b = check(a, b)
    for i in range(len(a)):
        if b[i] == 0:
            x.append(1)
        else:
            x.append(a[i]/b[i])
    return list(x)


def analytic_quotient(a, b):
    x = []
    for i in range(len(a)):
        x.append(a[i]/np.sqrt(1+b[i]*b[i]))
    return list(x)


def concat1(a, b): ##[list, list], Array)
    a, b = check(a, b)
    return np.concatenate((a, b),axis=1)

def concat2(a, b):## concat2, [list, Array], Array
    a, b = check(a, b)
    return np.concatenate((a, list(b)),axis=1)


def concat3(a, b):###concat3, [Array, Array], Array)
    a, b = check(a, b)
    return np.concatenate((list(a), list(b)),axis=1)

def concat4(a, b):##
    return np.concatenate((a, b))

def min_v(a, b):
    a, b = fix_vectors(a, b)
    x = []
    for i in range(len(a)):
        x.append(min(a[i], b[i]))
    return x


def max_v(a, b):
    a, b = fix_vectors(a, b)
    x = []
    for i in range(len(a)):
        x.append(max(a[i], b[i]))
    return x


def if_v(a, b, c):
    if type(a) == float:
        a = [a]
    if type(b) == float:
        b = [b]
    if type(c) == float:
        c = [c]
    l = min(len(a), len(b), len(c))
    x = []
    for i in range(l):
        if a[i] >= 0:
            x.append(b[i])
        else:
            x.append(c[i])
    return x

class Array:
    def __init__(ndarray):
        pass


def check(a, b):
    if isinstance(a, list):
        a=np.array(a)
    if isinstance(b, list):
        b=np.array(b)
    if  len(a.shape) == 1:
        a = a.reshape(-1, 1)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)
    return a, b
