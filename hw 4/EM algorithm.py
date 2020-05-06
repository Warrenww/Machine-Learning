import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from scipy.stats import multivariate_normal as mvn
import time
import sys

def get_data(filename = 'train-images.idx3-ubyte'):
    with open(filename, "rb") as f:
        bytes = f.read()
        size = int.from_bytes(bytes[4:8],  byteorder='big', signed=False)
        # size //= 10
        data = np.zeros((size,28*28))
        f.seek(16)
        for i in range(size):
            data[i] = np.array([i//128 for i in f.read(28*28)])

    return data

def get_label(filename = 'train-labels.idx1-ubyte'):
    labels = list()
    with open(filename, "rb") as f:
        bytes = f.read()
        size = int.from_bytes(bytes[4:8],  byteorder='big', signed=False)
        # size //= 10
    for i in range(size):
        labels.append(bytes[i + 8])

    return np.array(labels)

def Estimatation():
    for n, image in enumerate(images):
        # print("Image {:5}/{:5}".format(n,len(images)), end='\r')
        p = lm.copy()
        for k in range(10):
            for i, pixel in enumerate(image):
                if pixel:
                    p[0,k] = p[0,k] * (0.0001 if P[i,k] == 0 else P[i,k])
                else:
                    p[0,k] = p[0,k] * (0.0001 if P[i,k] == 1 else (1 - P[i,k]))

        W[n] = p / (0.0001 if np.sum(p) == 0 else np.sum(p))

    return W

def Maximization():
    s = np.sum(W, axis=0)
    lm = s / len(images)
    P = images.transpose()*W
    for k in range(10):
        P[:,k] = P[:,k] / (0.0001 if s[0,k] == 0 else s[0,k])

    return lm, P

def log_iterations(i, diff):
    for k in range(10):
        print('class {}: \n {} \n'.format(k,np.array([0 if i < 0.5 else 1 for i in P[:,k]]).reshape(28,28)))
    print('\nNo. of Iteration: {:3}, Difference: {:7f}'.format(i, diff))
    print('\n--------------------------------')

def assign_label():
    table = np.zeros((10,10))
    for n, image in enumerate(images):
        # print("Image {:5}/{:5}".format(n,len(images)), end='\r')
        p = lm.copy()
        for k in range(10):
            for i, pixel in enumerate(image):
                if pixel:
                    p[0,k] = p[0,k] * (0.0001 if P[i,k] == 0 else P[i,k])
                else:
                    p[0,k] = p[0,k] * (0.0001 if P[i,k] == 1 else (1 - P[i,k]))

        # table[int(labels[n]), np.argmax(np.array([np.sum(abs(P[:,k] - image)) for k in range(10)]))] += 1
        table[int(labels[n]), np.argmax(p)] += 1

    # print(table)
    np.save('./dump/L2C_table',table)
    for k in range(10):
        index = np.unravel_index(np.argmax(table), table.shape)
        label2cluster[index[0]] = index[1]
        table[index[0],:] = -1
        table[:,index[1]] = -1

    return label2cluster
    # return np.argmax(table, axis = 1)

def print_confusion_matrix():
    confusion_matrix = np.zeros((10,3))
    error = 0
    for n, w in enumerate(W):
        # print("Image {:5}/{:5}".format(n,len(images)), end='\r')
        index = np.argmax(w)
        if(int(labels[n]) == index):
            confusion_matrix[int(labels[n]),0]+=1
        else:
            confusion_matrix[int(labels[n]),1]+=1
            confusion_matrix[index,2]+=1

    for k in range(10):
        print("""
Confusion Matrix {cluster}:
                        Predict number {cluster} Predict not number {cluster}
Is number {cluster}     {:24} {:24}
Isn't number {cluster}  {:24} {:24}
Sensitivity (Successfully predict number {cluster}): {:.5f}
Specificity (Successfully predict not number {cluster}): {:.5f}
----------------------------------------------------------------
        """.format(
        confusion_matrix[k,0],
        confusion_matrix[k,1],
        confusion_matrix[k,2],
        60000 - np.sum(confusion_matrix[k]),
        confusion_matrix[k,0]/(confusion_matrix[k,0] + confusion_matrix[k,1]),
        (60000 - np.sum(confusion_matrix[k]))/(confusion_matrix[k,2] + 60000 - np.sum(confusion_matrix[k])),
        cluster = k))
        error += confusion_matrix[k,1]+confusion_matrix[k,2]

    return error

if __name__ == '__main__':
    images = get_data()
    print('Images loaded')
    labels = get_label()
    print('Labels loaded')
    lm = np.matrix(np.ones(10)) / 10
    W = np.matrix(np.zeros((len(images),10)))
    P = np.matrix(np.random.rand(28*28,10))
    label2cluster = np.zeros(10)

    iteration = 0

    # while True:
    while True:
        iteration += 1
        P_prev = P.copy()
        # print('Estimatation !   ')
        W = Estimatation()
        # print('Maximization !   ')
        lm, P = Maximization()
        diff = np.sum(abs(P_prev - P))
        log_iterations(iteration, diff)
        np.savez("./dump/iteration {}.npz".format(iteration), W = W, P = P, lm = lm)
        if diff < 10: break

    # print('Assign labels !   ')
    label2cluster = assign_label()
    print(label2cluster)
    error_rate = print_confusion_matrix()
    print("""
Total iteration to converge: {}
Total error rate: {}
    """.format(iteration, error_rate))
