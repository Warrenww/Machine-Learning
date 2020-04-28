import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from scipy.stats import multivariate_normal as mvn
import time
import sys

def get_image(index = 0, training = True):
    if training: filename = 'train-images.idx3-ubyte'
    else: filename = 't10k-images.idx3-ubyte'

    with open(filename, "rb") as f:
        f.seek(index*28*28+16)
        bytes = f.read(28*28)

    img = np.array([i for i in bytes]).reshape((28,28))
    return img

def get_data(filename = 'train-images.idx3-ubyte'):
    with open(filename, "rb") as f:
        bytes = f.read()
        size = int.from_bytes(bytes[4:8],  byteorder='big', signed=False)
        data = np.zeros((size,28*28))
        f.seek(16)
        for i in range(size):
            data[i] = np.array([i for i in f.read(28*28)])

    return data

def get_label(filename = 'train-labels.idx1-ubyte'):
    labels = list()
    with open(filename, "rb") as f:
        bytes = f.read()
        size = int.from_bytes(bytes[4:8],  byteorder='big', signed=False)
    for i in range(size):
        labels.append(bytes[i + 8])

    return np.array(labels)

def prior(n):
    assert (0 <= n and n < 10)
    counter = Counter(training_labels)
    return (counter[n] / sum(counter.values()))

def likelihood(img):
    result = np.zeros((10))
    for i in range(10):
        mean, var = means[i], vars[i]
        result[i] = mvn.logpdf(img, mean=mean, cov=var) + np.log(priors[i])
    return result

def predict(X, mode = 0):
    P = np.zeros((X.shape[0],10))
    if mode:
        for i in range(10):
            mean, var = means[i], vars[i]
            P[:,i] = mvn.logpdf(X, mean=mean, cov=var) + np.log(priors[i])
    else:
        for i in range(10):
            pmf = PMFs[i]
            x = np.floor(X / 8).astype('Int32')
            P[:,i] = np.log(pmf[np.arange(x.shape[1]),x]).sum(axis = 1)

    return P



def training(mode = 0,means = list(),vars = list(), PMFs = list()):
    if mode:
        for n in range(10):
            X = training_data[training_labels == n]
            mean = X.mean(axis = 0)
            var = X.var(axis = 0) + 1e-2
            means.append(mean)
            vars.append(var)
    else:
        for n in range(10):
            X = training_data[training_labels == n]
            X = np.floor(X / 8)
            pmf = np.ones((X.shape[1],32))
            for i in range(X.shape[1]):
                unique, counts = np.unique(X[:,i], return_counts=True)
                for z in zip(unique,counts):
                    pmf[i,int(z[0])] = z[1]
            pmf = pmf / X.shape[0]
            PMFs.append(pmf)




mode = int(sys.argv[5])
t0 = time.time()
training_data = get_data(sys.argv[1])
testing_data = get_data(sys.argv[3])
training_labels = get_label(sys.argv[2])
testing_labels = get_label(sys.argv[4])
priors = [prior(i) for i in range(10)]
means = list()
vars = list()
PMFs = list()
training(mode,means,vars,PMFs)
print("training time: {}s".format(time.time() - t0))


test_case = len(testing_data)
# test_case = 1000
error = 0
result = open("output.txt","w")
t0 = time.time()
for n,p in enumerate(predict(testing_data[:test_case], mode)):
    result.write("Posteirior:\n")
    for i,x in enumerate(p):
        result.write("{}: {}\n".format(i,x))

    result.write("Prediction: {}, Ans: {}\n\n".format(np.argmax(p), testing_labels[n]))
    print("{}/{}".format(n,test_case), end='\r')
    if not ((np.argmax(p) == testing_labels[n])): error += 1
print("testing time: {}s".format(time.time() - t0))


result.write("Imagination of numbers in Bayesian classifier:\n")
for i in range(10):
    prediction = means[i] if mode else PMFs[i].argmax(axis=1)
    threashold = [1 if i > 0 else 0 for i in prediction]
    result.write("{}:".format(i))
    for index,p in enumerate(threashold):
        if index % 28 == 0: result.write("\n")
        result.write("{} ".format(p))
    result.write("\n")

result.write("\n")
print("Error rate: {}".format(error/test_case))
result.write("Error rate: {}".format(error/test_case))
result.close()
