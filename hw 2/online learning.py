import sys
from collections import Counter
from scipy.stats import beta
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
import math

with open('testfile.txt','r') as test_data:
    data = test_data.read().split("\n")

data.pop()
a = int(sys.argv[1])
b = int(sys.argv[2])
counter = Counter({'1':a, '0':b})
prior = 0.5
for index,test in enumerate(data):
    print("case {}: {}".format(index + 1, test))

    print("Beta prior: a = {} b = {}".format(counter['1'],counter['0']))
    temp = Counter(test)
    counter.update(temp)
    print("Likelihood: {}".format(binom.pmf(temp['1'],sum(temp.values()),prior)))
    prior = beta.mean(counter['1'],counter['0'])
    print("Beta posterior: a = {} b = {}\n".format(counter['1'],counter['0']))
