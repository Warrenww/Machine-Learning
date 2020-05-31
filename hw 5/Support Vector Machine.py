import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
import sys

def get_data(filename):
    result = list()
    with open(filename, 'r') as f:
        c = 0
        for line in  f.readlines():
            print("Reading datas {}         ".format(c), end="\r")
            c += 1
            result.append(np.array([float(i) for i in line.split(",")]))
    return np.array(result)

def get_label(filename):
    result = list()
    with open(filename, 'r') as f:
        c = 0
        for line in  f.readlines():
            print("Reading labels {}        ".format(c), end="\r")
            c += 1
            result.append(int(line))
    return np.array(result)

if __name__ == '__main__':


    if len(sys.argv) > 1:
        training_data = get_data('X_train.csv')
        training_labels = get_label('Y_train.csv').reshape(-1)

        testing_data = get_data('X_test.csv')
        testing_labels = get_label('Y_test.csv').reshape(-1)

        if sys.argv[1] == '-p1':
            kernel_function_name = ['linear', 'polynomial', 'RBF']
            for i in range(3):
                print("Kernel function: {}".format(kernel_function_name[i]), end="\n\t")
                t0 = time.time()
                model = svm_train(training_labels, training_data, '-q -t ' + str(i))
                svm_predict(testing_labels, testing_data, model)
                print("\tTraining time: {}s".format(time.time() - t0))
        elif sys.argv[1] == '-p2':

            linear_kernel_results = list()
            polynomial_kernel_results = list()
            RBF_kernel_results = list()

            for c in range(-5,6,1):
                params = "-q -v 5 -c {}".format(2**c)
                acc = svm_train(training_labels, training_data, params + " -t 0")
                linear_kernel_results.append((acc, 2**c))

                for gamma in range(-5,6,1):
                    params += " -g {}".format(2**gamma)
                    acc = svm_train(training_labels, training_data, params + " -t 2")
                    RBF_kernel_results.append((acc, 2**c, 2**gamma))
                    for d in range(2,5,1):
                        for r in range(2):
                            params += " -d {} -r {}".format(d,r)
                            acc = svm_train(training_labels, training_data, params + " -t 1")
                            polynomial_kernel_results.append((acc, 2**c, 2**gamma, d, r))

            # with open('output.txt', 'w') as f:
            #     f.write(str(linear_kernel_results) + '\n')
            #     f.write(str(polynomial_kernel_results) + '\n')
            #     f.write(str(RBF_kernel_results) + '\n')

            # def parse(s):
            #     tmp_1 = ''
            #     tmp_2 = ''
            #     result = list()
            #     flag = False
            #     for i in s:
            #         # print(i, end='')
            #         if i == '[' or i == ']' or i == '\n' or i == ' ' : continue
            #         elif i == '(':
            #             tmp_1 = tuple()
            #             flag = True
            #         elif i == ',':
            #             # print('\n', flag, tmp_1, tmp_2)
            #             if flag:
            #                 tmp_1 += (float(tmp_2),)
            #                 tmp_2 = ''
            #             else:
            #                 result.append(tmp_1)
            #                 tmp_1= ''
            #         elif i == ')':
            #             tmp_1 += (float(tmp_2),)
            #             flag = False
            #             tmp_2 = ''
            #
            #         else:
            #             tmp_2 += i
            #     return result
            #
            #
            # with open('output.txt') as f:
            #     linear_kernel_results = parse(f.readline())
            #     polynomial_kernel_results = parse(f.readline())
            #     RBF_kernel_results = parse(f.readline())

            linear_kernel_results.sort(key = lambda x: x[0], reverse = True)
            polynomial_kernel_results.sort(key = lambda x: x[0], reverse = True)
            RBF_kernel_results.sort(key = lambda x: x[0], reverse = True)
            print(linear_kernel_results[0])
            print(polynomial_kernel_results[0])
            print(RBF_kernel_results[0])


            plt.figure(1)
            plt.ylabel('accuracy')
            plt.xlabel('C')
            plt.title('Relationship between C and accuracy for linear kernel')
            linear_kernel_results.sort(key = lambda x: x[1], reverse = True)
            plt.plot([c for acc,c in linear_kernel_results], [acc for acc,c in linear_kernel_results], label= 'linear')
            plt.savefig('linear')

            plt.figure(2)
            plt.ylabel('accuracy')
            plt.xlabel('C')
            plt.title('Relationship between C and accuracy for RBF kernel')
            for gamma in range(-5,6,1):
                tmp = [i for i in RBF_kernel_results if i[2] == 2**gamma]
                tmp.sort(key = lambda x: x[1], reverse = True)
                plt.plot([c for acc, c, g in tmp], [acc for acc, c, g in tmp], label='RFB gamma={}'.format(2**gamma))
            plt.legend()
            plt.savefig('RBF c vs acc')

            plt.figure(3)
            plt.ylabel('accuracy')
            plt.xlabel('Υ')
            plt.title('Relationship between Υ and accuracy for RBF kernel')
            for c in range(-5,6,1):
                tmp = [i for i in RBF_kernel_results if i[1] == 2**c]
                tmp.sort(key = lambda x: x[2], reverse = True)
                plt.plot([g for acc,c,g in tmp], [acc for acc,c,g in tmp], label='RFB c={}'.format(2**c))
            plt.legend()
            plt.savefig('RBF gamma vs acc')

            plt.figure(4,figsize=(10, 12), dpi=80)
            plt.subplots_adjust(hspace=.3)
            for gamma in range(-5,6,1):
                for d in range(2,5,1):
                    for r in range(2):
                        tmp = [i for i in polynomial_kernel_results if i[2] == 2**gamma and i[3] == d and i[4] == r]
                        plt.subplot(3,2,(d-2)*2+(r+1))
                        plt.title("d={} r={}".format(d, r))
                        plt.ylabel('accuracy')
                        plt.ylim(95,99)
                        plt.xlabel('C')
                        tmp.sort(key = lambda x: x[1], reverse = True)
                        plt.plot([c for acc, c, g, d, r in tmp], [acc for acc, c, g, d, r in tmp], label='RFB gamma={}'.format(2**gamma))

            plt.savefig('polynomial c vs acc')

            plt.figure(5,figsize=(10, 12), dpi=80)
            plt.subplots_adjust(hspace=.3)
            for c in range(-5,6,1):
                for d in range(2,5,1):
                    for r in range(2):
                        tmp = [i for i in polynomial_kernel_results if i[1] == 2**c and i[3] == d and i[4] == r]
                        plt.subplot(3,2,(d-2)*2+(r+1))
                        plt.title("d={} r={}".format(d, r))
                        plt.ylabel('accuracy')
                        plt.ylim(95,99)
                        plt.xlabel('Υ')
                        tmp.sort(key = lambda x: x[2], reverse = True)
                        plt.plot([g for acc, c, g, d, r in tmp], [acc for acc, c, g, d, r in tmp], label='RFB c={}'.format(2**c))

            plt.savefig('polynomial Υ vs acc')

            plt.show()
        elif sys.argv[1] == '-p3':
            training_data_size = training_data.shape[0]
            testing_data_size = testing_data.shape[0]
            gamma = 0.03125

            t0 = time.time()

            linear_kernel_training = np.dot(training_data, training_data.T)
            RBF_kernel_training = squareform(np.exp(- gamma * pdist(training_data, 'sqeuclidean')))
            linear_kernel_testing = np.dot(testing_data, testing_data.T)
            RBF_kernel_testing = squareform(np.exp(- gamma * pdist(testing_data, 'sqeuclidean')))

            training_kernel = np.hstack((np.arange(1, training_data_size + 1).reshape((-1,1)), np.add(linear_kernel_training, RBF_kernel_training)))
            testing_kernel = np.hstack((np.arange(1, testing_data_size + 1).reshape((-1,1)), np.add(linear_kernel_testing, RBF_kernel_testing)))

            model = svm_train(training_labels, training_kernel, '-q -t 4')
            svm_predict(testing_labels, testing_kernel, model)
            print("Training time: {}s".format(time.time() - t0))
    else:
        print("""
Usage:
    -p1: problem 1
    -p2: problem 2
    -p3: problem 3
        """)
