import matplotlib.pyplot as plt
import numpy as np
import sys

X = list()
Y = list()

def readData(filename):
    f = open(filename)
    line = f.readline()
    while line:
        X.append(float(line.split(',')[0]))
        Y.append(float(line.split(',')[1]))
        line = f.readline()

class Regression(object):
    def __init__(self, n):
        self.n = n
        self.X = None
        self.Y = None
        self.dm = None
        self.weight = None
        self.predict = None
        self.error = 0

    def design_matrix(self):
        result = list()
        for x in self.X:
            temp = list()
            for i in range(self.n):
                temp.append(x**i)
            result.append(temp)
        return np.matrix(result)

    def inverse(self, matrix):
        size = matrix.shape[0]
        result = np.matrix(np.zeros(matrix.shape))

        for i in range(size):
            temp = self.solver(matrix, np.matrix([(1 if i == j else 0) for j in range(size)]).transpose())
            result[:, i] = temp

        return result

    def solver(self, A,b):
        # To solve equation Ax=b
        size = b.shape[0]
        temp = np.matrix(np.zeros(b.shape))
        result = np.matrix(np.zeros(b.shape))
        (L,U) = self.LU_decompose(A)

        for i in range(size):
            temp[i] = (b[i] - sum(L[i, j] * temp[j] for j in range(i))) / L[i, i]

        for i in range(size-1, -1, -1):
            result[i] = temp[i] - sum(U[i, j] * result[j] for j in range(i, size))

        return result

    def LU_decompose(self, matrix):
        size = matrix.shape[0]
        L_matrix = np.matrix(np.zeros(matrix.shape))
        U_matrix = np.matrix(np.zeros(matrix.shape))
        for i in range(size):
            for k in range(i, size):
                L_matrix[k, i] = matrix[k, i] - sum(L_matrix[k, j] * U_matrix[j, i] for j in range(i))
            for k in range(i, size):
                U_matrix[i, k] = (matrix[i, k] - sum(L_matrix[i, j] * U_matrix[j, k] for j in range(i))) / L_matrix[i, i]

        return (L_matrix, U_matrix)

    def plot(self, i = 0):
        if self.X:
            plt.subplot(2, 1, i)
            plt.plot(self.X, self.Y,'ro')
            plt.plot(self.X, self.predict)
            plt.title(self.name)

    def report(self):
        equation = [str(w[0,0])+("*x^"+str(i) if i > 0 else "") for i,w in enumerate(self.weight)]
        print("""
{}
  Fitting line: {}
  Total error: {}
        """.format(self.name, "+".join(equation), self.error))


class r_LSE(Regression):
    def __init__(self, n, l):
        super(r_LSE, self).__init__(n)
        self.l = l
        self.name = "LSE"

    def calculate(self, x, y):
        self.X = x
        self.Y = y
        self.dm = self.design_matrix()
        self.weight = self.inverse(self.dm.transpose()*self.dm + self.l*np.identity(self.n))*self.dm.transpose()*np.matrix(y).transpose()
        self.predict = self.dm*self.weight
        self.error = sum(abs(self.predict[i] - Y[i])**2 for i in range(len(Y)))[0,0]


class Newton(Regression):
    def __init__(self, n):
        super(Newton, self).__init__(n)
        self.name =  "Newton's Method"
    def gradient(self, x):
        return (2*self.dm.transpose()*self.dm*x - 2*self.dm.transpose()*self.Y)
    def hessian(self, x):
        return 2*self.dm.transpose()*self.dm

    def calculate(self, x, y, init_G = None):
        self.X = x
        self.Y = np.matrix(y).transpose()
        self.dm = self.design_matrix()
        if init_G == None:
            init_G = np.matrix(np.ones(self.n)).transpose()

        self.weight = init_G - self.inverse(self.hessian(init_G))*self.gradient(init_G)
        self.predict = self.dm*self.weight
        self.error = sum(abs(self.predict[i] - Y[i])**2 for i in range(len(Y)))[0,0]

def main():
    setting = dict()
    for cmd in sys.argv:
        cmd = cmd.split("=")
        if len(cmd) > 1:
            setting[cmd[0]] = cmd[1]

    readData(setting.get('file','test.txt'))
    fig , ax = plt.subplots(2,1,sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    rlse = r_LSE(int(setting.get('n',3)),int(setting.get('lambda',10000)))
    rlse.calculate(X,Y)
    rlse.plot(1)
    rlse.report()

    nm = Newton(int(setting.get('n',3)))
    nm.calculate(X,Y)
    nm.plot(2)
    nm.report()

    plt.show()

main()
