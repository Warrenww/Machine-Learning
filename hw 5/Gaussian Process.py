import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


X = list()
Y = list()
noise = .2

def kernel(X_m, X_n, sigma = 1, alpha = 1, l = 1):
    d = (X_m**2)-2*np.dot(X_m,X_n.T)+(X_n**2).flatten()
    return (sigma**2)*(1 + d**2/(2*alpha*l))**(-alpha)

def prediction(X_train, Y_train, X_test, noise = .2, sigma = 1, alpha = 1, l = 1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    C = np.matrix(kernel(X_train, X_train, sigma, alpha, l).reshape(n_train, n_train)) + (noise**2)*np.eye(n_train)
    k_s = np.matrix(kernel(X_train, X_test, sigma, alpha, l).reshape(n_train, n_test))
    k_ss = np.matrix(kernel(X_test, X_test, sigma, alpha, l).reshape(n_test, n_test))
    mean = k_s.T*np.linalg.inv(C)*np.matrix(Y_train)
    var = k_ss + noise**2 -  k_s.T*np.linalg.inv(C)*k_s

    return (mean, var)

def likelihood(params):
    n = X.shape[0]
    C = np.matrix(kernel(X, X, params[0], params[1], params[2])).reshape(n, n) + (noise**2)*np.eye(n)

    return 1/2 * np.log(np.linalg.det(C)) + 1/2 * Y.T * np.linalg.inv(C) * Y + n/2 * np.log(2*np.pi)


if __name__ == '__main__':
    with open('input.data') as f:
        for line in  f.readlines():
            x,y = (line.split(" "))
            X.append(float(x))
            Y.append(float(y))

    X = np.array(X).reshape(-1,1)
    Y = np.array(Y).reshape(-1,1)
    X_test = np.arange(-60,60,0.5).reshape(-1,1)

    m,cov = prediction(X,Y,X_test,noise,1,1,1)

    plt.plot(X,Y,'bo')
    plt.plot(X_test, m, 'r')
    cert = 1.96 * (np.sqrt(np.diag(cov))).reshape(-1,1)

    plt.fill_between(X_test.ravel(), np.array(m + cert).ravel(), np.array(m - cert).ravel(), alpha=0.2)
    plt.show()

    optimal_parasms = minimize(likelihood, [1,1,1])
    print(optimal_parasms.x)
    opt_sigma, opt_alpha, opt_l = optimal_parasms.x
    opt_m, opt_cov = prediction(X,Y,X_test,noise ,opt_sigma,opt_alpha,opt_l)

    plt.plot(X,Y,'bo')
    plt.plot(X_test, opt_m, 'r')
    cert = 1.96 * (np.sqrt(np.diag(opt_cov))).reshape(-1,1)

    plt.fill_between(X_test.ravel(), np.array(opt_m + cert).ravel(), np.array(opt_m - cert).ravel(), alpha=0.2)
    plt.show()
