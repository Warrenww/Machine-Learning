import sys
import numpy as np
import matplotlib.pyplot as plt

def gaussian_generator(m = 0, s = 1):
    (u, v) = (np.random.uniform(), np.random.uniform())
    result = np.sqrt(-2 * np.log(u)) * np.cos(2*np.pi*v)
    return result*np.sqrt(s) + m

def Newton_method(n, x, y):
    X = np.matrix(x).transpose()
    Y = np.matrix(y).transpose()
    dm = np.matrix(np.ones((X.shape[0],3)))


    for i,x in enumerate(X):
        for j in range(3):
            dm[i,j] = x**j

    # print(dm)
    init_G = np.matrix(np.ones(3)).transpose()
    gradient = 2*dm.transpose()*dm*init_G - 2*dm.transpose()*Y
    hessian = 2*dm.transpose()*dm

    weight = init_G - np.linalg.inv(hessian)*gradient

    return weight

if __name__ == '__main__':
    mode = None
    number_of_data = 1
    for idx, cmd in enumerate(sys.argv):
        if cmd == "-lr":
            mode = "logistic"
        elif cmd == "-p":
            mode = "polynomial"
        elif cmd == "-e":
            mode = "estimator"
        elif cmd == "-r":
            mode = "regression"
        elif cmd == "-n":
            assert len(sys.argv) > (idx + 1)
            number_of_data = int(sys.argv[idx + 1])

    if mode == "logistic":
        number_of_data = int(input("Number of data: "))
        m_x_1 = m_y_1 = 1
        m_x_2 = m_y_2 = 3
        v_x_1 = v_y_1 = 2
        v_x_2 = v_y_2 = 4
        # m_x_1 = float(input("Mean of x for data 1: "))
        # v_x_1 = float(input("variance of x for data 1: "))
        # m_y_1 = float(input("Mean of y for data 1: "))
        # v_y_1 = float(input("variance of y for data 1: "))
        # m_x_2 = float(input("Mean of x for data 2: "))
        # v_x_2 = float(input("variance of x for data 2: "))
        # m_y_2 = float(input("Mean of y for data 2: "))
        # v_y_2 = float(input("variance of y for data 2: "))

        data_1 = [(gaussian_generator(m_x_1, v_x_1), gaussian_generator(m_y_1, v_y_1)) for i in range(number_of_data)]
        data_2 = [(gaussian_generator(m_x_2, v_x_2), gaussian_generator(m_y_2, v_y_2)) for i in range(number_of_data)]
        data_all = data_1 + data_2
        x,y = zip(*data_all)
        w = Newton_method(number_of_data,x,y)

        X = np.linspace(min(x),max(x),50)
        DM = np.matrix([X**i for i in range(3)])
        plt.plot(X,np.array(w.transpose()*DM)[0], 'k-')
        data_1_x, data_1_y = zip(*data_1)
        data_2_x, data_2_y = zip(*data_2)
        plt.plot(data_1_x, data_1_y, 'bo')
        plt.plot(data_2_x, data_2_y, 'ro')
        plt.show()
    elif mode == 'estimator':
        mean = float(input("mean: "))
        var = float(input("variance: "))
        sequential_estimator(mean, var)
    elif mode == 'regression':
        b = float(input("b: "))
        n = int(input("n: "))
        w = np.matrix([float(input("w_{}: ".format(index))) for index in range(n)])
        var = float(input("variance: "))
        linear_regression(b, n, w, var)
    else:
        print("""
usage:
-lr  : Logistic Regression
# -e   : sequential estimator
-r   : Baysian linear regression
        """)
