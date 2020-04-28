import sys
import numpy as np
import matplotlib.pyplot as plt

def gaussian_generator(m = 0, s = 1):
    (u, v) = (np.random.uniform(), np.random.uniform())
    result = np.sqrt(-2 * np.log(u)) * np.cos(2*np.pi*v)
    return result*np.sqrt(s) + m

def polynomial_generator(n, w ,s = 0):
    # w is a row matrix
    e = gaussian_generator(0, s)
    x = np.random.uniform(-1,1)
    X = np.matrix([x**i for i in range(n)])
    y = w*X.transpose() + e
    return (x,y[0,0])

def sequential_estimator(m = 0, s = 1):
    current_mean = 0
    current_var = 0
    M_2 = 0
    count = 0
    while True:
        data = gaussian_generator(m, s)
        count += 1
        delta = data - current_mean
        mean = current_mean + delta / count
        delta_2 = data - mean
        M_2 += delta * delta_2
        var = M_2 / count
        print("""
data point: {:11f}
mean:       {:11f}
variance:   {:11f}
------------------------
        """.format(data,mean,var))
        if abs(mean - current_mean) < 1e-4 and abs(var - current_var) < 1e-4:
            break
        current_mean = mean
        current_var = var

def linear_regression(b, n, w, s = 1):
    beta = 1 / b
    count = 0
    data_x = []
    data_y = []
    prev_m = np.matrix(np.zeros((n,1)))
    prev_S = np.matrix(np.identity(n) / b)
    while True:
        x, y  = polynomial_generator(n, w, s)
        data_x.append(x)
        data_y.append(y)
        count += 1
        design_matrix = np.matrix([x**i for i in range(n)])

        S = np.linalg.inv((np.linalg.inv(prev_S) + beta*design_matrix.transpose()*design_matrix))
        m = S*(np.linalg.inv(prev_S)*prev_m + beta*design_matrix.transpose()*y)

        print("""
Add data point ({:8f}, {:8f})

Postirior mean:
{}

Postirior variance:
{}

Predictive distribution ~ N({:8f}, {:8f})
        """.format(x, y, m, S, (design_matrix*m)[0,0], (design_matrix*S*design_matrix.transpose())[0,0]))

        if (np.all(abs(m - prev_m) < 1e-5)) and (np.all(abs(S - prev_S) < 1e-5)) and count > 100:
            break

        prev_S = S
        prev_m = m
        if count == 10:
            s_10 = prev_S
            m_10 = prev_m
        if count == 50:
            s_50 = prev_S
            m_50 = prev_m

    plt.subplot(221)
    plt.title("Ground truth")
    plt.xlim(-2,2)
    plt.ylim(-20,20)
    X = np.linspace(-2,2,50)
    DM = np.matrix([X**i for i in range(n)])
    plt.plot(X,np.array(w*DM)[0], 'k-')
    plt.plot(X,np.array(w*DM)[0] + s, 'r-')
    plt.plot(X,np.array(w*DM)[0] - s, 'r-')

    plt.subplot(222)
    plt.title("Predict result")
    plt.xlim(-2,2)
    plt.ylim(-20,20)
    plt.plot(data_x, data_y, 'bo', markersize=3)
    er = 1 / s + DM.transpose()*prev_S*DM
    plt.plot(X,np.array(prev_m.transpose()*DM)[0], 'k-')
    plt.plot(X,np.array(prev_m.transpose()*DM + np.array([er[i,i] for i in range(50)]))[0], 'r-')
    plt.plot(X,np.array(prev_m.transpose()*DM - np.array([er[i,i] for i in range(50)]))[0], 'r-')

    plt.subplot(223)
    plt.title("After 10 incomes")
    plt.xlim(-2,2)
    plt.ylim(-20,20)
    plt.plot(data_x[:10], data_y[:10], 'bo', markersize=3)
    er = 1 / s + DM.transpose()*s_10*DM
    plt.plot(X,np.array(m_10.transpose()*DM)[0], 'k-')
    plt.plot(X,np.array(m_10.transpose()*DM + np.array([er[i,i] for i in range(50)]))[0], 'r-')
    plt.plot(X,np.array(m_10.transpose()*DM - np.array([er[i,i] for i in range(50)]))[0], 'r-')

    plt.subplot(224)
    plt.title("After 50 incomes")
    plt.xlim(-2,2)
    plt.ylim(-20,20)
    plt.plot(data_x[:50], data_y[:50], 'bo', markersize=3)
    er = 1 / s + DM.transpose()*s_50*DM
    plt.plot(X,np.array(m_50.transpose()*DM)[0], 'k-')
    plt.plot(X,np.array(m_50.transpose()*DM + np.array([er[i,i] for i in range(50)]))[0], 'r-')
    plt.plot(X,np.array(m_50.transpose()*DM - np.array([er[i,i] for i in range(50)]))[0], 'r-')

    plt.show()

if __name__ == '__main__':
    mode = None
    number_of_data = 1
    for idx, cmd in enumerate(sys.argv):
        if cmd == "-g":
            mode = "gaussian"
        elif cmd == "-p":
            mode = "polynomial"
        elif cmd == "-e":
            mode = "estimator"
        elif cmd == "-r":
            mode = "regression"
        elif cmd == "-n":
            assert len(sys.argv) > (idx + 1)
            number_of_data = int(sys.argv[idx + 1])

    if mode == "gaussian":
        mean = float(input("mean: "))
        var = float(input("variance: "))
        data = [gaussian_generator(mean, var) for i in range(number_of_data)]
        if number_of_data > 1:
            print(np.mean(data),np.var(data))
            n, bins, patches = plt.hist(data, 100, density=1, facecolor='g', alpha=0.75)
            plt.grid(True)
            plt.show()
        else:
            print(data)
    elif mode == "polynomial":
        n = int(input("n: "))
        w = np.matrix([float(input("w_{}: ".format(index))) for index in range(n)])
        var = float(input("variance: "))
        data_x =list()
        data =list()
        for i in range(number_of_data):
            x, y = polynomial_generator(n, w, var)
            data_x.append(x)
            data.append(y)
        if number_of_data > 1:
            x = np.linspace(-2,2,30)
            xx = np.matrix([x**i for i in range(n)])
            plt.xlim(-2,2)
            plt.plot(x,np.array(w*xx)[0],'k-')
            plt.plot(x,np.array(w*xx)[0] + var,'r-')
            plt.plot(x,np.array(w*xx)[0] - var,'r-')
            plt.plot(data_x,data,'bo',alpha=.5)
            plt.show()
        else:
            print(data_x,data)
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
-g [-n <int>]   : gaussian data generator [with n data points and figure]
-p [-n <int>]   : polynomial data generator [with n data points and figure]
-e              : sequential estimator
-r              : Baysian linear regression
        """)
