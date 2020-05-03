import sys
import numpy as np
import matplotlib.pyplot as plt

def gaussian_generator(m = 0, s = 1):
    (u, v) = (np.random.uniform(), np.random.uniform())
    result = np.sqrt(-2 * np.log(u)) * np.cos(2*np.pi*v)
    return result*np.sqrt(s) + m

class Regression(object):
    def __init__(self, x, y):
        self.X = np.matrix(x)
        self.Y = np.matrix(y)
        self.weight = np.matrix(np.zeros(3)).transpose()

    def gradient(self):
        return self.X.transpose()*(1/(1+np.exp(-self.X*self.weight)) - self.Y)

    def hessian(self):
        D = np.matrix(np.zeros((self.X.shape[0],self.X.shape[0])))
        for i,x in enumerate(self.X):
            D[i,i] = np.exp(-x*self.weight)/(1+np.exp(-x*self.weight))**2
        return self.X.transpose()*D*self.X

    def solve(self):
        count = 0
        while True:
            count += 1
            weight_new = self.weight - self.delta()
            if abs(np.sum(weight_new - self.weight)) < 1e-3 or count > 1e4:
                self.weight = weight_new
                break
            self.weight = weight_new
        return self.weight

class Newton_method(Regression):
    def __init__(self, x, y):
        super(Newton_method, self).__init__(x, y)

    def delta(self):
        return np.linalg.inv(self.hessian())*self.gradient()

class Gradient_descent(Regression):
    def __init__(self, x, y):
        super(Gradient_descent, self).__init__(x, y)
        self.learning_rate = 0.001

    def delta(self):
        return self.learning_rate*self.gradient()

if __name__ == '__main__':
    mode = None
    number_of_data = 1
    for idx, cmd in enumerate(sys.argv):
        if cmd == "-lr":
            mode = "logistic"
        elif cmd == "-em":
            mode = "EM"
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

        data_1 = [(gaussian_generator(m_x_1, v_x_1), gaussian_generator(m_y_1, v_y_1), 1) for i in range(number_of_data)]
        data_2 = [(gaussian_generator(m_x_2, v_x_2), gaussian_generator(m_y_2, v_y_2), 1) for i in range(number_of_data)]

        data_all = np.matrix(data_1 + data_2)
        labels = np.matrix([j for j in range(2) for i in range(number_of_data)]).transpose()

        w0 = Gradient_descent(data_all,labels).solve()
        w1 = Newton_method(data_all,labels).solve()

        plt.subplot(131)
        plt.title("Ground truth")
        x_1, y_1, _ = zip(*data_1)
        x_2, y_2, _ = zip(*data_2)
        plt.plot(x_1, y_1, 'bo')
        plt.plot(x_2, y_2, 'ro')

        print("""
Gradient descent:

w:
{}

Confusion Matrix:
                Predict cluster 1 Predict cluster 2
Is cluster 1    {:^18}{:^18}
Is cluster 2    {:^18}{:^18}

Sensitivity (Successfully predict cluster 1): {}
Specificity (Successfully predict cluster 2): {}

----------------------------------------------------------------
        """.format(
        w0,
        np.sum(data_1*w0 < 0),
        np.sum(data_1*w0 > 0),
        np.sum(data_2*w0 < 0),
        np.sum(data_2*w0 > 0),
        np.sum(data_1*w0 < 0) / len(data_1),
        np.sum(data_2*w0 > 0) / len(data_2)
        ))

        plt.subplot(132)
        plt.title("Gradient descent")
        class_0_0 = np.matrix([ np.array(x)[0] for x in data_all if x*w0 > 0])
        class_0_1 = np.matrix([ np.array(x)[0] for x in data_all if x*w0 < 0])
        plt.plot(class_0_0[:,0], class_0_0[:,1], 'ro')
        plt.plot(class_0_1[:,0], class_0_1[:,1], 'bo')

        print("""
Newton's method:

w:
{}

Confusion Matrix:
                Predict cluster 1 Predict cluster 2
Is cluster 1    {:^18}{:^18}
Is cluster 2    {:^18}{:^18}

Sensitivity (Successfully predict cluster 1): {}
Specificity (Successfully predict cluster 2): {}

----------------------------------------------------------------
        """.format(
        w1,
        np.sum(data_1*w1 < 0),
        np.sum(data_1*w1 > 0),
        np.sum(data_2*w1 < 0),
        np.sum(data_2*w1 > 0),
        np.sum(data_1*w1 < 0) / len(data_1),
        np.sum(data_2*w1 > 0) / len(data_2)
        ))

        plt.subplot(133)
        plt.title("Newton's method")
        class_1_0 = np.matrix([ np.array(x)[0] for x in data_all if x*w1 > 0])
        class_1_1 = np.matrix([ np.array(x)[0] for x in data_all if x*w1 < 0])
        plt.plot(class_1_0[:,0], class_1_0[:,1], 'ro')
        plt.plot(class_1_1[:,0], class_1_1[:,1], 'bo')

        plt.show()
    elif mode == 'EM':
        mean = float(input("mean: "))
        var = float(input("variance: "))
        sequential_estimator(mean, var)
    else:
        print("""
usage:
-lr  : Logistic Regression
-em  : EM algorithm
        """)
