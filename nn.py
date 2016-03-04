import numpy as np
import matplotlib.pyplot as plt
import time

MAX_IT = 50000
N = 30

def load_data():
    # load data
    data = np.genfromtxt('data.txt', delimiter=' ')
    x = data[:, 0].reshape((1, -1))
    y = data[:, 1].reshape((1, -1))
    return x, y

def init():
    W1 = np.random.normal(0, 1e-3, (N, 1))
    b1 = np.random.normal(0, 1e-3, (N, 1))
    W2 = np.random.normal(0, 1e-3, (1, N))
    b2 = np.random.normal(0, 1e-3, (1, 1))
    return W1, b1, W2, b2

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def compute_nn(W1, b1, W2, b2):
    x = np.linspace(0, 2*np.pi, 1000).reshape((1, -1))
    # x = np.array([0, 1]).reshape(1, -1)
    t1 = time.time()
    z2 = np.dot(W1, x) + b1
    a2 = np.tanh(z2)
    z3 = np.dot(W2, a2) + b2
    a3 = np.tanh(z3)
    t2 = time.time()
    print "time is ms", (t2 - t1) * 1000000
    y = a3
    print y
    return x[0, :], y[0, :]

def train(x, y):
    W1, b1, W2, b2 = init()
    alpha = 1e-4
    errs = []
    
    t1 = time.time()
    for i in xrange(MAX_IT):
        z2 = np.dot(W1, x) + b1
        a2 = np.tanh(z2)
        z3 = np.dot(W2, a2) + b2
        a3 = np.tanh(z3)

        err = a3 - y
        errs.append(np.mean(err ** 2))
        d3 = err * tanh_prime(z3)
        d2 = np.dot(W2.T, d3) * tanh_prime(z2)

        grad_W2 = np.dot(d3, a2.T)
        grad_b2 = np.sum(d3, axis=1, keepdims=True)
        grad_W1 = np.dot(d2, x.T)
        grad_b1 = np.sum(d2, axis=1, keepdims=True)
        
        W1 -= alpha * grad_W1
        b1 -= alpha * grad_b1
        W2 -= alpha * grad_W2
        b2 -= alpha * grad_b2

        if i % 1000 == 999:
            t2 = time.time()
            print "Epoch %d, time elapsed for 1000 epoch %f" % (i + 1, t2 - t1)
            t1 = time.time()
            print "error %f" % np.mean(err ** 2)

    # plt.subplot(121)
    # plt.plot(np.arange(MAX_IT), errs)
    return W1, b1, W2, b2

def main():
    np.random.seed(42)
    x, y = load_data()
    print "start training on CPU"
    W1, b1, W2, b2 = train(x, y)
    # np.savetxt("W1.txt", W1)
    # np.savetxt("b1.txt", b1)
    # np.savetxt("W2.txt", W2)
    # np.savetxt("b2.txt", b2)
    nn_x, nn_y = compute_nn(W1, b1, W2, b2)
    cuda_nn_y = np.genfromtxt("output.txt", delimiter="\n")
    plt.subplot(121)
    plt.title("Fitted curve of CPU NN")
    plt.plot(nn_x, nn_y)
    plt.subplot(122)
    plt.title("Fitted curve of GPU NN")
    plt.plot(nn_x, cuda_nn_y)
    # plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()
