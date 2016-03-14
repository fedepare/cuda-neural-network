import numpy as np
import time
import cPickle

MAX_IT = 50000
N = 30

class FullyConnectedLayer:
    def __init__(self, dim_input, n_neuron):
        self.W = np.random.normal(0, 1e-3, (n_neuron, dim_input))
        self.b = np.random.normal(0, 1e-3, (n_neuron, 1))
        self.z = None
        self.a = None
        self.a_pre = None
        self.grad_W1 = None
        self.delta = None

    def forward(self, a_pre):
        self.z = np.dot(self.W, a_pre) + self.b
        self.a = np.tanh(self.z)
        self.a_pre = a_pre
        return self.a
    
    def backward(self, prod):
        self.delta = prod * tanh_prime(self.z)
        return np.dot(self.W.T, self.delta)
    
    def grad_desc(self, rate):
        self.grad_W = np.dot(self.delta, self.a_pre.T)
        self.grad_b = np.sum(self.delta, axis=1, keepdims=True)
        self.W -= rate * self.grad_W
        self.b -= rate * self.grad_b
        # print self.W
        # print self.b

class LeastSquareLayer(FullyConnectedLayer):
    def __init__(self, dim_input, n_neuron):
        FullyConnectedLayer.__init__(self, dim_input, n_neuron)

    def backward(self, y_true):
        self.delta = (self.a - y_true) * tanh_prime(self.z)
        return np.dot(self.W.T, self.delta)

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        a_pre = x
        for layer in self.layers:
            a_pre = layer.forward(a_pre)
        return a_pre

    def fit(self, x_train, y_train, max_it=10000):
        rate = 1e-4
        t1 = time.time()
        for i in xrange(max_it):
            #forward
            result = self.forward(x_train)

            #backward
            prod = self.layers[-1].backward(y_train)
            for layer in reversed(self.layers[:-1]):
                prod = layer.backward(prod)

            #gradient descent
            for layer in self.layers:
                layer.grad_desc(rate)

            if i % 1000 == 999:
                t2 = time.time()
                print "Epoch %d, Time elapsed in ms %f, error %f" % (1 + i, (t2 - t1) * 1000000, np.mean((result- y_train) ** 2))
                t1 = time.time()
    
    def predict(self, x_test):
        return self.forward(x_test)
    
def load_data():
    # load data
    data = np.genfromtxt('data.txt', delimiter=' ')
    x = data[:, 0].reshape((1, -1))
    y = data[:, 1].reshape((1, -1))
    return x, y

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def main():
    np.random.seed(42)
    x, y = load_data()

    network = Network()
    network.add_layer(FullyConnectedLayer(1, N)) #hidden layer
    network.add_layer(LeastSquareLayer(N, 1))    #output layer
    network.fit(x, y, 100000)
    with open("nn.pkl", "wb") as fout:
        cPickle.dump(network, fout, 2)

if __name__ == "__main__":
    main()
