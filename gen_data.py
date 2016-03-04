import numpy as np
import matplotlib.pyplot as plt

def draw_sin(a, b):
    x = np.linspace(a, b, 10000)
    y = np.sin(x)
    plt.plot(x, y, 'r-')

def main():
    N = 1000
    x = np.random.uniform(0, 2 * np.pi, N)
    y = np.sin(x) + np.random.normal(0, 0.1, N)
    draw_sin(0, 2 * np.pi)
    plt.scatter(x, y)
    plt.show()
    data = np.concatenate((x.reshape(-1, 1), y.reshape((-1, 1))), axis=1)
    np.savetxt("data.txt", data, fmt="%.8f")

if __name__ == "__main__":
    main()
