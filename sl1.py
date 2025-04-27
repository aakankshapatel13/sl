import numpy as np
import matplotlib.pyplot as plt

def plot_sigmoid():
    x = np.linspace(-10, 10, 100) 
    y = 1 / (1 + np.exp(-x))  
    plt.plot(x, y)
    plt.xlabel('Input')
    plt.ylabel('Sigmoid Output')
    plt.title('Sigmoid Activation Function')
    plt.grid(True)
    plt.show()
plot_sigmoid()    


def plot_tanh():
    x = np.linspace(-10, 10, 100)
    tanh = np.tanh(x)
    plt.plot(x, tanh)
    plt.title("Hyperbolic Tangent (tanh) Activation Function")
    plt.xlabel("x")
    plt.ylabel("tanh(x)")
    plt.grid(True)
    plt.show()
plot_tanh()   

def plot_relu():
    x = np.linspace(-10, 10, 100)
    relu = np.maximum(0, x)
    plt.plot(x, relu)
    plt.title("ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True)
    plt.show()
plot_relu()    



def plot_leaky_relu(alpha=0.01):
    x = np.linspace(-10, 10, 100)
    leaky_relu = np.where(x > 0, x, alpha * x)
    plt.plot(x, leaky_relu)
    plt.title("Leaky ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("Leaky ReLU(x)")
    plt.grid(True)
    plt.show()
plot_leaky_relu()



def plot_identity():
    x = list(range(-5, 6))  
    y = x  
    plt.plot(x, y)
    plt.title("Identity Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()
plot_identity()


def plot_softmax():
    x = np.linspace(-5, 5, 100)  
    y = np.exp(x) / np.sum(np.exp(x))  
    plt.plot(x, y, label='Softmax')
    plt.xlabel('Input')
    plt.ylabel('Activation')
    plt.title('Activation Functions')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_softmax()    