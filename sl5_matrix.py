import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

digits = {
    0: [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1],
    1: [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1],
    2: [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1],
    3: [1,1,1, 0,0,1, 0,1,1, 0,0,1, 1,1,1],
    4: [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1],
    5: [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1],
    6: [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1],
    7: [1,1,1, 0,0,1, 0,1,0, 0,1,0, 1,1,0],
    8: [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1],
    9: [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]
}

X = np.array([digits[d] for d in range(10)])
y = np.eye(10)  

input_neurons = 15
hidden_neurons = 10
output_neurons = 10

np.random.seed(1)
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

epochs = 10000
learning_rate = 0.1

for _ in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, hidden_weights) + hidden_bias
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, output_weights) + output_bias
    predicted_output = sigmoid(output_input)

    
    error = y - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(output_weights.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    
    output_weights += hidden_output.T.dot(d_output) * learning_rate
    output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden) * learning_rate
    hidden_bias += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


def predict_digit(test_matrix):
    test_input = np.array(test_matrix).reshape(1, 15)
    hidden_layer = sigmoid(np.dot(test_input, hidden_weights) + hidden_bias)
    output_layer = sigmoid(np.dot(hidden_layer, output_weights) + output_bias)
    return np.argmax(output_layer)


test_digit = [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1]  # Digit 0
print("Predicted Digit:", predict_digit(test_digit))