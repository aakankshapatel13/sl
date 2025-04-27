import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(42)
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1
W1 = np.random.uniform(size=(input_layer_size, hidden_layer_size))
W2 = np.random.uniform(size=(hidden_layer_size, output_layer_size))
b1 = np.random.uniform(size=(1, hidden_layer_size))
b2 = np.random.uniform(size=(1, output_layer_size))
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)


    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)


    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal Output after training:")
for i in range(4):
    print(f"Input: {X[i]} → Predicted: {round(final_output[i][0])} (Actual: {y[i][0]})")