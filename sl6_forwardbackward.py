import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


input_features = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

target_output = np.array([[0],
                          [1],
                          [1],
                          [0]])


np.random.seed(1)
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1


hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
output_bias = np.random.uniform(size=(1, output_layer_neurons))


learning_rate = 0.1


for epoch in range(10000):
    ## FORWARD PROPAGATION
    hidden_layer_input = np.dot(input_features, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    
    loss = np.mean((target_output - predicted_output) ** 2)

   
    
    output_error = target_output - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)


    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    
    output_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
    output_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    
    hidden_weights += input_features.T.dot(hidden_delta) * learning_rate
    hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("\nFinal Output after Training:")
print(predicted_output)
