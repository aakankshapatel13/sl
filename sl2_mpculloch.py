import numpy as np

def mp_neuron(inputs, weights, threshold): 
    weighted_sum = np.dot(inputs, weights)  
    output = 1 if weighted_sum >= threshold else 0  
    return output 

def and_not(x1, x2): 
    weights = np.array([1, -1])  
    threshold = 1   
    inputs = np.array([x1, x2])  
    output = mp_neuron(inputs, weights, threshold)  
    formula = f"({x1} * 1) + ({x2} * -1) = {np.dot(inputs, weights)}"  
    return output, formula 


print("X1\t X2\t Formula\t\t\t Output") 
for x1 in [0, 1]: 
    for x2 in [0, 1]: 
        output, formula = and_not(x1, x2)  
        print(f"{x1}\t{x2}\t{formula}\t\t{output}")
