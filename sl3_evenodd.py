import numpy as np 
 
step_function = lambda x: 1 if x >= 0 else 0 
 
training_data = [ 
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1}, 
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0}, 
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1}, 
    {'input': [1, 1, 0, 0, 1, 1], 'label': 0}, 
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1}, 
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0}, 
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1}, 
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0}, 
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1}, 
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0}, 
] 
 
weights = np.zeros(6) 
for _ in range(10): 
    for data in training_data: 
        inputs = np.array(data['input']) 
        label = data['label'] 
        output = step_function(np.dot(inputs, weights)) 
        weights += 0.1 * (label - output) * inputs 
 
while True: 
    user_input = input("Enter a Number (0-9) or 'exit' to quit: ") 
    if user_input.lower() == 'exit': 
        break 
    if user_input.isdigit() and 0 <= int(user_input) <= 9: 
        inputs = [int(x) for x in '{0:06b}'.format(int(user_input))] 
        result = "even" if step_function(np.dot(inputs, weights)) == 1 else "odd" 
        print(user_input, "is", result) 
    else: 
        print("Invalid input. Try again.")