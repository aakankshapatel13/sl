# Step 1: Import Libraries Tensorflow/keras
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 3: Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 4: Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      # Converts 2D image to 1D
    layers.Dense(128, activation='relu'),      # Hidden layer
    layers.Dense(64, activation='relu'),       # Additional layer (optional)
    layers.Dense(10, activation='softmax')     # Output layer for 10 digits
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Step 7: Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 8: Make predictions and show first 10 results
predictions = model.predict(x_test[:10])

for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, True: {y_test[i]}")
    plt.axis('off')
    plt.show()
