#Pytorch implementation
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN(input_size=X_train.shape[1])
# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    
    # Calculate loss
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y_train_tensor).float().mean()
    
    train_losses.append(loss.item())
    train_accuracies.append(accuracy.item())
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    predicted_test = (outputs_test > 0.5).float()
    accuracy_test = (predicted_test == y_test_tensor).float().mean()

print(f"Test Accuracy: {accuracy_test.item() * 100:.2f}%")


plt.plot(train_losses, label='Train Loss')
plt.legend()
plt.title('Training Loss')
plt.show()

plt.plot(train_accuracies, label='Train Accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()