# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1881" height="923" alt="image" src="https://github.com/user-attachments/assets/33af1fdc-2e02-43fa-b886-17a6a014fa79" />


## DESIGN STEPS

### STEP 1
Load and preprocess the dataset (handle missing values, encode categorical features, scale numeric data).

### STEP 2
Split the dataset into training and testing sets, convert to tensors, and create DataLoader objects.

### STEP 3
Build the neural network model, train it with CrossEntropyLoss and Adam optimizer, then evaluate with confusion matrix and classification report.

## PROGRAM
### Name: NATARAJ KUMARAN S
### Register Number: 212223230137

```python
# Define Neural Network(Model1)
class NeuralNetwork(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = torch.nn.Linear(size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return 
```
```python
# Initialize the Model, Loss Function, and Optimizer
expai = NeuralNetwork(X_train.shape[1])
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(expai.parameters(), lr=0.001)
```
```python

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
```

## Dataset Information

<img width="1451" height="638" alt="image" src="https://github.com/user-attachments/assets/cb844952-5aa5-440f-8864-09197b6e52c3" />

## OUTPUT

### Confusion Matrix

<i<img width="558" height="472" alt="image" src="https://github.com/user-attachments/assets/be78e42b-bd0e-4cbd-be5d-acd81117669b" />


### Classification Report

<img width="550" height="347" alt="image" src="https://github.com/user-attachments/assets/56d16b95-c89c-48f6-9ce6-4ae5abf4666d" />


### New Sample Data Prediction

<img width="1547" height="97" alt="image" src="https://github.com/user-attachments/assets/d1101063-f520-4087-ab49-a881990ffae2" />


## RESULT
The neural network model was successfully built and trained to handle classification tasks.
