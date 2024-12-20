# -*- coding: utf-8 -*-
"""MLPs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HVYdEYsBzZGlr0HwWWS87nEmarf5zf-V
"""

# Install required libraries if not already installed
!pip install torch torchvision

# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Download the dataset
!gdown --id 1qiUDDoYyRLBiKOoYWdFl_5WByHE8Cugu

# Load dataset
dataset = pd.read_csv('Auto_MPG_data.csv')
X = dataset.drop(columns='MPG').values
y = dataset['MPG'].values

# Split the data
random_state = 59
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.125, random_state=random_state)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Model parameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val).squeeze()
        val_loss = criterion(y_val_pred, y_val)

    print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

# Calculate R² score
def r_squared(y_true, y_pred):
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return 1 - (ss_res / ss_tot)

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).squeeze()
    r2_score = r_squared(y_test, y_test_pred)
    print(f"R² score on test set: {r2_score.item()}")