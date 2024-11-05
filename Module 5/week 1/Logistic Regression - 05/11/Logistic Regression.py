# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset_path = r'C:\Users\Admin\AIO-189-phatpham-1\Module 5\week 1\titanic_modified_dataset.csv'  # Update the path as needed
df = pd.read_csv(dataset_path, index_col='PassengerId')


# Convert DataFrame to numpy array
dataset_arr = df.to_numpy().astype(np.float64)
X, y = dataset_arr[:, :-1], dataset_arr[:, -1]

# Add bias to X
intercept = np.ones((X.shape[0], 1))
X_b = np.concatenate((intercept, X), axis=1)

# Split dataset into train, validation, and test sets
val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X_b, y, test_size=val_size, random_state=random_state, shuffle=is_shuffle
)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=test_size, random_state=random_state, shuffle=is_shuffle
)

# Normalize the data
normalizer = StandardScaler()
X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define prediction function
def predict(X, theta):
    dot_product = np.dot(X, theta)
    return sigmoid(dot_product)

# Define loss function
def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

# Define gradient function
def compute_gradient(X, y, y_hat):
    return np.dot(X.T, (y_hat - y)) / y.size

# Update weights function
def update_theta(theta, gradient, lr):
    return theta - lr * gradient

# Accuracy function
def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta).round()
    return (y_hat == y).mean()

# Initialize hyperparameters
lr = 0.01
epochs = 100
batch_size = 16
np.random.seed(random_state)
theta = np.random.uniform(size=X_train.shape[1])

# Training the model
train_accs, train_losses, val_accs, val_losses = [], [], [], []

for epoch in range(epochs):
    train_batch_losses, train_batch_accs = [], []
    val_batch_losses, val_batch_accs = [], []

    for i in range(0, X_train.shape[0], batch_size):
        X_i = X_train[i:i+batch_size]
        y_i = y_train[i:i+batch_size]

        y_hat = predict(X_i, theta)
        train_loss = compute_loss(y_hat, y_i)
        gradient = compute_gradient(X_i, y_i, y_hat)
        theta = update_theta(theta, gradient, lr)

        train_batch_losses.append(train_loss)

    train_acc = compute_accuracy(X_train, y_train, theta)
    val_acc = compute_accuracy(X_val, y_val, theta)

    train_losses.append(np.mean(train_batch_losses))
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f'EPOCH {epoch + 1}:\tTraining loss: {train_losses[-1]:.3f}\tValidation accuracy: {val_acc:.3f}')

# Evaluate the model
val_set_acc = compute_accuracy(X_val, y_val, theta)
test_set_acc = compute_accuracy(X_test, y_test, theta)
print('Evaluation on validation and test set:')
print(f'Validation Accuracy: {val_set_acc:.3f}')
print(f'Test Accuracy: {test_set_acc:.3f}')

# Visualizing results
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_losses)
ax[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
ax[0, 1].plot(val_losses, 'orange')
ax[0, 1].set(xlabel='Epoch', ylabel='Loss', title='Validation Loss')
ax[1, 0].plot(train_accs)
ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy', title='Training Accuracy')
ax[1, 1].plot(val_accs, 'orange')
ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy', title='Validation Accuracy')
plt.show()
