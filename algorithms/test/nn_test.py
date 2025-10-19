# test_nn.py
import sys
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("../../cpp/build"))

import mlcore_cpp

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X.astype(np.float64)
y = y.astype(np.float64).reshape(-1, 1)  # make it a column vector
print(X.shape, y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.T  # shape: (n_features, n_train_samples)
X_test = X_test.T  # shape: (n_features, n_test_samples)
y_train = y_train.T  # shape: (1, n_train_samples)
y_test = y_test.T  # shape: (1, n_test_samples)
# Create Neural Network
# n_x = number of features, n_y = 1 for binary classification
nn = mlcore_cpp.NeuralNetwork2L(
    x=X_train,
    y=y_train,
    n_x=X_train.shape[0],
    n_y=1,
    iterations=1000,
    learning_rate=0.01,
    n_h=8,
)

nn.train()

y_pred = nn.predict(X_test)

y_pred_flat = y_pred.flatten()

accuracy = np.mean(y_pred_flat == y_test.flatten())
print(f"Test Accuracy: {accuracy*100:.2f}%")
