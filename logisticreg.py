from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
weights = np.zeros(X_train.shape[1])
bias = 0
y_train = y_train.values  # Convert to NumPy array
y_test = y_test.values  # Convert to NumPy array

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# Convert the pandas Series y_train and y_test to NumPy arrays


def compute_cost(y_true, y_pred):
    # Ensure y_true and y_pred are NumPy arrays and have the same shape
    m = y_true.shape[0]
    
    # Add a small value (epsilon) to avoid log(0) errors
    epsilon = 1e-10
    cost = (-1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return cost 


def backward_propagration(X, y_true, y_pred):
    m = X.shape[0]
    dz = y_pred - y_true
    dw = (1 / m) * np.dot(X.T, dz)
    db = (1 / m) * np.sum(dz)
    return dw, db

def update_params(weights, bias, dw, db, learning_rate):
    weights -= learning_rate * db
    bias -= learning_rate * db
    return weights, bias 

def train(X_train, y_train, weights, bias, learning_rate, epochs):
    for i in range(epochs):
        # forward propogation 
        y_pred = forward_propagation(X_train, weights, bias)

        # cost
        cost = compute_cost(y_train, y_pred)

        # backpropagation
        dw, db = backward_propagration(X_train, y_train, y_pred)

        # update weights, bias
        weights, bias = update_params(weights, bias, dw, db ,learning_rate)

        # print cost every 100 iterations 
        if i % 100 == 0:
            print(f'Epoch {i}, Cost {cost}')
    return weights, bias 

weights, bias = train(X_train, y_train, weights, bias, learning_rate=0.01, epochs=1000)

from sklearn.metrics import accuracy_score, precision_score, recall_score

def predict(X, weights, bias):
    y_pred = forward_propagation(X, weights, bias)
    y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
    return y_pred_class

y_test_pred = predict(X_test, weights, bias)

print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
print(f'Precision: {precision_score(y_test, y_test_pred)}')
print(f'Recall: {recall_score(y_test, y_test_pred)}')
