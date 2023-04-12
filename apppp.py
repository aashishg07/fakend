# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("data.csv")

# Split the dataset into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Initialize the KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the KNN classifier on the training data
knn.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = knn.predict(X_test)

# Calculate the accuracy score of the KNN classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)
