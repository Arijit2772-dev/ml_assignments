"""
Lab Assignment 6 - Question 1
Gaussian Naïve Bayes Classifier on Iris Dataset
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("="*70)
print("PART (i): STEP-BY-STEP IMPLEMENTATION")
print("="*70)

# Step-by-step Implementation
class GaussianNaiveBayesFromScratch:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / n_samples
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
    
    def _calculate_likelihood(self, x, mean, var):
        eps = 1e-6
        coefficient = 1.0 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coefficient * exponent
    
    def _calculate_posterior(self, x):
        posteriors = {}
        for c in self.classes:
            posterior = np.log(self.priors[c])
            for i in range(len(x)):
                likelihood = self._calculate_likelihood(x[i], self.mean[c][i], self.var[c][i])
                posterior += np.log(likelihood + 1e-10)
            posteriors[c] = posterior
        return posteriors
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return np.array(predictions)

# Train and test
gnb_custom = GaussianNaiveBayesFromScratch()
gnb_custom.fit(X_train, y_train)
y_pred_custom = gnb_custom.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom, target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))

print("\n" + "="*70)
print("PART (ii): USING BUILT-IN FUNCTION")
print("="*70)

# Built-in function
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train, y_train)
y_pred_sklearn = gnb_sklearn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_sklearn, target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_sklearn))