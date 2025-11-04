"""
Lab Assignment 6 - Question 2
GridSearchCV for K-NN Classifier Hyperparameter Tuning
"""

import numpy as np
import pandas as pd  # Add this import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("="*70)
print("GRIDSEARCHCV FOR K-NN CLASSIFIER")
print("="*70)
print(f"Dataset: Iris")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("="*70)

# Define parameter grid for K values
param_grid = {
    'n_neighbors': list(range(1, 31)),  # K values from 1 to 30
    'weights': ['uniform', 'distance'],  # Weight functions
    'metric': ['euclidean', 'manhattan']  # Distance metrics
}

print("\nParameter Grid:")
print(f"  n_neighbors: {param_grid['n_neighbors']}")
print(f"  weights: {param_grid['weights']}")
print(f"  metric: {param_grid['metric']}")
print(f"\nTotal combinations: {len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])}")

# Create K-NN classifier
knn = KNeighborsClassifier()

# Perform GridSearchCV
print("\nPerforming GridSearchCV with 5-fold cross-validation...")
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

print("\n" + "="*70)
print("GRIDSEARCHCV RESULTS")
print("="*70)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
print(f"Best K value: {grid_search.best_params_['n_neighbors']}")

# Test the best model
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display top 10 parameter combinations
print("\n" + "="*70)
print("TOP 10 PARAMETER COMBINATIONS")
print("="*70)
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[['param_n_neighbors', 'param_weights', 'param_metric', 'mean_test_score', 'std_test_score']]
print(top_10.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: K value vs Accuracy for different weights (euclidean)
euclidean_uniform = results_df[(results_df['param_metric'] == 'euclidean') & (results_df['param_weights'] == 'uniform')]
euclidean_distance = results_df[(results_df['param_metric'] == 'euclidean') & (results_df['param_weights'] == 'distance')]

axes[0, 0].plot(euclidean_uniform['param_n_neighbors'], euclidean_uniform['mean_test_score'], 
                marker='o', label='Uniform', linewidth=2)
axes[0, 0].plot(euclidean_distance['param_n_neighbors'], euclidean_distance['mean_test_score'], 
                marker='s', label='Distance', linewidth=2)
axes[0, 0].axvline(grid_search.best_params_['n_neighbors'], color='red', 
                   linestyle='--', label=f"Best K = {grid_search.best_params_['n_neighbors']}")
axes[0, 0].set_xlabel('K (Number of Neighbors)')
axes[0, 0].set_ylabel('Cross-Validation Accuracy')
axes[0, 0].set_title('K vs Accuracy (Euclidean Distance)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: K value vs Accuracy for different metrics (uniform)
uniform_euclidean = results_df[(results_df['param_weights'] == 'uniform') & (results_df['param_metric'] == 'euclidean')]
uniform_manhattan = results_df[(results_df['param_weights'] == 'uniform') & (results_df['param_metric'] == 'manhattan')]

axes[0, 1].plot(uniform_euclidean['param_n_neighbors'], uniform_euclidean['mean_test_score'], 
                marker='o', label='Euclidean', linewidth=2)
axes[0, 1].plot(uniform_manhattan['param_n_neighbors'], uniform_manhattan['mean_test_score'], 
                marker='s', label='Manhattan', linewidth=2)
axes[0, 1].set_xlabel('K (Number of Neighbors)')
axes[0, 1].set_ylabel('Cross-Validation Accuracy')
axes[0, 1].set_title('K vs Accuracy (Uniform Weights)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Heatmap of mean scores
pivot_table = results_df[results_df['param_metric'] == 'euclidean'].pivot_table(
    values='mean_test_score', 
    index='param_weights', 
    columns='param_n_neighbors'
)
sns.heatmap(pivot_table, annot=False, cmap='YlGnBu', ax=axes[1, 0], cbar_kws={'label': 'Accuracy'})
axes[1, 0].set_title('Accuracy Heatmap (Euclidean Distance)')
axes[1, 0].set_xlabel('K (Number of Neighbors)')
axes[1, 0].set_ylabel('Weight Function')

# Plot 4: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1, 1])
axes[1, 1].set_title(f'Confusion Matrix (Best Model: K={grid_search.best_params_["n_neighbors"]})')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('gridsearch_knn_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved!")

print("\n" + "="*70)
print("GRIDSEARCHCV COMPLETE!")
print("="*70)