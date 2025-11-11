"""
SVM Lab Assignment Solution
Author: Generated Solution
Date: 2025-11-11

This script demonstrates:
1. SVM classification on Iris dataset with different kernels
2. Effect of feature scaling on SVM performance using Breast Cancer dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')

print("="*80)
print("SVM LAB ASSIGNMENT SOLUTION")
print("="*80)

# ============================================================================
# PART 1: IRIS DATASET WITH DIFFERENT KERNELS
# ============================================================================

print("\n" + "="*80)
print("PART 1: IRIS DATASET - COMPARING SVM KERNELS")
print("="*80)

# Load Iris dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"\nIris Dataset Shape: {X_iris.shape}")
print(f"Number of classes: {len(np.unique(y_iris))}")
print(f"Class names: {iris.target_names}")
print(f"Feature names: {iris.feature_names}")

# Train-test split (80:20)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

print(f"\nTraining set size: {X_train_iris.shape[0]}")
print(f"Testing set size: {X_test_iris.shape[0]}")

# Define kernels to test
kernels = ['linear', 'poly', 'rbf']
kernel_results = {}

# Train and evaluate models with different kernels
for kernel in kernels:
    print(f"\n{'-'*80}")
    print(f"Training SVM with {kernel.upper()} kernel")
    print(f"{'-'*80}")

    # Create and train SVM model
    if kernel == 'poly':
        svm_model = SVC(kernel=kernel, degree=3, random_state=42)
    else:
        svm_model = SVC(kernel=kernel, random_state=42)

    svm_model.fit(X_train_iris, y_train_iris)

    # Make predictions
    y_pred = svm_model.predict(X_test_iris)

    # Calculate metrics
    accuracy = accuracy_score(y_test_iris, y_pred)
    precision = precision_score(y_test_iris, y_pred, average='weighted')
    recall = recall_score(y_test_iris, y_pred, average='weighted')
    f1 = f1_score(y_test_iris, y_pred, average='weighted')
    cm = confusion_matrix(y_test_iris, y_pred)

    # Store results
    kernel_results[kernel] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    # Print results
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(cm)

    print(f"\nClassification Report:")
    print(classification_report(y_test_iris, y_pred, target_names=iris.target_names))

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices for Different SVM Kernels (Iris Dataset)', fontsize=14, fontweight='bold')

for idx, kernel in enumerate(kernels):
    cm = kernel_results[kernel]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    axes[idx].set_title(f'{kernel.upper()} Kernel\nAccuracy: {kernel_results[kernel]["accuracy"]:.4f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('/Users/arijitsingh/Documents/thapar_sem5/ml_lab/Ass_7/iris_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"\nConfusion matrices saved as 'iris_confusion_matrices.png'")

# Compare all kernels
print(f"\n{'='*80}")
print("COMPARISON OF ALL KERNELS (IRIS DATASET)")
print(f"{'='*80}")

results_df = pd.DataFrame({
    'Kernel': kernels,
    'Accuracy': [kernel_results[k]['accuracy'] for k in kernels],
    'Precision': [kernel_results[k]['precision'] for k in kernels],
    'Recall': [kernel_results[k]['recall'] for k in kernels],
    'F1-Score': [kernel_results[k]['f1_score'] for k in kernels]
})

print("\n", results_df.to_string(index=False))

# Identify best kernel
best_kernel = max(kernel_results.items(), key=lambda x: x[1]['accuracy'])[0]
print(f"\n{'='*80}")
print(f"BEST PERFORMING KERNEL: {best_kernel.upper()}")
print(f"{'='*80}")
print(f"\nReason:")
if best_kernel == 'rbf':
    print("The RBF (Radial Basis Function) kernel performs best because:")
    print("- It can handle non-linear decision boundaries effectively")
    print("- The Iris dataset has some non-linear separability between classes")
    print("- RBF kernel maps data to infinite-dimensional space, allowing complex patterns")
    print("- It's particularly effective when classes are not linearly separable")
elif best_kernel == 'linear':
    print("The Linear kernel performs best because:")
    print("- The Iris dataset classes are mostly linearly separable")
    print("- Linear kernel is simpler and less prone to overfitting")
    print("- It works well when data is already well-separated in the feature space")
else:
    print("The Polynomial kernel performs best because:")
    print("- It can capture polynomial relationships between features")
    print("- Degree-3 polynomial provides good balance between flexibility and complexity")
    print("- It's effective when decision boundaries have polynomial characteristics")

# Plot performance comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(kernels))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics):
    values = [kernel_results[k][metric.lower().replace('-', '_')] for k in kernels]
    ax.bar(x + i*width, values, width, label=metric)

ax.set_xlabel('Kernel', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Performance Comparison of SVM Kernels on Iris Dataset', fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([k.upper() for k in kernels])
ax.legend()
ax.set_ylim([0.9, 1.01])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/arijitsingh/Documents/thapar_sem5/ml_lab/Ass_7/iris_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPerformance comparison chart saved as 'iris_performance_comparison.png'")

# ============================================================================
# PART 2: BREAST CANCER DATASET - EFFECT OF FEATURE SCALING
# ============================================================================

print("\n\n" + "="*80)
print("PART 2: BREAST CANCER DATASET - EFFECT OF FEATURE SCALING")
print("="*80)

# Load Breast Cancer dataset
cancer = datasets.load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print(f"\nBreast Cancer Dataset Shape: {X_cancer.shape}")
print(f"Number of classes: {len(np.unique(y_cancer))}")
print(f"Class names: {cancer.target_names}")
print(f"Number of features: {X_cancer.shape[1]}")

# Show feature statistics to demonstrate different scales
print(f"\nFeature Statistics (showing scale differences):")
feature_stats = pd.DataFrame({
    'Feature': cancer.feature_names[:5],  # Show first 5 features
    'Mean': X_cancer[:, :5].mean(axis=0),
    'Std': X_cancer[:, :5].std(axis=0),
    'Min': X_cancer[:, :5].min(axis=0),
    'Max': X_cancer[:, :5].max(axis=0)
})
print(feature_stats.to_string(index=False))
print("... (and 25 more features)")

# Train-test split
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

print(f"\nTraining set size: {X_train_cancer.shape[0]}")
print(f"Testing set size: {X_test_cancer.shape[0]}")

# ============================================================================
# A) WITHOUT FEATURE SCALING
# ============================================================================

print(f"\n{'-'*80}")
print("A) SVM (RBF Kernel) WITHOUT Feature Scaling")
print(f"{'-'*80}")

# Train SVM without scaling
svm_no_scale = SVC(kernel='rbf', random_state=42)
svm_no_scale.fit(X_train_cancer, y_train_cancer)

# Predictions
y_train_pred_no_scale = svm_no_scale.predict(X_train_cancer)
y_test_pred_no_scale = svm_no_scale.predict(X_test_cancer)

# Calculate metrics
train_acc_no_scale = accuracy_score(y_train_cancer, y_train_pred_no_scale)
test_acc_no_scale = accuracy_score(y_test_cancer, y_test_pred_no_scale)

print(f"\nPerformance WITHOUT Scaling:")
print(f"  Training Accuracy: {train_acc_no_scale:.4f}")
print(f"  Testing Accuracy:  {test_acc_no_scale:.4f}")

print(f"\nConfusion Matrix (Without Scaling):")
cm_no_scale = confusion_matrix(y_test_cancer, y_test_pred_no_scale)
print(cm_no_scale)

print(f"\nClassification Report (Without Scaling):")
print(classification_report(y_test_cancer, y_test_pred_no_scale, target_names=cancer.target_names))

# ============================================================================
# B) WITH FEATURE SCALING
# ============================================================================

print(f"\n{'-'*80}")
print("B) SVM (RBF Kernel) WITH Feature Scaling (StandardScaler)")
print(f"{'-'*80}")

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_cancer)
X_test_scaled = scaler.transform(X_test_cancer)

print(f"\nAfter scaling - Sample feature statistics:")
scaled_stats = pd.DataFrame({
    'Feature': cancer.feature_names[:5],
    'Mean': X_train_scaled[:, :5].mean(axis=0),
    'Std': X_train_scaled[:, :5].std(axis=0),
    'Min': X_train_scaled[:, :5].min(axis=0),
    'Max': X_train_scaled[:, :5].max(axis=0)
})
print(scaled_stats.to_string(index=False))
print("(All features now have mean ≈ 0 and std ≈ 1)")

# Train SVM with scaling
svm_with_scale = SVC(kernel='rbf', random_state=42)
svm_with_scale.fit(X_train_scaled, y_train_cancer)

# Predictions
y_train_pred_with_scale = svm_with_scale.predict(X_train_scaled)
y_test_pred_with_scale = svm_with_scale.predict(X_test_scaled)

# Calculate metrics
train_acc_with_scale = accuracy_score(y_train_cancer, y_train_pred_with_scale)
test_acc_with_scale = accuracy_score(y_test_cancer, y_test_pred_with_scale)

print(f"\nPerformance WITH Scaling:")
print(f"  Training Accuracy: {train_acc_with_scale:.4f}")
print(f"  Testing Accuracy:  {test_acc_with_scale:.4f}")

print(f"\nConfusion Matrix (With Scaling):")
cm_with_scale = confusion_matrix(y_test_cancer, y_test_pred_with_scale)
print(cm_with_scale)

print(f"\nClassification Report (With Scaling):")
print(classification_report(y_test_cancer, y_test_pred_with_scale, target_names=cancer.target_names))

# ============================================================================
# C) COMPARISON AND DISCUSSION
# ============================================================================

print(f"\n{'='*80}")
print("C) COMPARISON: WITH vs WITHOUT FEATURE SCALING")
print(f"{'='*80}")

comparison_df = pd.DataFrame({
    'Model': ['Without Scaling', 'With Scaling'],
    'Training Accuracy': [train_acc_no_scale, train_acc_with_scale],
    'Testing Accuracy': [test_acc_no_scale, test_acc_with_scale],
    'Improvement': [0, test_acc_with_scale - test_acc_no_scale]
})

print("\n", comparison_df.to_string(index=False))

print(f"\nAccuracy Improvement: {(test_acc_with_scale - test_acc_no_scale)*100:.2f}%")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Plot 1: Accuracy Comparison
ax1 = axes[0]
x_pos = [0, 1]
train_accs = [train_acc_no_scale, train_acc_with_scale]
test_accs = [test_acc_no_scale, test_acc_with_scale]

x_axis = np.arange(len(['Without Scaling', 'With Scaling']))
width = 0.35

ax1.bar(x_axis - width/2, train_accs, width, label='Training Accuracy', color='skyblue')
ax1.bar(x_axis + width/2, test_accs, width, label='Testing Accuracy', color='coral')

ax1.set_xlabel('Model', fontweight='bold')
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Accuracy Comparison: With vs Without Scaling', fontweight='bold')
ax1.set_xticks(x_axis)
ax1.set_xticklabels(['Without Scaling', 'With Scaling'])
ax1.legend()
ax1.set_ylim([0.6, 1.0])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    ax1.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Confusion Matrix without scaling
ax2 = axes[1]
sns.heatmap(cm_no_scale, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
ax2.set_title(f'Without Scaling\nAccuracy: {test_acc_no_scale:.4f}', fontweight='bold')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

# Plot 3: Confusion Matrix with scaling
ax3 = axes[2]
sns.heatmap(cm_with_scale, annot=True, fmt='d', cmap='Greens', ax=ax3,
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
ax3.set_title(f'With Scaling\nAccuracy: {test_acc_with_scale:.4f}', fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('/Users/arijitsingh/Documents/thapar_sem5/ml_lab/Ass_7/breast_cancer_scaling_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nScaling comparison plots saved as 'breast_cancer_scaling_comparison.png'")

# ============================================================================
# DISCUSSION
# ============================================================================

print(f"\n{'='*80}")
print("DISCUSSION: EFFECT OF FEATURE SCALING ON SVM PERFORMANCE")
print(f"{'='*80}")

print("""
Feature scaling has a SIGNIFICANT impact on SVM performance:

1. WHY SCALING MATTERS FOR SVM:
   - SVMs use distance-based computations to find the optimal hyperplane
   - Features with larger scales dominate the distance calculations
   - Without scaling, features with larger ranges have disproportionate influence
   - The RBF kernel computes: exp(-gamma * ||x - y||²)
   - Unscaled features make this calculation unstable and biased

2. OBSERVED RESULTS:
   - Without scaling: Training Acc = {:.4f}, Testing Acc = {:.4f}
   - With scaling: Training Acc = {:.4f}, Testing Acc = {:.4f}
   - Improvement: {:.2f}%

3. IMPACT ON THE BREAST CANCER DATASET:
   - The dataset has features with vastly different scales
   - Example: 'mean radius' (6-28) vs 'worst concave points' (0-0.3)
   - Without scaling, high-magnitude features dominate
   - Scaling ensures all features contribute proportionally

4. STANDARDSCALER EFFECT:
   - Transforms each feature to have mean = 0 and std = 1
   - All features now contribute equally to distance calculations
   - Improves convergence and model performance
   - Essential for RBF and polynomial kernels

5. BEST PRACTICES:
   - ALWAYS use feature scaling for SVM (especially with RBF/polynomial kernels)
   - StandardScaler is most common for SVM
   - Fit scaler only on training data to avoid data leakage
   - Apply same transformation to test data

6. CONCLUSION:
   Feature scaling is CRITICAL for SVM performance. It ensures fair contribution
   from all features and significantly improves model accuracy and generalization.
""".format(train_acc_no_scale, test_acc_no_scale,
           train_acc_with_scale, test_acc_with_scale,
           (test_acc_with_scale - test_acc_no_scale) * 100))

print(f"\n{'='*80}")
print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
print(f"{'='*80}")
print("\nGenerated files:")
print("1. iris_confusion_matrices.png")
print("2. iris_performance_comparison.png")
print("3. breast_cancer_scaling_comparison.png")
print("\nAll analysis and visualizations have been saved.")
print("="*80)
