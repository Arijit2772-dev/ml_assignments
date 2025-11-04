import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Question 5: Regularization Comparison (Ridge vs Lasso vs ElasticNet)")
print("="*80)

# Step 1: Generate synthetic dataset with multicollinearity
print("\nStep 1: Generating Synthetic Dataset")
print("="*80)

np.random.seed(42)
n_samples = 200
n_features = 20

# Generate features with multicollinearity
X = np.random.randn(n_samples, n_features)

# Create multicollinearity: make some features highly correlated
X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.1  # Feature 1 ≈ Feature 0
X[:, 2] = X[:, 0] + np.random.randn(n_samples) * 0.1  # Feature 2 ≈ Feature 0
X[:, 5] = X[:, 4] + np.random.randn(n_samples) * 0.1  # Feature 5 ≈ Feature 4
X[:, 10] = X[:, 9] + np.random.randn(n_samples) * 0.1  # Feature 10 ≈ Feature 9

# Generate target with only some features being truly important
true_coefficients = np.zeros(n_features)
important_features = [0, 3, 7, 12, 15]
true_coefficients[important_features] = [5, -3, 4, -2, 3]

y = X @ true_coefficients + np.random.randn(n_samples) * 2

print(f"Dataset size: {n_samples} samples, {n_features} features")
print(f"True important features: {important_features}")
print(f"True coefficients: {true_coefficients[important_features]}")
print(f"Features with multicollinearity: (0,1,2), (4,5), (9,10)")

# Step 2: Split and scale data
print("\n" + "="*80)
print("Step 2: Splitting and Scaling Data")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("✓ Features scaled using StandardScaler")

# Step 3: Train models with different regularization techniques
print("\n" + "="*80)
print("Step 3: Training Regularization Models")
print("="*80)

# Define alpha values to test
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

results = []

# Test different alpha values for each model type
for alpha in alphas:
    print(f"\nTesting alpha = {alpha}...")
    
    # Ridge Regression
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge_train = ridge.predict(X_train_scaled)
    y_pred_ridge_test = ridge.predict(X_test_scaled)
    
    # Lasso Regression
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso_train = lasso.predict(X_train_scaled)
    y_pred_lasso_test = lasso.predict(X_test_scaled)
    
    # ElasticNet Regression (l1_ratio=0.5 means equal mix of L1 and L2)
    elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000, random_state=42)
    elastic.fit(X_train_scaled, y_train)
    y_pred_elastic_train = elastic.predict(X_train_scaled)
    y_pred_elastic_test = elastic.predict(X_test_scaled)
    
    # Store results
    results.append({
        'Alpha': alpha,
        'Model': 'Ridge',
        'Train MSE': mean_squared_error(y_train, y_pred_ridge_train),
        'Test MSE': mean_squared_error(y_test, y_pred_ridge_test),
        'Train R²': r2_score(y_train, y_pred_ridge_train),
        'Test R²': r2_score(y_test, y_pred_ridge_test),
        'MAE': mean_absolute_error(y_test, y_pred_ridge_test),
        'Non-zero Coefs': np.sum(np.abs(ridge.coef_) > 1e-5),
        'Coefficients': ridge.coef_,
        'Model Object': ridge
    })
    
    results.append({
        'Alpha': alpha,
        'Model': 'Lasso',
        'Train MSE': mean_squared_error(y_train, y_pred_lasso_train),
        'Test MSE': mean_squared_error(y_test, y_pred_lasso_test),
        'Train R²': r2_score(y_train, y_pred_lasso_train),
        'Test R²': r2_score(y_test, y_pred_lasso_test),
        'MAE': mean_absolute_error(y_test, y_pred_lasso_test),
        'Non-zero Coefs': np.sum(np.abs(lasso.coef_) > 1e-5),
        'Coefficients': lasso.coef_,
        'Model Object': lasso
    })
    
    results.append({
        'Alpha': alpha,
        'Model': 'ElasticNet',
        'Train MSE': mean_squared_error(y_train, y_pred_elastic_train),
        'Test MSE': mean_squared_error(y_test, y_pred_elastic_test),
        'Train R²': r2_score(y_train, y_pred_elastic_train),
        'Test R²': r2_score(y_test, y_pred_elastic_test),
        'MAE': mean_absolute_error(y_test, y_pred_elastic_test),
        'Non-zero Coefs': np.sum(np.abs(elastic.coef_) > 1e-5),
        'Coefficients': elastic.coef_,
        'Model Object': elastic
    })
    
    print(f"  Ridge - Test R²: {results[-3]['Test R²']:.4f}, Non-zero: {results[-3]['Non-zero Coefs']}")
    print(f"  Lasso - Test R²: {results[-2]['Test R²']:.4f}, Non-zero: {results[-2]['Non-zero Coefs']}")
    print(f"  ElasticNet - Test R²: {results[-1]['Test R²']:.4f}, Non-zero: {results[-1]['Non-zero Coefs']}")

# Also train ordinary Linear Regression for comparison
print("\nTraining Ordinary Linear Regression (no regularization)...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr_train = lr.predict(X_train_scaled)
y_pred_lr_test = lr.predict(X_test_scaled)

lr_result = {
    'Alpha': 0,
    'Model': 'Linear',
    'Train MSE': mean_squared_error(y_train, y_pred_lr_train),
    'Test MSE': mean_squared_error(y_test, y_pred_lr_test),
    'Train R²': r2_score(y_train, y_pred_lr_train),
    'Test R²': r2_score(y_test, y_pred_lr_test),
    'MAE': mean_absolute_error(y_test, y_pred_lr_test),
    'Non-zero Coefs': np.sum(np.abs(lr.coef_) > 1e-5),
    'Coefficients': lr.coef_,
    'Model Object': lr
}

print(f"  Linear - Test R²: {lr_result['Test R²']:.4f}, Non-zero: {lr_result['Non-zero Coefs']}")

# Step 4: Results comparison
print("\n" + "="*80)
print("Step 4: Model Comparison")
print("="*80)

results_df = pd.DataFrame([{
    'Model': r['Model'],
    'Alpha': r['Alpha'],
    'Train MSE': r['Train MSE'],
    'Test MSE': r['Test MSE'],
    'Train R²': r['Train R²'],
    'Test R²': r['Test R²'],
    'MAE': r['MAE'],
    'Non-zero Coefs': r['Non-zero Coefs']
} for r in results])

# Add Linear Regression result
lr_df = pd.DataFrame([{
    'Model': lr_result['Model'],
    'Alpha': lr_result['Alpha'],
    'Train MSE': lr_result['Train MSE'],
    'Test MSE': lr_result['Test MSE'],
    'Train R²': lr_result['Train R²'],
    'Test R²': lr_result['Test R²'],
    'MAE': lr_result['MAE'],
    'Non-zero Coefs': lr_result['Non-zero Coefs']
}])

full_results_df = pd.concat([lr_df, results_df], ignore_index=True)

print("\nAll Models Performance:")
print(full_results_df.to_string(index=False))

# Find best model
best_idx = results_df['Test R²'].idxmax()
best_model = results_df.loc[best_idx]
print(f"\n✓ Best regularized model: {best_model['Model']} (Alpha={best_model['Alpha']})")
print(f"  Test R²: {best_model['Test R²']:.4f}")
print(f"  Test MSE: {best_model['Test MSE']:.4f}")
print(f"  Non-zero coefficients: {best_model['Non-zero Coefs']}/{n_features}")

# Step 5: Visualizations
print("\n" + "="*80)
print("Step 5: Creating Visualizations")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Plot 1: Test R² vs Alpha for all models
ax1 = fig.add_subplot(gs[0, 0])
for model_type in ['Ridge', 'Lasso', 'ElasticNet']:
    model_data = results_df[results_df['Model'] == model_type]
    ax1.plot(model_data['Alpha'], model_data['Test R²'], 'o-', 
             label=model_type, linewidth=2, markersize=8)
ax1.axhline(y=lr_result['Test R²'], color='black', linestyle='--', 
            label='Linear (no reg.)', linewidth=2)
ax1.set_xlabel('Alpha (Regularization Strength)')
ax1.set_ylabel('Test R² Score')
ax1.set_title('Test R² vs Alpha')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Test MSE vs Alpha
ax2 = fig.add_subplot(gs[0, 1])
for model_type in ['Ridge', 'Lasso', 'ElasticNet']:
    model_data = results_df[results_df['Model'] == model_type]
    ax2.plot(model_data['Alpha'], model_data['Test MSE'], 'o-', 
             label=model_type, linewidth=2, markersize=8)
ax2.axhline(y=lr_result['Test MSE'], color='black', linestyle='--', 
            label='Linear (no reg.)', linewidth=2)
ax2.set_xlabel('Alpha (Regularization Strength)')
ax2.set_ylabel('Test MSE')
ax2.set_title('Test MSE vs Alpha')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Number of Non-zero Coefficients vs Alpha
ax3 = fig.add_subplot(gs[0, 2])
for model_type in ['Ridge', 'Lasso', 'ElasticNet']:
    model_data = results_df[results_df['Model'] == model_type]
    ax3.plot(model_data['Alpha'], model_data['Non-zero Coefs'], 'o-', 
             label=model_type, linewidth=2, markersize=8)
ax3.axhline(y=lr_result['Non-zero Coefs'], color='black', linestyle='--', 
            label='Linear (no reg.)', linewidth=2)
ax3.set_xlabel('Alpha (Regularization Strength)')
ax3.set_ylabel('Number of Non-zero Coefficients')
ax3.set_title('Feature Selection vs Alpha')
ax3.set_xscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4-6: Coefficient paths for Ridge, Lasso, ElasticNet
plot_idx = 3
for model_type in ['Ridge', 'Lasso', 'ElasticNet']:
    row = (plot_idx) // 3
    col = (plot_idx) % 3
    ax = fig.add_subplot(gs[row, col])
    
    model_data = results_df[results_df['Model'] == model_type]
    
    for feature_idx in range(n_features):
        coefs = [r['Coefficients'][feature_idx] for r in results if r['Model'] == model_type]
        
        # Highlight true important features
        if feature_idx in important_features:
            ax.plot(alphas, coefs, linewidth=2.5, label=f'Feature {feature_idx}*', alpha=0.8)
        else:
            ax.plot(alphas, coefs, linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{model_type}: Coefficient Paths\n(* = True Important Features)')
    ax.set_xscale('log')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    if model_type == 'Ridge':
        ax.legend(fontsize=7, loc='best')
    
    plot_idx += 1

# Plot 7: Coefficient comparison at optimal alpha
ax7 = fig.add_subplot(gs[2, 0])
optimal_alpha = 1.0  # Choose a reasonable alpha for comparison
ridge_coefs = [r['Coefficients'] for r in results if r['Model'] == 'Ridge' and r['Alpha'] == optimal_alpha][0]
lasso_coefs = [r['Coefficients'] for r in results if r['Model'] == 'Lasso' and r['Alpha'] == optimal_alpha][0]
elastic_coefs = [r['Coefficients'] for r in results if r['Model'] == 'ElasticNet' and r['Alpha'] == optimal_alpha][0]

x = np.arange(n_features)
width = 0.25

ax7.bar(x - width, ridge_coefs, width, label='Ridge', alpha=0.8)
ax7.bar(x, lasso_coefs, width, label='Lasso', alpha=0.8)
ax7.bar(x + width, elastic_coefs, width, label='ElasticNet', alpha=0.8)

# Highlight true important features
for feat in important_features:
    ax7.axvline(x=feat, color='red', linestyle='--', alpha=0.3, linewidth=2)

ax7.set_xlabel('Feature Index')
ax7.set_ylabel('Coefficient Value')
ax7.set_title(f'Coefficient Comparison (Alpha={optimal_alpha})\nRed lines = True Important Features')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')
ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 8: Train vs Test R² comparison
ax8 = fig.add_subplot(gs[2, 1])
models_list = ['Linear', 'Ridge\n(α=1)', 'Lasso\n(α=1)', 'ElasticNet\n(α=1)']
train_r2_list = [
    lr_result['Train R²'],
    results_df[(results_df['Model'] == 'Ridge') & (results_df['Alpha'] == 1.0)]['Train R²'].values[0],
    results_df[(results_df['Model'] == 'Lasso') & (results_df['Alpha'] == 1.0)]['Train R²'].values[0],
    results_df[(results_df['Model'] == 'ElasticNet') & (results_df['Alpha'] == 1.0)]['Train R²'].values[0]
]
test_r2_list = [
    lr_result['Test R²'],
    results_df[(results_df['Model'] == 'Ridge') & (results_df['Alpha'] == 1.0)]['Test R²'].values[0],
    results_df[(results_df['Model'] == 'Lasso') & (results_df['Alpha'] == 1.0)]['Test R²'].values[0],
    results_df[(results_df['Model'] == 'ElasticNet') & (results_df['Alpha'] == 1.0)]['Test R²'].values[0]
]

x = np.arange(len(models_list))
width = 0.35

ax8.bar(x - width/2, train_r2_list, width, label='Train R²', alpha=0.8)
ax8.bar(x + width/2, test_r2_list, width, label='Test R²', alpha=0.8)
ax8.set_ylabel('R² Score')
ax8.set_title('Train vs Test R² Comparison')
ax8.set_xticks(x)
ax8.set_xticklabels(models_list)
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Overfitting analysis
ax9 = fig.add_subplot(gs[2, 2])
overfitting = [train - test for train, test in zip(train_r2_list, test_r2_list)]
colors = ['red' if lr_result['Model'] == 'Linear' else 'green' for _ in range(len(models_list))]
colors[0] = 'red'  # Linear regression
ax9.bar(models_list, overfitting, color=colors, alpha=0.7, edgecolor='black')
ax9.set_ylabel('Train R² - Test R² (Overfitting)')
ax9.set_title('Overfitting Analysis\n(Lower is Better)')
ax9.grid(True, alpha=0.3, axis='y')
ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 10: Feature importance heatmap
ax10 = fig.add_subplot(gs[3, :])
coef_matrix = np.array([
    true_coefficients,
    lr_result['Coefficients'],
    ridge_coefs,
    lasso_coefs,
    elastic_coefs
])

im = ax10.imshow(coef_matrix, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
ax10.set_yticks(range(5))
ax10.set_yticklabels(['True', 'Linear', 'Ridge', 'Lasso', 'ElasticNet'])
ax10.set_xticks(range(n_features))
ax10.set_xticklabels(range(n_features))
ax10.set_xlabel('Feature Index')
ax10.set_title('Coefficient Heatmap Comparison (Alpha=1.0)')

# Highlight true important features
for feat in important_features:
    ax10.axvline(x=feat-0.5, color='yellow', linestyle='-', linewidth=3, alpha=0.7)
    ax10.axvline(x=feat+0.5, color='yellow', linestyle='-', linewidth=3, alpha=0.7)

plt.colorbar(im, ax=ax10, label='Coefficient Value')

# Add text annotations for non-zero values
for i in range(5):
    for j in range(n_features):
        if abs(coef_matrix[i, j]) > 0.5:
            text = ax10.text(j, i, f'{coef_matrix[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=7)

plt.suptitle('Regularization Comparison: Ridge vs Lasso vs ElasticNet', 
             fontsize=18, fontweight='bold', y=0.998)
plt.savefig('q5_regularization_comparison_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'q5_regularization_comparison_results.png'")
plt.show()

# Step 6: Detailed Analysis
print("\n" + "="*80)
print("Step 6: Detailed Analysis")
print("="*80)

print("\n1. Linear Regression (No Regularization):")
print(f"   - Test R²: {lr_result['Test R²']:.4f}")
print(f"   - Test MSE: {lr_result['Test MSE']:.4f}")
print(f"   - All {n_features} features have non-zero coefficients")
print("   - May suffer from overfitting due to multicollinearity")

print("\n2. Ridge Regression (L2 Regularization):")
ridge_best = results_df[results_df['Model'] == 'Ridge'].loc[results_df[results_df['Model'] == 'Ridge']['Test R²'].idxmax()]
print(f"   - Best Alpha: {ridge_best['Alpha']}")
print(f"   - Test R²: {ridge_best['Test R²']:.4f}")
print(f"   - Shrinks all coefficients but keeps all features")
print(f"   - All {ridge_best['Non-zero Coefs']} features remain non-zero")
print("   - Good for dealing with multicollinearity")

print("\n3. Lasso Regression (L1 Regularization):")
lasso_best = results_df[results_df['Model'] == 'Lasso'].loc[results_df[results_df['Model'] == 'Lasso']['Test R²'].idxmax()]
print(f"   - Best Alpha: {lasso_best['Alpha']}")
print(f"   - Test R²: {lasso_best['Test R²']:.4f}")
print(f"   - Performs automatic feature selection")
print(f"   - Selected {lasso_best['Non-zero Coefs']} out of {n_features} features")
print("   - Sets irrelevant features to exactly zero")

print("\n4. ElasticNet Regression (L1 + L2 Regularization):")
elastic_best = results_df[results_df['Model'] == 'ElasticNet'].loc[results_df[results_df['Model'] == 'ElasticNet']['Test R²'].idxmax()]
print(f"   - Best Alpha: {elastic_best['Alpha']}")
print(f"   - Test R²: {elastic_best['Test R²']:.4f}")
print(f"   - Combines benefits of Ridge and Lasso")
print(f"   - Selected {elastic_best['Non-zero Coefs']} out of {n_features} features")
print("   - Good when features are correlated and want feature selection")

# Step 7: Summary
print("\n" + "="*80)
print("Summary and Recommendations")
print("="*80)

print(f"\n✓ Dataset Characteristics:")
print(f"  - {n_samples} samples, {n_features} features")
print(f"  - {len(important_features)} truly important features: {important_features}")
print(f"  - Multicollinearity present in features")

print("\n✓ Key Findings:")
print("  1. All regularization methods improve generalization over Linear Regression")
print("  2. Lasso successfully identifies important features through sparsity")
print("  3. Ridge maintains all features but with smaller coefficients")
print("  4. ElasticNet provides a balance between Ridge and Lasso")

print("\n✓ When to Use Each Method:")
print("  - Ridge: When all features are potentially relevant, multicollinearity present")
print("  - Lasso: When feature selection is important, sparse solutions desired")
print("  - ElasticNet: When you have correlated features AND want feature selection")
print("  - Linear: Only when features are truly independent and n >> p")

print("\n✓ Regularization Effects:")
print("  - Low Alpha (weak regularization): Similar to Linear Regression")
print("  - Medium Alpha: Good balance between bias and variance")
print("  - High Alpha (strong regularization): May underfit, too much shrinkage")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)