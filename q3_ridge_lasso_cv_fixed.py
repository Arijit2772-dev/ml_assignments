import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Question 3: Cross Validation for Ridge and Lasso Regression")
print("="*80)

# Step 1: Load Boston Housing Dataset
print("\nStep 1: Loading Boston Housing Dataset...")
print("Attempting to load from multiple sources...")

try:
    # Try loading from the original source
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)  # Fixed: added 'r' prefix
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Create DataFrame
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    
    print("✓ Dataset loaded from Carnegie Mellon repository")
    
except Exception as e:
    print(f"✗ Could not load from CMU: {e}")
    print("Creating sample dataset for demonstration...")
    
    # Create a sample dataset if loading fails
    np.random.seed(42)
    n_samples = 506
    df = pd.DataFrame({
        'CRIM': np.random.exponential(3, n_samples),
        'ZN': np.random.uniform(0, 100, n_samples),
        'INDUS': np.random.uniform(0, 28, n_samples),
        'CHAS': np.random.binomial(1, 0.07, n_samples),
        'NOX': np.random.uniform(0.3, 0.9, n_samples),
        'RM': np.random.normal(6.3, 0.7, n_samples),
        'AGE': np.random.uniform(0, 100, n_samples),
        'DIS': np.random.exponential(3.8, n_samples),
        'RAD': np.random.randint(1, 25, n_samples),
        'TAX': np.random.uniform(180, 720, n_samples),
        'PTRATIO': np.random.uniform(12, 22, n_samples),
        'B': np.random.uniform(0, 400, n_samples),
        'LSTAT': np.random.uniform(2, 38, n_samples)
    })
    df['MEDV'] = 35 - 0.5*df['CRIM'] + 0.8*df['RM'] - 0.6*df['LSTAT'] + np.random.normal(0, 3, n_samples)

# Display dataset information
print(f"Dataset Shape: {df.shape}")
print(f"Features: {feature_names}")
print(f"Target: MEDV")

print("\nDataset Description:")
descriptions = {
    'CRIM': 'per capita crime rate by town',
    'ZN': 'proportion of residential land zoned for lots over 25,000 sq.ft.',
    'INDUS': 'proportion of non-retail business acres per town',
    'CHAS': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
    'NOX': 'nitric oxides concentration (parts per 10 million)',
    'RM': 'average number of rooms per dwelling',
    'AGE': 'proportion of owner-occupied units built prior to 1940',
    'DIS': 'weighted distances to five Boston employment centres',
    'RAD': 'index of accessibility to radial highways',
    'TAX': 'full-value property-tax rate per $10,000',
    'PTRATIO': 'pupil-teacher ratio by town',
    'B': '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
    'LSTAT': '% lower status of the population',
    'MEDV': "Median value of owner-occupied homes in $1000's (TARGET)"
}

for feature, desc in descriptions.items():
    print(f"{feature:8s} - {desc}")

print("\nDataset Statistics:")
print(df.describe())

# Step 2: Prepare data
print("\n" + "="*80)
print("Step 2: Splitting and Scaling Data")
print("="*80)

X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# Step 3: Ridge Cross Validation
print("\n" + "="*80)
print("Step 3: Ridge Cross Validation (RidgeCV)")
print("="*80)

print("RidgeCV automatically selects the best alpha using cross-validation.")
alphas = np.logspace(-3, 3, 100)
print(f"Testing alpha values from {alphas.min():.3f} to {alphas.max():.0f}...")

print("Training RidgeCV with 5-fold cross-validation...")
ridge_cv = RidgeCV(
    alphas=alphas,
    cv=5  # Fixed: removed store_cv_results=True
)
ridge_cv.fit(X_train_scaled, y_train)

print(f"\n✓ Best alpha selected by RidgeCV: {ridge_cv.alpha_:.4f}")

# Make predictions
y_train_pred_ridge = ridge_cv.predict(X_train_scaled)
y_test_pred_ridge = ridge_cv.predict(X_test_scaled)

# Calculate metrics
train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)

print("\nRidge Regression Results:")
print(f"Training MSE: {train_mse_ridge:.4f}")
print(f"Test MSE: {test_mse_ridge:.4f}")
print(f"Training R²: {train_r2_ridge:.4f}")
print(f"Test R²: {test_r2_ridge:.4f}")

# Step 4: Lasso Cross Validation
print("\n" + "="*80)
print("Step 4: Lasso Cross Validation (LassoCV)")
print("="*80)

print("LassoCV automatically selects the best alpha using cross-validation.")
print(f"Testing alpha values from {alphas.min():.3f} to {alphas.max():.0f}...")

print("Training LassoCV with 5-fold cross-validation...")
lasso_cv = LassoCV(
    alphas=alphas,
    cv=5,
    max_iter=10000,
    random_state=42
)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\n✓ Best alpha selected by LassoCV: {lasso_cv.alpha_:.4f}")

# Make predictions
y_train_pred_lasso = lasso_cv.predict(X_train_scaled)
y_test_pred_lasso = lasso_cv.predict(X_test_scaled)

# Calculate metrics
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
train_r2_lasso = r2_score(y_train, y_train_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)

print("\nLasso Regression Results:")
print(f"Training MSE: {train_mse_lasso:.4f}")
print(f"Test MSE: {test_mse_lasso:.4f}")
print(f"Training R²: {train_r2_lasso:.4f}")
print(f"Test R²: {test_r2_lasso:.4f}")

# Feature selection with Lasso
lasso_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_cv.coef_
})
lasso_coefs = lasso_coefs.sort_values('Coefficient', key=abs, ascending=False)

print("\nLasso Coefficients (Feature Importance):")
print(lasso_coefs)

zero_coefs = (lasso_cv.coef_ == 0).sum()
print(f"\n✓ Number of features set to zero by Lasso: {zero_coefs}/{len(feature_names)}")

# Step 5: Comparison
print("\n" + "="*80)
print("Step 5: Model Comparison")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['Ridge', 'Lasso'],
    'Best Alpha': [ridge_cv.alpha_, lasso_cv.alpha_],
    'Train MSE': [train_mse_ridge, train_mse_lasso],
    'Test MSE': [test_mse_ridge, test_mse_lasso],
    'Train R²': [train_r2_ridge, train_r2_lasso],
    'Test R²': [test_r2_ridge, test_r2_lasso]
})

print(comparison.to_string(index=False))

# Step 6: Visualizations
print("\n" + "="*80)
print("Step 6: Creating Visualizations")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Ridge and Lasso Regression with Cross-Validation', fontsize=16, fontweight='bold')

# Plot 1: Ridge - Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_test_pred_ridge, alpha=0.6, edgecolors='k')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title(f'Ridge: Actual vs Predicted\nTest R² = {test_r2_ridge:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Lasso - Actual vs Predicted (Test Set)
axes[0, 1].scatter(y_test, y_test_pred_lasso, alpha=0.6, edgecolors='k', color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].set_title(f'Lasso: Actual vs Predicted\nTest R² = {test_r2_lasso:.4f}')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: MSE Comparison
models = ['Ridge', 'Lasso']
train_mses = [train_mse_ridge, train_mse_lasso]
test_mses = [test_mse_ridge, test_mse_lasso]

x = np.arange(len(models))
width = 0.35

axes[0, 2].bar(x - width/2, train_mses, width, label='Train MSE', alpha=0.8)
axes[0, 2].bar(x + width/2, test_mses, width, label='Test MSE', alpha=0.8)
axes[0, 2].set_xlabel('Model')
axes[0, 2].set_ylabel('MSE')
axes[0, 2].set_title('Mean Squared Error Comparison')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(models)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Plot 4: Ridge Residuals
residuals_ridge = y_test - y_test_pred_ridge
axes[1, 0].scatter(y_test_pred_ridge, residuals_ridge, alpha=0.6, edgecolors='k')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Ridge: Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Lasso Residuals
residuals_lasso = y_test - y_test_pred_lasso
axes[1, 1].scatter(y_test_pred_lasso, residuals_lasso, alpha=0.6, edgecolors='k', color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Lasso: Residual Plot')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Feature Coefficients Comparison
ridge_model = Ridge(alpha=ridge_cv.alpha_)
ridge_model.fit(X_train_scaled, y_train)

coef_comparison = pd.DataFrame({
    'Feature': feature_names,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_cv.coef_
})

x_pos = np.arange(len(feature_names))
axes[1, 2].barh(x_pos - 0.2, coef_comparison['Ridge'], 0.4, label='Ridge', alpha=0.8)
axes[1, 2].barh(x_pos + 0.2, coef_comparison['Lasso'], 0.4, label='Lasso', alpha=0.8)
axes[1, 2].set_yticks(x_pos)
axes[1, 2].set_yticklabels(feature_names)
axes[1, 2].set_xlabel('Coefficient Value')
axes[1, 2].set_title('Feature Coefficients Comparison')
axes[1, 2].legend()
axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('q3_ridge_lasso_cv_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'q3_ridge_lasso_cv_results.png'")
plt.show()

# Summary
print("\n" + "="*80)
print("Summary")
print("="*80)
print("\n1. Cross-Validation:")
print(f"   - Both models used 5-fold cross-validation")
print(f"   - Tested {len(alphas)} alpha values ranging from {alphas.min():.3f} to {alphas.max():.0f}")

print("\n2. Best Alpha Values:")
print(f"   - Ridge: {ridge_cv.alpha_:.4f}")
print(f"   - Lasso: {lasso_cv.alpha_:.4f}")

print("\n3. Performance:")
better_model = 'Ridge' if test_r2_ridge > test_r2_lasso else 'Lasso'
print(f"   - {better_model} performs slightly better on test set (R² score)")

print("\n4. Feature Selection:")
print(f"   - Lasso eliminated {zero_coefs} out of {len(feature_names)} features")
print(f"   - Ridge kept all features but with regularized coefficients")

print("\n5. Key Insights:")
print("   - RidgeCV and LassoCV automatically select optimal alpha values")
print("   - Lasso provides automatic feature selection (sparse solutions)")
print("   - Ridge is better when all features are potentially relevant")
print("   - Both models help prevent overfitting compared to ordinary least squares")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
