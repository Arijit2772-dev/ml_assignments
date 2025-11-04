import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Question 3: Cross Validation for Ridge and Lasso Regression")
print("=" * 80)

# Step 1: Load the Boston Housing dataset
print("\nStep 1: Loading Boston Housing Dataset...")
print("Attempting to load from multiple sources...")

try:
    # Try loading from sklearn (deprecated but may work)
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name='MEDV')
    print("✓ Dataset loaded from sklearn.datasets.load_boston")
except:
    try:
        # Try loading from alternative source
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        X = pd.DataFrame(data, columns=feature_names)
        y = pd.Series(target, name='MEDV')
        print("✓ Dataset loaded from Carnegie Mellon repository")
    except:
        try:
            # Try loading from OpenML
            from sklearn.datasets import fetch_openml
            boston_data = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
            X = boston_data.data
            y = boston_data.target
            print("✓ Dataset loaded from OpenML")
        except:
            # Create synthetic Boston-like dataset
            print("⚠ Creating synthetic Boston Housing dataset (original not accessible)")
            np.random.seed(42)
            n_samples = 506
            
            X = pd.DataFrame({
                'CRIM': np.random.exponential(3.6, n_samples),
                'ZN': np.random.uniform(0, 100, n_samples),
                'INDUS': np.random.uniform(0, 28, n_samples),
                'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
                'NOX': np.random.uniform(0.3, 0.9, n_samples),
                'RM': np.random.normal(6.3, 0.7, n_samples),
                'AGE': np.random.uniform(0, 100, n_samples),
                'DIS': np.random.uniform(1, 12, n_samples),
                'RAD': np.random.choice(range(1, 25), n_samples),
                'TAX': np.random.uniform(180, 720, n_samples),
                'PTRATIO': np.random.uniform(12, 22, n_samples),
                'B': np.random.uniform(0, 400, n_samples),
                'LSTAT': np.random.uniform(2, 38, n_samples)
            })
            
            # Create target based on important features
            y = (X['RM'] * 5 - X['LSTAT'] * 0.5 - X['CRIM'] * 0.3 + 
                 X['CHAS'] * 4 - X['NOX'] * 10 + np.random.randn(n_samples) * 2 + 10)
            y = pd.Series(y, name='MEDV')

print(f"\nDataset Shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target: {y.name}")

# Display dataset information
print("\nDataset Description:")
print("CRIM    - per capita crime rate by town")
print("ZN      - proportion of residential land zoned for lots over 25,000 sq.ft.")
print("INDUS   - proportion of non-retail business acres per town")
print("CHAS    - Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
print("NOX     - nitric oxides concentration (parts per 10 million)")
print("RM      - average number of rooms per dwelling")
print("AGE     - proportion of owner-occupied units built prior to 1940")
print("DIS     - weighted distances to five Boston employment centres")
print("RAD     - index of accessibility to radial highways")
print("TAX     - full-value property-tax rate per $10,000")
print("PTRATIO - pupil-teacher ratio by town")
print("B       - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
print("LSTAT   - % lower status of the population")
print("MEDV    - Median value of owner-occupied homes in $1000's (TARGET)")

print("\nDataset Statistics:")
print(X.describe())

# Step 2: Split and scale the data
print("\n" + "=" * 80)
print("Step 2: Splitting and Scaling Data")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# Step 3: Implement RidgeCV
print("\n" + "=" * 80)
print("Step 3: Ridge Cross Validation (RidgeCV)")
print("=" * 80)

print("\nRidgeCV automatically selects the best alpha using cross-validation.")
print("Testing alpha values from 0.001 to 1000...")

# Define range of alpha values
alphas = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000

# RidgeCV with 5-fold cross-validation and store_cv_values for compatibility
try:
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error', store_cv_values=True)
except:
    # Older sklearn versions don't have store_cv_values
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')

print("\nTraining RidgeCV with 5-fold cross-validation...")
ridge_cv.fit(X_train_scaled, y_train)

# Get best alpha
best_ridge_alpha = ridge_cv.alpha_
print(f"\n✓ Best Ridge Alpha selected by CV: {best_ridge_alpha:.4f}")

# Make predictions
y_train_pred_ridge = ridge_cv.predict(X_train_scaled)
y_test_pred_ridge = ridge_cv.predict(X_test_scaled)

# Calculate metrics
ridge_train_r2 = r2_score(y_train, y_train_pred_ridge)
ridge_test_r2 = r2_score(y_test, y_test_pred_ridge)
ridge_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge))
ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
ridge_train_mae = mean_absolute_error(y_train, y_train_pred_ridge)
ridge_test_mae = mean_absolute_error(y_test, y_test_pred_ridge)

print(f"\nRidgeCV Performance:")
print(f"  Train R²: {ridge_train_r2:.4f}")
print(f"  Test R²: {ridge_test_r2:.4f}")
print(f"  Train RMSE: {ridge_train_rmse:.4f}")
print(f"  Test RMSE: {ridge_test_rmse:.4f}")
print(f"  Train MAE: {ridge_train_mae:.4f}")
print(f"  Test MAE: {ridge_test_mae:.4f}")

# Step 4: Implement LassoCV
print("\n" + "=" * 80)
print("Step 4: Lasso Cross Validation (LassoCV)")
print("=" * 80)

print("\nLassoCV automatically selects the best alpha using cross-validation.")
print("Testing alpha values from 0.001 to 10...")

# Define range of alpha values for Lasso
lasso_alphas = np.logspace(-3, 1, 50)  # 50 values from 0.001 to 10

# LassoCV with 5-fold cross-validation
lasso_cv = LassoCV(alphas=lasso_alphas, cv=5, max_iter=10000, random_state=42)

print("\nTraining LassoCV with 5-fold cross-validation...")
lasso_cv.fit(X_train_scaled, y_train)

# Get best alpha
best_lasso_alpha = lasso_cv.alpha_
print(f"\n✓ Best Lasso Alpha selected by CV: {best_lasso_alpha:.4f}")

# Make predictions
y_train_pred_lasso = lasso_cv.predict(X_train_scaled)
y_test_pred_lasso = lasso_cv.predict(X_test_scaled)

# Calculate metrics
lasso_train_r2 = r2_score(y_train, y_train_pred_lasso)
lasso_test_r2 = r2_score(y_test, y_test_pred_lasso)
lasso_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_lasso))
lasso_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
lasso_train_mae = mean_absolute_error(y_train, y_train_pred_lasso)
lasso_test_mae = mean_absolute_error(y_test, y_test_pred_lasso)

print(f"\nLassoCV Performance:")
print(f"  Train R²: {lasso_train_r2:.4f}")
print(f"  Test R²: {lasso_test_r2:.4f}")
print(f"  Train RMSE: {lasso_train_rmse:.4f}")
print(f"  Test RMSE: {lasso_test_rmse:.4f}")
print(f"  Train MAE: {lasso_train_mae:.4f}")
print(f"  Test MAE: {lasso_test_mae:.4f}")

# Feature selection by Lasso
non_zero_coefs = np.sum(lasso_cv.coef_ != 0)
print(f"\nFeature Selection:")
print(f"  Non-zero coefficients: {non_zero_coefs}/{len(X.columns)}")
print(f"  Features eliminated: {len(X.columns) - non_zero_coefs}")

# Step 5: Compare Results
print("\n" + "=" * 80)
print("Step 5: Comparison of RidgeCV and LassoCV")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': ['RidgeCV', 'LassoCV'],
    'Best Alpha': [best_ridge_alpha, best_lasso_alpha],
    'Train R²': [ridge_train_r2, lasso_train_r2],
    'Test R²': [ridge_test_r2, lasso_test_r2],
    'Train RMSE': [ridge_train_rmse, lasso_train_rmse],
    'Test RMSE': [ridge_test_rmse, lasso_test_rmse],
    'Train MAE': [ridge_train_mae, lasso_train_mae],
    'Test MAE': [ridge_test_mae, lasso_test_mae]
})

print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Determine best model
if ridge_test_r2 > lasso_test_r2:
    best_model = "RidgeCV"
    best_r2 = ridge_test_r2
else:
    best_model = "LassoCV"
    best_r2 = lasso_test_r2

print(f"\n{'=' * 80}")
print(f"BEST MODEL: {best_model} (Test R² = {best_r2:.4f})")
print(f"{'=' * 80}")

# Coefficient comparison
print("\n" + "=" * 80)
print("Coefficient Comparison")
print("=" * 80)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'RidgeCV': ridge_cv.coef_,
    'LassoCV': lasso_cv.coef_
})

print("\nFeature Coefficients:")
print(coef_df.to_string(index=False))

print(f"\nFeatures with zero coefficient in LassoCV:")
zero_coef_features = coef_df[coef_df['LassoCV'] == 0]['Feature'].tolist()
if zero_coef_features:
    for feat in zero_coef_features:
        print(f"  - {feat}")
else:
    print("  None (all features retained)")

# Step 6: Visualizations
print("\n" + "=" * 80)
print("Step 6: Generating Visualizations")
print("=" * 80)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Alpha Path for Ridge (manually compute CV scores)
ax1 = fig.add_subplot(gs[0, :2])
ridge_cv_scores = []
for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(ridge_temp, X_train_scaled, y_train, cv=5, 
                             scoring='neg_mean_squared_error')
    ridge_cv_scores.append(-scores.mean())

ax1.semilogx(alphas, ridge_cv_scores, label='Mean CV MSE', color='blue', linewidth=2)
ax1.axvline(best_ridge_alpha, color='red', linestyle='--', 
            label=f'Best α = {best_ridge_alpha:.4f}', linewidth=2)
ax1.set_xlabel('Alpha (Regularization Parameter)')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('RidgeCV: Cross-Validation MSE vs Alpha')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. CV Path for Lasso
ax2 = fig.add_subplot(gs[0, 2])
mean_lasso_mse = np.mean(lasso_cv.mse_path_, axis=1)
std_lasso_mse = np.std(lasso_cv.mse_path_, axis=1)
ax2.semilogx(lasso_cv.alphas_, mean_lasso_mse, color='green', linewidth=2)
ax2.fill_between(lasso_cv.alphas_, 
                  mean_lasso_mse - std_lasso_mse,
                  mean_lasso_mse + std_lasso_mse, 
                  alpha=0.2, color='green')
ax2.axvline(best_lasso_alpha, color='red', linestyle='--',
            label=f'Best α = {best_lasso_alpha:.4f}', linewidth=2)
ax2.set_xlabel('Alpha')
ax2.set_ylabel('MSE')
ax2.set_title('LassoCV: MSE Path')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Performance Comparison
ax3 = fig.add_subplot(gs[1, 0])
metrics = ['Train R²', 'Test R²']
ridge_vals = [ridge_train_r2, ridge_test_r2]
lasso_vals = [lasso_train_r2, lasso_test_r2]

x_pos = np.arange(len(metrics))
width = 0.35
ax3.bar(x_pos - width/2, ridge_vals, width, label='RidgeCV', alpha=0.8)
ax3.bar(x_pos + width/2, lasso_vals, width, label='LassoCV', alpha=0.8)
ax3.set_ylabel('R² Score')
ax3.set_title('R² Score Comparison')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Coefficients Comparison
ax4 = fig.add_subplot(gs[1, 1:])
x_pos = np.arange(len(X.columns))
width = 0.4
ax4.bar(x_pos - width/2, ridge_cv.coef_, width, label='RidgeCV', alpha=0.8)
ax4.bar(x_pos + width/2, lasso_cv.coef_, width, label='LassoCV', alpha=0.8)
ax4.set_xlabel('Features')
ax4.set_ylabel('Coefficient Value')
ax4.set_title('Coefficient Comparison')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(X.columns, rotation=45, ha='right')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Ridge Predictions vs Actual
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(y_test, y_test_pred_ridge, alpha=0.6, edgecolors='k', linewidths=0.5)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax5.set_xlabel('Actual Values')
ax5.set_ylabel('Predicted Values')
ax5.set_title(f'RidgeCV: Predictions vs Actual\nR² = {ridge_test_r2:.4f}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Lasso Predictions vs Actual
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(y_test, y_test_pred_lasso, alpha=0.6, edgecolors='k', linewidths=0.5)
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax6.set_xlabel('Actual Values')
ax6.set_ylabel('Predicted Values')
ax6.set_title(f'LassoCV: Predictions vs Actual\nR² = {lasso_test_r2:.4f}')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Residual Plot
ax7 = fig.add_subplot(gs[2, 2])
ridge_residuals = y_test - y_test_pred_ridge
lasso_residuals = y_test - y_test_pred_lasso
ax7.scatter(y_test_pred_ridge, ridge_residuals, alpha=0.5, label='RidgeCV')
ax7.scatter(y_test_pred_lasso, lasso_residuals, alpha=0.5, label='LassoCV')
ax7.axhline(y=0, color='red', linestyle='--', lw=2)
ax7.set_xlabel('Predicted Values')
ax7.set_ylabel('Residuals')
ax7.set_title('Residual Plot')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.savefig('q3_ridge_lasso_cv_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q3_ridge_lasso_cv_analysis.png")

# Save results to CSV
comparison_df.to_csv('q3_model_comparison.csv', index=False)
print("✓ Saved: q3_model_comparison.csv")

coef_df.to_csv('q3_coefficients.csv', index=False)
print("✓ Saved: q3_coefficients.csv")

print("\n" + "=" * 80)
print("Question 3 Complete!")
print("=" * 80)

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)
print("""
1. RidgeCV and LassoCV automatically select optimal regularization parameters
2. RidgeCV retains all features with shrunk coefficients
3. LassoCV performs feature selection by setting some coefficients to zero
4. Cross-validation ensures the model generalizes well to unseen data
5. The best alpha balances bias-variance tradeoff
""")
