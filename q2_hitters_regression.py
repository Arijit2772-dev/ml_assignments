import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Question 2: Linear, Ridge, and Lasso Regression on Hitters Dataset")
print("=" * 80)

# Step 1: Load the dataset
print("\nStep 1: Loading Hitters Dataset...")
print("Attempting to load from multiple sources...")

try:
    # Try loading from ISLP package
    from ISLP import load_data
    df = load_data('Hitters')
    print("✓ Dataset loaded from ISLP package")
except:
    try:
        # Try loading from URL (GitHub)
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"
        df = pd.read_csv(url)
        print("✓ Dataset loaded from GitHub")
    except:
        # Create a sample Hitters dataset if all else fails
        print("⚠ Creating sample Hitters dataset (original not accessible)")
        np.random.seed(42)
        n_samples = 322
        
        df = pd.DataFrame({
            'AtBat': np.random.randint(16, 700, n_samples),
            'Hits': np.random.randint(1, 240, n_samples),
            'HmRun': np.random.randint(0, 50, n_samples),
            'Runs': np.random.randint(0, 150, n_samples),
            'RBI': np.random.randint(0, 150, n_samples),
            'Walks': np.random.randint(0, 150, n_samples),
            'Years': np.random.randint(1, 25, n_samples),
            'CAtBat': np.random.randint(0, 5000, n_samples),
            'CHits': np.random.randint(0, 2500, n_samples),
            'CHmRun': np.random.randint(0, 600, n_samples),
            'CRuns': np.random.randint(0, 2000, n_samples),
            'CRBI': np.random.randint(0, 2000, n_samples),
            'CWalks': np.random.randint(0, 1600, n_samples),
            'League': np.random.choice(['A', 'N'], n_samples),
            'Division': np.random.choice(['E', 'W'], n_samples),
            'PutOuts': np.random.randint(0, 1500, n_samples),
            'Assists': np.random.randint(0, 500, n_samples),
            'Errors': np.random.randint(0, 35, n_samples),
            'NewLeague': np.random.choice(['A', 'N'], n_samples)
        })
        # Create salary based on performance metrics
        df['Salary'] = (df['Hits'] * 2 + df['HmRun'] * 5 + df['RBI'] * 1.5 + 
                       df['Years'] * 20 + np.random.randn(n_samples) * 100 + 200)
        # Add some missing values
        missing_indices = np.random.choice(n_samples, 59, replace=False)
        df.loc[missing_indices, 'Salary'] = np.nan

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Step 2: Data Preprocessing
print("\n" + "=" * 80)
print("Step 2(a): Data Preprocessing")
print("=" * 80)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing Salary (target variable)
print(f"\nRows before removing missing values: {len(df)}")
df = df.dropna(subset=['Salary'])
print(f"Rows after removing missing values: {len(df)}")

# Handle any remaining missing values in features (if any)
if df.isnull().sum().sum() > 0:
    print("\nFilling remaining missing values with median...")
    df = df.fillna(df.median(numeric_only=True))

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target from numerical columns
if 'Salary' in numerical_cols:
    numerical_cols.remove('Salary')

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns (features): {numerical_cols}")

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

print("\n✓ Preprocessing complete!")

# Step 3: Separate features and target, and perform scaling
print("\n" + "=" * 80)
print("Step 2(b): Separating Features and Target, Performing Scaling")
print("=" * 80)

# Separate input and output
X = df.drop('Salary', axis=1)
y = df['Salary']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features scaled using StandardScaler")

# Step 4: Train models
print("\n" + "=" * 80)
print("Step 2(c): Training Linear, Ridge, and Lasso Regression Models")
print("=" * 80)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=0.5748),
    'Lasso Regression': Lasso(alpha=0.5748, max_iter=10000)
}

# Train and store results
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results[model_name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_test_pred': y_test_pred
    }
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")

print("\n✓ All models trained successfully!")

# Step 5: Evaluate and compare models
print("\n" + "=" * 80)
print("Step 2(d): Model Evaluation and Comparison")
print("=" * 80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train R²': [results[m]['train_r2'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Train RMSE': [results[m]['train_rmse'] for m in results.keys()],
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'Train MAE': [results[m]['train_mae'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()]
})

print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
print(f"\n{'=' * 80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'=' * 80}")
print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"Test RMSE: {results[best_model_name]['test_rmse']:.2f}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.2f}")

# Explanation
print(f"\n{'=' * 80}")
print("WHY THIS MODEL PERFORMS BEST:")
print(f"{'=' * 80}")

if best_model_name == 'Linear Regression':
    print("""
Linear Regression performed best because:
1. The relationship between features and salary is primarily linear
2. No significant overfitting is present in the data
3. The features don't require heavy regularization
4. The model has optimal bias-variance tradeoff for this dataset
    """)
elif best_model_name == 'Ridge Regression':
    print("""
Ridge Regression performed best because:
1. L2 regularization helps handle multicollinearity in baseball statistics
2. Many features are correlated (e.g., hits correlate with at-bats)
3. Ridge shrinks coefficients but keeps all features
4. The regularization parameter (0.5748) provides good bias-variance tradeoff
5. Prevents overfitting while maintaining model interpretability
    """)
else:
    print("""
Lasso Regression performed best because:
1. L1 regularization performs feature selection by zeroing out coefficients
2. Some features may not be important for predicting salary
3. Provides a sparse model that's easier to interpret
4. The regularization parameter (0.5748) effectively identifies key features
5. Reduces model complexity while maintaining performance
    """)

# Coefficient comparison
print(f"\n{'=' * 80}")
print("Coefficient Comparison")
print(f"{'=' * 80}")

coef_comparison = pd.DataFrame({
    'Feature': X.columns,
    'Linear': results['Linear Regression']['model'].coef_,
    'Ridge': results['Ridge Regression']['model'].coef_,
    'Lasso': results['Lasso Regression']['model'].coef_
})

print("\nTop 10 Features (by Linear Regression):")
top_features = coef_comparison.reindex(coef_comparison['Linear'].abs().sort_values(ascending=False).index).head(10)
print(top_features.to_string(index=False))

# Visualizations
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results))
width = 0.35
ax1.bar(x_pos - width/2, [results[m]['train_r2'] for m in results.keys()], 
        width, label='Train', alpha=0.8)
ax1.bar(x_pos + width/2, [results[m]['test_r2'] for m in results.keys()], 
        width, label='Test', alpha=0.8)
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results.keys(), rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RMSE comparison
ax2 = axes[0, 1]
ax2.bar(x_pos - width/2, [results[m]['train_rmse'] for m in results.keys()], 
        width, label='Train', alpha=0.8)
ax2.bar(x_pos + width/2, [results[m]['test_rmse'] for m in results.keys()], 
        width, label='Test', alpha=0.8)
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results.keys(), rotation=15, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Predictions vs Actual
ax3 = axes[1, 0]
ax3.scatter(y_test, results[best_model_name]['y_test_pred'], alpha=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Salary')
ax3.set_ylabel('Predicted Salary')
ax3.set_title(f'{best_model_name}\nPredictions vs Actual')
ax3.grid(True, alpha=0.3)

# Residuals
ax4 = axes[1, 1]
residuals = y_test - results[best_model_name]['y_test_pred']
ax4.scatter(results[best_model_name]['y_test_pred'], residuals, alpha=0.5)
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Salary')
ax4.set_ylabel('Residuals')
ax4.set_title(f'{best_model_name}\nResidual Plot')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q2_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q2_model_comparison.png")

# Coefficient plot
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(X.columns))
width = 0.25
ax.bar(x_pos - width, coef_comparison['Linear'], width, label='Linear', alpha=0.8)
ax.bar(x_pos, coef_comparison['Ridge'], width, label='Ridge', alpha=0.8)
ax.bar(x_pos + width, coef_comparison['Lasso'], width, label='Lasso', alpha=0.8)
ax.set_ylabel('Coefficient Value')
ax.set_title('Model Coefficients Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(X.columns, rotation=90)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_coefficients.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q2_coefficients.png")

comparison_df.to_csv('q2_results.csv', index=False)
print("✓ Saved: q2_results.csv")

print("\n" + "=" * 80)
print("Question 2 Complete!")
print("=" * 80)
