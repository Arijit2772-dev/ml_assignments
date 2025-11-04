import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset with highly correlated features
def generate_correlated_data(n_samples=1000, n_features=7):
    """
    Generate a dataset with highly correlated features
    """
    # Generate base features
    base_feature = np.random.randn(n_samples, 1)
    
    # Create correlated features by adding small noise to base feature
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        noise = np.random.randn(n_samples, 1) * 0.3  # Small noise for high correlation
        X[:, i] = (base_feature + noise).flatten()
    
    # Generate target variable as linear combination of features plus noise
    true_weights = np.random.randn(n_features, 1) * 2
    y = X @ true_weights + np.random.randn(n_samples, 1) * 0.5
    y = y.flatten()
    
    return X, y

# Ridge Regression Cost Function
def ridge_cost(X, y, theta, lambda_reg):
    """
    Calculate Ridge Regression cost (MSE + L2 penalty)
    """
    m = len(y)
    predictions = X @ theta
    mse = (1/(2*m)) * np.sum((predictions - y)**2)
    l2_penalty = (lambda_reg/(2*m)) * np.sum(theta[1:]**2)  # Don't penalize intercept
    return mse + l2_penalty

# Gradient Descent for Ridge Regression
def ridge_gradient_descent(X, y, learning_rate, lambda_reg, n_iterations=1000):
    """
    Implement Ridge Regression using Gradient Descent
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for iteration in range(n_iterations):
        # Predictions
        predictions = X @ theta
        
        # Calculate gradients
        errors = predictions - y
        gradients = (1/m) * (X.T @ errors)
        
        # Add L2 regularization to gradients (except for intercept term)
        reg_term = np.zeros(n)
        reg_term[1:] = (lambda_reg/m) * theta[1:]
        gradients += reg_term
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Calculate and store cost
        cost = ridge_cost(X, y, theta, lambda_reg)
        cost_history.append(cost)
    
    return theta, cost_history

# Calculate R2 Score
def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared score
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

print("=" * 80)
print("Question 1: Ridge Regression using Gradient Descent Optimization")
print("=" * 80)

# Generate dataset
print("\nStep 1: Generating synthetic dataset with 7 highly correlated features...")
X, y = generate_correlated_data(n_samples=1000, n_features=7)

# Check correlation
print("\nCorrelation Matrix of Features:")
corr_matrix = np.corrcoef(X.T)
print(pd.DataFrame(corr_matrix, 
                   columns=[f'Feature_{i+1}' for i in range(7)],
                   index=[f'Feature_{i+1}' for i in range(7)]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add intercept term (bias)
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")
print(f"Number of features (including intercept): {X_train_scaled.shape[1]}")

# Define hyperparameter grids
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lambda_values = [1e-15, 1e-10, 1e-5, 1e-3, 0, 1, 10, 20]

print("\n" + "=" * 80)
print("Step 2: Grid Search for Best Hyperparameters")
print("=" * 80)

# Store results
results = []

print("\nTesting combinations of learning rates and regularization parameters...")
print("(This may take a moment...)\n")

# Grid search
for lr in learning_rates:
    for lambda_reg in lambda_values:
        try:
            # Train model
            theta, cost_history = ridge_gradient_descent(
                X_train_scaled, y_train, lr, lambda_reg, n_iterations=1000
            )
            
            # Calculate final cost
            final_cost = cost_history[-1]
            
            # Make predictions
            y_train_pred = X_train_scaled @ theta
            y_test_pred = X_test_scaled @ theta
            
            # Calculate R2 scores
            train_r2 = calculate_r2(y_train, y_train_pred)
            test_r2 = calculate_r2(y_test, y_test_pred)
            
            # Check if cost converged (not NaN or Inf)
            if np.isfinite(final_cost) and np.isfinite(test_r2):
                results.append({
                    'learning_rate': lr,
                    'lambda': lambda_reg,
                    'final_cost': final_cost,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'theta': theta,
                    'cost_history': cost_history
                })
        except:
            # Skip combinations that cause overflow or other errors
            continue

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("=" * 80)
print("Top 10 Configurations by Test R² Score:")
print("=" * 80)
top_results = results_df.nlargest(10, 'test_r2')[['learning_rate', 'lambda', 'final_cost', 'test_r2']]
print(top_results.to_string(index=False))

# Find best parameters
best_result = results_df.loc[results_df['test_r2'].idxmax()]

print("\n" + "=" * 80)
print("BEST HYPERPARAMETERS:")
print("=" * 80)
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Regularization Parameter (λ): {best_result['lambda']}")
print(f"Final Cost: {best_result['final_cost']:.6f}")
print(f"Train R² Score: {best_result['train_r2']:.6f}")
print(f"Test R² Score: {best_result['test_r2']:.6f}")
print(f"\nLearned Coefficients (θ):")
print(f"Intercept: {best_result['theta'][0]:.4f}")
for i, coef in enumerate(best_result['theta'][1:], 1):
    print(f"Feature {i}: {coef:.4f}")

# Plot cost convergence for best model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(best_result['cost_history'])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title(f'Cost Convergence (LR={best_result["learning_rate"]}, λ={best_result["lambda"]})')
plt.grid(True)

# Plot predictions vs actual for best model
plt.subplot(1, 2, 2)
y_test_pred = X_test_scaled @ best_result['theta']
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Predictions vs Actual (Test Set)\nR² = {best_result["test_r2"]:.4f}')
plt.grid(True)

plt.tight_layout()
plt.savefig('q1_ridge_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: q1_ridge_results.png")

# Create heatmap of R2 scores across hyperparameters
pivot_table = results_df.pivot_table(
    values='test_r2', 
    index='lambda', 
    columns='learning_rate'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Test R² Score'})
plt.title('Test R² Score Heatmap\n(Learning Rate vs Regularization Parameter)')
plt.xlabel('Learning Rate')
plt.ylabel('Regularization Parameter (λ)')
plt.tight_layout()
plt.savefig('q1_hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Heatmap saved: q1_hyperparameter_heatmap.png")

# Save detailed results to CSV
results_df[['learning_rate', 'lambda', 'final_cost', 'train_r2', 'test_r2']].to_csv(
    'q1_detailed_results.csv', index=False
)
print("✓ Detailed results saved: q1_detailed_results.csv")

print("\n" + "=" * 80)
print("Question 1 Complete!")
print("=" * 80)
