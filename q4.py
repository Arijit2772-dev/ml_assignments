import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Question 4: Polynomial Regression")
print("="*80)

# Step 1: Generate synthetic dataset
print("\nStep 1: Generating Synthetic Dataset")
print("="*80)

np.random.seed(42)
n_samples = 100

# Generate data with non-linear relationship
X = np.sort(np.random.uniform(-3, 3, n_samples))
# True relationship: y = 0.5*x^3 - 2*x^2 + x + 3 + noise
y = 0.5 * X**3 - 2 * X**2 + X + 3 + np.random.normal(0, 2, n_samples)

# Reshape X for sklearn
X = X.reshape(-1, 1)

print(f"Dataset size: {n_samples} samples")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
print("True function: y = 0.5*x³ - 2*x² + x + 3 + noise")

# Step 2: Split the data
print("\n" + "="*80)
print("Step 2: Splitting Data")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 3: Train polynomial regression models with different degrees
print("\n" + "="*80)
print("Step 3: Training Polynomial Regression Models")
print("="*80)

degrees = [1, 2, 3, 4, 5, 10, 15]
results = []

print(f"Testing polynomial degrees: {degrees}\n")

for degree in degrees:
    print(f"Training degree {degree} polynomial...")
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Train linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'Degree': degree,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Model': model,
        'Poly Features': poly_features
    })
    
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print()

# Step 4: Results comparison
print("="*80)
print("Step 4: Model Comparison")
print("="*80)

results_df = pd.DataFrame([{
    'Degree': r['Degree'],
    'Train MSE': r['Train MSE'],
    'Test MSE': r['Test MSE'],
    'Train R²': r['Train R²'],
    'Test R²': r['Test R²']
} for r in results])

print(results_df.to_string(index=False))

# Find best model based on test R²
best_model_idx = results_df['Test R²'].idxmax()
best_degree = results_df.loc[best_model_idx, 'Degree']
print(f"\n✓ Best model: Degree {int(best_degree)} (Test R² = {results_df.loc[best_model_idx, 'Test R²']:.4f})")

# Step 5: Visualizations
print("\n" + "="*80)
print("Step 5: Creating Visualizations")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1-6: Individual polynomial fits
plot_degrees = [1, 2, 3, 4, 5, 10]
for idx, degree in enumerate(plot_degrees):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    # Get the model for this degree
    model_data = [r for r in results if r['Degree'] == degree][0]
    poly_features = model_data['Poly Features']
    model = model_data['Model']
    
    # Create smooth line for plotting
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    # Plot
    ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Train', edgecolors='k', linewidths=0.5)
    ax.scatter(X_test, y_test, alpha=0.6, s=30, label='Test', edgecolors='k', linewidths=0.5, color='orange')
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='Fitted curve')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Degree {degree}\nTest R² = {model_data["Test R²"]:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Highlight overfitting for high degrees
    if degree >= 10:
        ax.set_title(f'Degree {degree} (Overfitting)\nTest R² = {model_data["Test R²"]:.4f}', 
                     color='red', fontweight='bold')

# Plot 7: MSE vs Degree
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(results_df['Degree'], results_df['Train MSE'], 'o-', label='Train MSE', linewidth=2, markersize=8)
ax7.plot(results_df['Degree'], results_df['Test MSE'], 's-', label='Test MSE', linewidth=2, markersize=8)
ax7.set_xlabel('Polynomial Degree')
ax7.set_ylabel('Mean Squared Error')
ax7.set_title('MSE vs Polynomial Degree')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_xticks(degrees)

# Plot 8: R² vs Degree
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(results_df['Degree'], results_df['Train R²'], 'o-', label='Train R²', linewidth=2, markersize=8)
ax8.plot(results_df['Degree'], results_df['Test R²'], 's-', label='Test R²', linewidth=2, markersize=8)
ax8.set_xlabel('Polynomial Degree')
ax8.set_ylabel('R² Score')
ax8.set_title('R² Score vs Polynomial Degree')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_xticks(degrees)
ax8.axhline(y=results_df['Test R²'].max(), color='g', linestyle='--', alpha=0.5, label='Best Test R²')

# Plot 9: Overfitting Analysis (Train vs Test MSE)
ax9 = fig.add_subplot(gs[2, 2])
overfitting = results_df['Test MSE'] - results_df['Train MSE']
colors = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in overfitting]
ax9.bar(results_df['Degree'], overfitting, color=colors, alpha=0.7, edgecolor='black')
ax9.set_xlabel('Polynomial Degree')
ax9.set_ylabel('Test MSE - Train MSE')
ax9.set_title('Overfitting Analysis\n(Lower is Better)')
ax9.grid(True, alpha=0.3, axis='y')
ax9.set_xticks(degrees)
ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add text annotation
ax9.text(0.5, 0.95, 'Green: Good | Orange: Moderate | Red: Overfitting',
         transform=ax9.transAxes, ha='center', va='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Polynomial Regression Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('q4_polynomial_regression_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'q4_polynomial_regression_results.png'")
plt.show()

# Step 6: Detailed Analysis
print("\n" + "="*80)
print("Step 6: Detailed Analysis")
print("="*80)

print("\n1. Underfitting (Degree 1):")
degree_1 = results_df[results_df['Degree'] == 1].iloc[0]
print(f"   - Train R²: {degree_1['Train R²']:.4f}")
print(f"   - Test R²: {degree_1['Test R²']:.4f}")
print("   - Model is too simple to capture the non-linear relationship")

print("\n2. Good Fit (Degrees 2-4):")
good_fit = results_df[(results_df['Degree'] >= 2) & (results_df['Degree'] <= 4)]
print(good_fit[['Degree', 'Train R²', 'Test R²']].to_string(index=False))
print("   - Models capture the underlying pattern well")
print("   - Train and test performance are similar (no overfitting)")

print("\n3. Overfitting (Degrees 10+):")
overfit = results_df[results_df['Degree'] >= 10]
print(overfit[['Degree', 'Train R²', 'Test R²']].to_string(index=False))
print("   - Training performance is excellent but test performance degrades")
print("   - Model memorizes training data instead of learning the pattern")

# Step 7: Summary and Recommendations
print("\n" + "="*80)
print("Summary and Recommendations")
print("="*80)

print(f"\n✓ Best Polynomial Degree: {int(best_degree)}")
print(f"  - Test R²: {results_df.loc[best_model_idx, 'Test R²']:.4f}")
print(f"  - Test MSE: {results_df.loc[best_model_idx, 'Test MSE']:.4f}")

print("\nKey Insights:")
print("1. Linear model (degree 1) underfits - too simple for the data")
print(f"2. Polynomial degree {int(best_degree)} provides the best balance")
print("3. High-degree polynomials (10, 15) overfit - memorize training data")
print("4. The gap between train and test performance indicates overfitting")

print("\nBias-Variance Tradeoff:")
print("- Low degree → High bias, Low variance (Underfitting)")
print("- Optimal degree → Balanced bias and variance")
print("- High degree → Low bias, High variance (Overfitting)")

print("\nRecommendations:")
print("- Use cross-validation to select the optimal degree")
print("- Consider regularization (Ridge/Lasso) for high-degree polynomials")
print("- Monitor both training and test performance to detect overfitting")
print("- Simpler models are often better when performance is similar")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)