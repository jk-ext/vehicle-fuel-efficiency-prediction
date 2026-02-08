import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS, add_constant
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('auto-mpg.csv')

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values in horsepower (marked as '?')
df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = df['horsepower'].astype(float)
print(f"\nHorsepower missing values: {df['horsepower'].isnull().sum()}")

# Drop rows with missing horsepower
df = df.dropna(subset=['horsepower'])

# Drop 'car name' column as it's not useful for prediction
df = df.drop('car name', axis=1)

# Convert 'origin' to categorical
df['origin'] = df['origin'].astype('category')

# Display data types
print("\nData types:")
print(df.dtypes)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check the shape
print(f"\nDataset shape: {df.shape}")

# Statistical Analyses

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

# Linear regression tests - using OLS to examine relationships
# For simplicity, let's do pairwise regressions with mpg as dependent
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
for feature in features:
    X = df[[feature]]
    X = add_constant(X)
    y = df['mpg']
    model = OLS(y, X).fit()
    print(f"\nRegression of MPG on {feature}:")
    print(f"R-squared: {model.rsquared:.3f}")
    print(f"Coefficient: {model.params[feature]:.3f}")
    print(f"P-value: {model.pvalues[feature]:.3f}")

# Check for multicollinearity using VIF
X = df[features]
X = add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)
# Prepare data for machine learning
# One-hot encode 'origin'
df_encoded = pd.get_dummies(df, columns=['origin'], drop_first=True)

# Features and target
X = df_encoded.drop('mpg', axis=1)
y = df_encoded['mpg']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

trained_models = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    predictions[name] = model.predict(X_test)

# Evaluate models
evaluation_results = {}

for name, pred in predictions.items():
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    evaluation_results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}
    print(f"\n{name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")

# Display results in a DataFrame
results_df = pd.DataFrame(evaluation_results).T
print("\nModel Evaluation Summary:")
print(results_df)

# Visualize predictions vs actual values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, pred) in enumerate(predictions.items()):
    ax = axes[i]
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title(f'{name}: Actual vs Predicted MPG')
    ax.grid(True)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
plt.show()

# Interpret results and influential factors

# Feature importance from Random Forest
rf_model = trained_models['Random Forest Regressor']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

# Coefficients from Linear Regression
lr_model = trained_models['Linear Regression']
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', ascending=False)

print("\nLinear Regression Coefficients:")
print(coefficients)

# Insights
print("\nKey Insights:")
print("1. The most influential factors affecting fuel efficiency are weight, displacement, and horsepower.")
print("2. Heavier vehicles and those with larger engines consume more fuel.")
print("3. Model year shows positive correlation, indicating newer vehicles are more efficient.")
print("4. Random Forest performs best among the models, suggesting non-linear relationships.")
print("5. For sustainable automotive design: Focus on reducing vehicle weight, optimizing engine displacement, and improving horsepower efficiency.")
print("6. Encourage adoption of lighter materials and efficient engine technologies to improve MPG.")
