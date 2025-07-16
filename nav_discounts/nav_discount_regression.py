import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stargazer.stargazer import Stargazer
import datetime

# Load the combined dataset
print("Loading the combined dataset...")
combined_df = pd.read_csv(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_discount_drivers_combined_monthly.csv")

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Number of observations: {len(combined_df)}")
print("\nColumns in the dataset:")
print(combined_df.columns.tolist())
print("\nFirst few rows of the data:")
print(combined_df.head())

# Prepare variables for regression
print("\nPreparing variables for regression...")
X = combined_df[['10yr_Nominal_Zero_Coupon', 'cpih', 'UK_EPU_Index']]
y = combined_df['nav_discount_median']

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Print summary statistics of variables
print("\nSummary Statistics of Variables:")
print("\nDependent Variable (NAV Discount):")
print(y.describe())
print("\nIndependent Variables:")
print(X.describe())

# Run regression using statsmodels
print("\nRunning regression analysis...")
model = sm.OLS(y, X_sm)
results = model.fit()

# Create stargazer object
stargazer = Stargazer([results])

# Customize the table
stargazer.title('NAV Discount Drivers Regression Analysis')
stargazer.dependent_variable_name('NAV Discount Median')
stargazer.covariate_order(['const', '10yr_Nominal_Zero_Coupon', 'cpih', 'UK_EPU_Index'])
stargazer.rename_covariates({
    'const': 'Constant',
    '10yr_Nominal_Zero_Coupon': '10-Year Nominal Zero Coupon Rate',
    'cpih': 'CPIH',
    'UK_EPU_Index': 'UK Economic Policy Uncertainty Index'
})

# Save the table in different formats
os.makedirs(os.path.join('..', 'output', 'data'), exist_ok=True)
os.makedirs(os.path.join('..', 'output', 'charts'), exist_ok=True)

# Save as HTML
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with open(os.path.join('..', 'output', 'data', f'regression_table_{timestamp}.html'), 'w') as f:
    f.write(stargazer.render_html())

# Save as LaTeX
with open(os.path.join('..', 'output', 'data', f'regression_table_{timestamp}.tex'), 'w') as f:
    f.write(stargazer.render_latex())

# Print a message to check the output files
print("\nRegression Results Table saved as HTML and LaTeX. Please check the output files.")

# Create residual plots
plt.figure(figsize=(12, 8))

# Residuals vs Fitted
plt.subplot(2, 2, 1)
plt.scatter(results.fittedvalues, results.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# QQ Plot
plt.subplot(2, 2, 2)
sm.qqplot(results.resid, line='45')
plt.title('Q-Q Plot')

# Residuals Histogram
plt.subplot(2, 2, 3)
plt.hist(results.resid, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')

# Actual vs Predicted
plt.subplot(2, 2, 4)
plt.scatter(y, results.fittedvalues)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')

plt.tight_layout()
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
plt.savefig(os.path.join('..', 'output', 'charts', f'regression_diagnostics_{timestamp}.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nOutput files have been saved:")
print(f"- HTML table: output/data/regression_table.html")
print(f"- LaTeX table: output/data/regression_table.tex")
print(f"- Diagnostic plots: output/charts/regression_diagnostics.png")
