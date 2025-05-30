import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# File paths
nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"
rate_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\boe_yields_data.csv"
cpih_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\CPIH\cpih_data_monthly_1988-2025.xlsx"
uncertainty_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\Policy Uncertainty Data\UK_Policy_Uncertainty_Data.xlsx"

# Load data
nav_df = pd.read_csv(nav_path)
rate_df = pd.read_csv(rate_path)
cpih_df = pd.read_excel(cpih_path)
uncertainty_df = pd.read_excel(uncertainty_path)

# Print columns and head for debugging
print('nav_df columns:', nav_df.columns.tolist())
print(nav_df.head())
print('rate_df columns:', rate_df.columns.tolist())
print(rate_df.head())
print('uncertainty_df columns:', uncertainty_df.columns.tolist())
print(uncertainty_df.head())

# Print CPIH data structure
print("\nCPIH DataFrame columns:")
print(cpih_df.columns.tolist())
print("\nCPIH DataFrame head:")
print(cpih_df.head())

# Convert date columns to datetime with appropriate formats
nav_df['datetime'] = pd.to_datetime(nav_df['datetime'], format='%d/%m/%Y')
rate_df['datetime'] = pd.to_datetime(rate_df['datetime'], format='%d/%m/%Y')
cpih_df['datetime'] = pd.to_datetime(cpih_df['datetime'])
uncertainty_df['datetime'] = pd.to_datetime(uncertainty_df['datetime'])

# Filter data from May 1st, 2015 onwards
start_date = pd.to_datetime('2015-05-01')
nav_df = nav_df[nav_df['datetime'] >= start_date]
rate_df = rate_df[rate_df['datetime'] >= start_date]
cpih_df = cpih_df[cpih_df['datetime'] >= start_date]
uncertainty_df = uncertainty_df[uncertainty_df['datetime'] >= start_date]

# Calculate NAV discount statistics
nav_df['Nav Discount'] = (nav_df['Price'] / nav_df['NAV']) - 1
nav_df['Nav Discount Percentage'] = nav_df['Nav Discount'] * 100

# Create monthly date column for all dataframes
nav_df['Month'] = nav_df['datetime'].dt.to_period('M')
rate_df['Month'] = rate_df['datetime'].dt.to_period('M')
cpih_df['Month'] = cpih_df['datetime'].dt.to_period('M')
uncertainty_df['Month'] = uncertainty_df['datetime'].dt.to_period('M')

# Calculate monthly statistics for NAV discounts
monthly_nav_stats = nav_df.groupby('Month').agg({
    'Nav Discount Percentage': ['median', 'std']
}).reset_index()

# Flatten the multi-level columns
monthly_nav_stats.columns = ['Month', 'nav_discount_median', 'nav_discount_std']

# Calculate the bands
monthly_nav_stats['nav_discount_upper_2std'] = monthly_nav_stats['nav_discount_median'] + 2 * monthly_nav_stats['nav_discount_std']
monthly_nav_stats['nav_discount_lower_2std'] = monthly_nav_stats['nav_discount_median'] - 2 * monthly_nav_stats['nav_discount_std']
monthly_nav_stats['nav_discount_upper_1std'] = monthly_nav_stats['nav_discount_median'] + monthly_nav_stats['nav_discount_std']
monthly_nav_stats['nav_discount_lower_1std'] = monthly_nav_stats['nav_discount_median'] - monthly_nav_stats['nav_discount_std']

# Aggregate other datasets to monthly level
monthly_rate = rate_df.groupby('Month')['10yr_Nominal_Zero_Coupon'].mean().reset_index()
monthly_cpih = cpih_df.groupby('Month')['cpih'].mean().reset_index()
monthly_uncertainty = uncertainty_df.groupby('Month')['UK_EPU_Index'].mean().reset_index()

# Merge all datasets on Month
combined_df = monthly_nav_stats.merge(monthly_rate, on='Month', how='inner')
combined_df = combined_df.merge(monthly_cpih, on='Month', how='inner')
combined_df = combined_df.merge(monthly_uncertainty, on='Month', how='inner')

# Convert Month back to datetime for plotting
combined_df['Date'] = combined_df['Month'].dt.to_timestamp()

# Save the combined dataset for regression analysis
combined_df.to_csv(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_discount_drivers_combined_monthly.csv", index=False)

# Print and save correlation matrix
correlation_matrix = combined_df[['nav_discount_median', '10yr_Nominal_Zero_Coupon', 'cpih', 'UK_EPU_Index']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
correlation_matrix.to_csv(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_discount_correlations_monthly.csv")

# Ensure the output directory exists
output_dir = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV"
os.makedirs(output_dir, exist_ok=True)

# Individual scatter plots and save as PNGs
factors = [
    ('10yr_Nominal_Zero_Coupon', '10-Year Nominal Zero Coupon Rate'),
    ('cpih', 'CPIH'),
    ('UK_EPU_Index', 'UK Economic Policy Uncertainty Index')
]
for factor, label in factors:
    plt.figure(figsize=(8, 6))
    plt.scatter(combined_df[factor], combined_df['nav_discount_median'], alpha=0.5)
    corr = correlation_matrix.loc['nav_discount_median', factor]
    plt.xlabel(label)
    plt.ylabel('NAV Discount (%)')
    plt.title(f'NAV Discount vs {label}\nCorrelation: {corr:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"correlation_{factor}.png"))
    plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# Time series plot of NAV Discount (keep for context)
plt.figure(figsize=(12, 6))
plt.plot(combined_df['Date'], combined_df['nav_discount_median'], label='Median NAV Discount')
plt.fill_between(combined_df['Date'], 
                 combined_df['nav_discount_lower_1std'],
                 combined_df['nav_discount_upper_1std'],
                 alpha=0.3, label='±1 Std Dev')
plt.xlabel('Date')
plt.ylabel('NAV Discount (%)')
plt.title('Monthly NAV Discount Over Time')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "nav_discount_timeseries.png"))
plt.show()


