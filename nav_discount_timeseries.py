import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"
category_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\AllFunds_categorised.csv"
# Use a relative path for market cap data
market_cap_path = r"..\..\..\..\20_Knowledge_Data\40_MarketData\Infra fund data\Uk_InfraFund_marketcap.xlsx"

# Load NAV data, including 'Fund Name'
# ASSUMING 'Fund Name' COLUMN EXISTS IN THE COMBINED CSV
nav_df = pd.read_csv(nav_path)

# Load category data
category_df = pd.read_csv(category_path)

# Debug: Print unique fund names in nav_df before merge
print("\nUnique Fund Names in NAV data before category merge:")
print(nav_df['Fund Name'].unique())

# Debug: Print category data
print("\nCategory Data:")
print(category_df.head())

# Merge NAV data with category data first
# ASSUMING 'Fund Name' IS THE JOIN KEY IN BOTH DATAFRAMES
nav_df = pd.merge(nav_df, category_df[['Fund Name', 'Category']], on='Fund Name', how='left')

# Debug: Print number of rows with missing categories after merge
print(f"\nNumber of rows with missing category after merge: {nav_df['Category'].isna().sum()}")

# Process market cap data
market_cap_df = pd.read_excel(market_cap_path, sheet_name="Funds Market Cap")

# Rename market cap columns to match expected names
market_cap_df.rename(columns={
    'requestId': 'Ticker',
    'Market Cap': 'Market Cap', # Assuming this is the correct column name
    'date': 'Date'
}, inplace=True)

# Convert Date column to datetime in NAV data
nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%d/%m/%Y')

# Convert Date column to datetime in market cap data
market_cap_df['Date'] = pd.to_datetime(market_cap_df['Date'], format='%Y-%m-%d')

# Sort market cap data by Date and Ticker
market_cap_df = market_cap_df.sort_values(['Ticker', 'Date'])

# Calculate the maximum allowed date difference (30 days)
max_date_diff = pd.Timedelta(days=30)

# For each row in nav_df_combined, find the closest market cap data
def get_closest_market_cap(row):
    if pd.isna(row['Market Cap']):
        # Get all market cap data for this ticker
        ticker_data = market_cap_df[market_cap_df['Ticker'] == row['Ticker']]
        if len(ticker_data) == 0:
            return np.nan
        
        # Calculate date differences
        date_diffs = abs(ticker_data['Date'] - row['Date'])
        
        # Find the closest date within the maximum allowed difference
        min_diff_idx = date_diffs.idxmin()
        if date_diffs[min_diff_idx] <= max_date_diff:
            return ticker_data.loc[min_diff_idx, 'Market Cap']
    return row['Market Cap']

# Add fund ticker mapping
fund_tickers = {
    "3i Infrastructure plc Ord NPV": "3IN-GB",
    "Aquila Energy Efficiency Trust plc ORD GBP0.01": "AEET-GB",
    "BBGI Global Infrastructure S.A. Ord NPV (DI)": "BBGI-GB",
    "Bluefield Solar Income Fund Limited Ordinary Shares NPV": "BSIF-GB",
    "Cordiant Digital Infrastructure Limited ORD NPV": "CORD-GB",
    "Digital 9 Infrastructure plc ORD NPV": "DGI9-GB",
    "Downing Renewables & Infrastructure Trust plc ORD GBP0.01": "DORE-GB",
    "Ecofin Global Utilities & Infrastructure Trust plc Ordinary 1p": "EGL-GB",
    "Foresight Solar Fund Ltd Ordinary NPV": "FSFL-GB",
    "GCP Infrastructure Investments Ltd Ordinary 1p": "GCP-GB",
    "Gore Street Energy Storage Fund plc Ordinary Shares": "GSF-GB",
    "Greencoat UK Wind plc Ordinary 1p": "UKW-GB",
    "Gresham House Energy Storage Fund Plc Ord GBP0.01": "GRID-GB",
    "HICL Infrastructure plc ORD GBP0.0001": "HICL-GB",
    "HydrogenOne Capital Growth plc ORD GBP0.01": "HGEN-GB",
    "International Public Partnerships Limited Ord GBP0.01": "INPP-GB",
    "NextEnergy Solar Fund Ltd Ordinary NPV": "NESF-GB",
    "Octopus Renewables Infrastructure Trust plc ORD GBP0.01": "ORIT-GB",
    "SDCL Energy Income Trust Plc ORD GBP0.01": "SEIT-GB",
    "Sequoia Economic Infrastructure Income Fund Ltd NPV": "SEQI-GB"
}

# Add Ticker column to nav_df
nav_df['Ticker'] = nav_df['Fund Name'].map(fund_tickers)

# Merge nav_df (which now has Category) with market_cap_df on Date and Ticker
# Using a left merge to keep all rows from nav_df and add market cap where available
nav_df_combined = pd.merge(nav_df, market_cap_df[['Date', 'Ticker', 'Market Cap']], 
                         on=['Date', 'Ticker'], how='left')

# Apply the function to get closest market cap data
nav_df_combined['Market Cap'] = nav_df_combined.apply(get_closest_market_cap, axis=1)

# Add NAV Discount column to the dataframe which is the ratio of Price to NAV -1 (Premium/Discount)
# Changed from NAV/Price - 1 to Price/NAV - 1 for Premium/Discount
nav_df_combined['Nav Discount'] = ( nav_df_combined['Price'] / nav_df_combined['NAV'] ) - 1

# Add NAV Discount Percentage column
nav_df_combined['Nav Discount Percentage'] = nav_df_combined['Nav Discount'] * 100

# Display columns to check the merge (optional)
# print(nav_df_combined.head())

# Save the combined dataframe to a new csv file
output_combined_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\COMBINED_NAV_CATEGORY_MARKETCAP.csv"
nav_df_combined.to_csv(output_combined_path, index=False)

# Create monthly aggregation with quartiles and percentiles (still needed for time series plot)
# Use the combined dataframe for aggregation
monthly_stats = nav_df_combined.groupby(nav_df_combined['Date'].dt.to_period('M')).agg({
    'Nav Discount Percentage': [
        ('median', 'median'),
        ('q1', lambda x: x.quantile(0.25)),
        ('q3', lambda x: x.quantile(0.75)),
        ('p10', lambda x: x.quantile(0.10)),
        ('p90', lambda x: x.quantile(0.90))
    ]
}).reset_index()

# Convert period to datetime for plotting
monthly_stats['Date'] = monthly_stats['Date'].dt.to_timestamp()

# --- Time Series Plot --- #
plt.figure(figsize=(15, 8))

# Plot the median line
plt.plot(monthly_stats['Date'], monthly_stats[('Nav Discount Percentage', 'median')], 
         label='Median NAV Premium/Discount', color='blue', linewidth=2)

# Fill the 10th-90th percentile area
plt.fill_between(monthly_stats['Date'], 
                 monthly_stats[('Nav Discount Percentage', 'p10')], 
                 monthly_stats[('Nav Discount Percentage', 'p90')], 
                 alpha=0.2, color='gray', label='10th-90th Percentile')

# Fill the IQR area
plt.fill_between(monthly_stats['Date'], 
                 monthly_stats[('Nav Discount Percentage', 'q1')], 
                 monthly_stats[('Nav Discount Percentage', 'q3')], 
                 alpha=0.3, color='blue', label='Interquartile Range')

plt.xlabel('Date')
# Change y-axis label to 'NAV Premium / Discount'
plt.ylabel('NAV Premium / Discount Percentage')
plt.title('Monthly NAV Premium/Discount Percentage with Quartile and Percentile Bands')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Set x-axis limits to match data range (remove free space)
min_date = monthly_stats['Date'].min()
max_date = monthly_stats['Date'].max()
if pd.notna(min_date) and pd.notna(max_date):
    plt.xlim(min_date, max_date)

# Save the time series plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_timeseries.png")
plt.close()

# --- Scatter Plot by Category --- #
plt.figure(figsize=(15, 8))

# Filter data for 2015-2025
nav_df_filtered_scatter = nav_df_combined[(nav_df_combined['Date'] >= '2015-01-01') & (nav_df_combined['Date'] <= '2025-12-31')].copy()

# Get unique categories for coloring
categories = nav_df_filtered_scatter['Category'].unique()

# Define specific, highly distinguishable colors for key categories
category_colors = {
    'renewables': 'green',    # Assigned green
    'infrastructure': 'orange', # Assigned orange
}

# Use a colormap for other categories (if any)
default_colors = plt.cm.get_cmap('tab10', len(categories))
color_map = {}

# Assign colors, using specific colors for renewables and infrastructure
for i, category in enumerate(categories):
    if category in category_colors:
        color_map[category] = category_colors[category]
    else:
        # Use colormap for others, ensuring unique colors if possible
        color_map[category] = default_colors(i)

# Create scatter plot, color-coded by category
for category in categories:
    subset = nav_df_filtered_scatter[nav_df_filtered_scatter['Category'] == category]
    # Use the assigned color from the color_map
    plt.scatter(subset['Date'], subset['Nav Discount Percentage'], 
                color=color_map[category], alpha=0.4, s=15, label=category)

# Add median line to scatter plot (using the overall median)
plt.plot(monthly_stats['Date'], monthly_stats[('Nav Discount Percentage', 'median')], 
         color='red', linewidth=2, label='Overall Median')

plt.xlabel('Date')
plt.ylabel('NAV Premium / Discount Percentage')
plt.title('NAV Premium/Discount Scatter Plot by Category (2015-2025)')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Set x-axis limits to match data range (remove free space)
min_date = nav_df_filtered_scatter['Date'].min()
max_date = nav_df_filtered_scatter['Date'].max()
if pd.notna(min_date) and pd.notna(max_date):
    plt.xlim(min_date, max_date)

# Save the scatter plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_scatter_category.png")
plt.close()

# --- New Scatter Plot: Infrastructure vs Renewables --- #
print('Unique categories before filtering:', nav_df_filtered_scatter['Category'].unique())

plt.figure(figsize=(15, 8))

# Normalize Category to lowercase for consistent filtering
nav_df_filtered_scatter['Category_lower'] = nav_df_filtered_scatter['Category'].str.lower()
infra_renew_df = nav_df_filtered_scatter[nav_df_filtered_scatter['Category_lower'].isin(['infrastructure', 'renewables'])].copy()

# Plot scatter points for each category with distinct colors
plt.scatter(
    infra_renew_df[infra_renew_df['Category_lower'] == 'infrastructure']['Date'],
    infra_renew_df[infra_renew_df['Category_lower'] == 'infrastructure']['Nav Discount Percentage'],
    color='darkorange', alpha=0.4, s=20, label='Infrastructure')
plt.scatter(
    infra_renew_df[infra_renew_df['Category_lower'] == 'renewables']['Date'],
    infra_renew_df[infra_renew_df['Category_lower'] == 'renewables']['Nav Discount Percentage'],
    color='forestgreen', alpha=0.4, s=20, label='Renewables')

# Calculate monthly medians for each category
monthly_medians = infra_renew_df.groupby([infra_renew_df['Date'].dt.to_period('M'), 'Category_lower'])['Nav Discount Percentage'].median().reset_index()
monthly_medians['Date'] = monthly_medians['Date'].dt.to_timestamp()

# Plot median lines for each category with distinct colors
infra_median = monthly_medians[monthly_medians['Category_lower'] == 'infrastructure']
renew_median = monthly_medians[monthly_medians['Category_lower'] == 'renewables']
plt.plot(infra_median['Date'], infra_median['Nav Discount Percentage'], 
         color='red', linewidth=3, label='Infrastructure Median')
plt.plot(renew_median['Date'], renew_median['Nav Discount Percentage'], 
         color='blue', linewidth=3, label='Renewables Median')

plt.xlabel('Date')
plt.ylabel('NAV Premium / Discount Percentage')
plt.title('Infrastructure vs Renewables NAV Premium/Discount (2015-2025)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Set x-axis limits to match data range
min_date = infra_renew_df['Date'].min()
max_date = infra_renew_df['Date'].max()
if pd.notna(min_date) and pd.notna(max_date):
    plt.xlim(min_date, max_date)

# Save the new scatter plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_infra_vs_renew.png")
plt.close()

# --- Report Subgroup Statistics at Specific Dates --- #
print("\nSubgroup Statistics at Specific Dates (Non-Market-Cap Weighted):")

specific_dates = ['2017-01-01', '2021-01-01', '2025-01-01']

for date_str in specific_dates:
    print(f"\nDate: {date_str}")
    # Filter data for the specific date
    data_at_date = nav_df_combined[nav_df_combined['Date'] == date_str].copy()
    
    if not data_at_date.empty:
        # Group by category and calculate median and IQR
        subgroup_stats = data_at_date.groupby('Category')['Nav Discount Percentage'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).reset_index()
        subgroup_stats.columns = ['Category', 'Median NAV Premium/Discount', 'IQR']
        print(subgroup_stats.to_string(index=False))
    else:
        print("No data available for this date.")

# --- Report Market-Cap Weighted Subgroup Statistics at Specific Dates ---
print("\nMarket-Cap Weighted Subgroup Statistics at Specific Dates:")

# Ensure combined_df is loaded and Date is datetime (it should be from previous steps)
# combined_data_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\COMBINED_NAV_CATEGORY_MARKETCAP.csv"
# combined_df = pd.read_csv(combined_data_path)
# combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Filter for Infrastructure and Renewables
infra_renew_df_stats = nav_df_combined[nav_df_combined['Category'].isin(['Infrastructure', 'Renewables'])].copy()

market_cap_weighted_subgroup_results = []

for date_str in specific_dates:
    # Filter data for the specific date and categories
    data_at_date_cat = infra_renew_df_stats[infra_renew_df_stats['Date'] == date_str].copy()
    
    print(f"\nDate: {date_str}")
    
    if not data_at_date_cat.empty:
        # Calculate Market-Cap Weighted NAV Discount for each category at this date
        for category in ['Infrastructure', 'Renewables']:
            category_data = data_at_date_cat[data_at_date_cat['Category'] == category].copy()
            
            market_cap_weighted_avg_discount = np.nan
            
            if not category_data.empty:
                # Drop rows with missing Market Cap
                category_data_weighted = category_data.dropna(subset=['Market Cap']).copy()
                
                if not category_data_weighted.empty:
                    weighted_sum_discount = (category_data_weighted['Nav Discount'] * category_data_weighted['Market Cap']).sum()
                    total_market_cap_discount = category_data_weighted['Market Cap'].sum()
                    
                    if total_market_cap_discount != 0:
                        market_cap_weighted_avg_discount = weighted_sum_discount / total_market_cap_discount
                        
            market_cap_weighted_subgroup_results.append({
                'Date': date_str,
                'Category': category,
                'Market-Cap Weighted NAV Discount': market_cap_weighted_avg_discount
            })
            
        # Print results for this date
        date_results_df = pd.DataFrame([res for res in market_cap_weighted_subgroup_results if res['Date'] == date_str])
        if not date_results_df.empty:
             # Format for display
            date_results_df['Market-Cap Weighted NAV Discount'] = date_results_df['Market-Cap Weighted NAV Discount'].map('{:.2%}'.format)
            print(date_results_df.to_string(index=False))
        else:
            print("No weighted data available for this date and categories.")

    else:
        print("No data available for this date and categories.")

# --- Report Mean NAV Premium/Discount at 01/05/2025 ---
print("\nMean NAV Premium/Discount at 01/05/2025:")
data_at_20250501 = nav_df_combined[nav_df_combined['Date'] == '2025-05-01']

if not data_at_20250501.empty:
    mean_discount_20250501 = data_at_20250501['Nav Discount Percentage'].mean()
    print(f"{mean_discount_20250501:.4f}%")
else:
    print("No data available for 01/05/2025.")

# --- Report Median NAV Premium/Discount for Specific Dates ---
print("\nMedian NAV Premium/Discount for All Funds, Infrastructure, and Renewables:")

specific_dates = ['2021-01-01', '2025-01-01']

for date_str in specific_dates:
    print(f"\nDate: {date_str}")
    # Filter data for the specific date
    data_at_date = nav_df_combined[nav_df_combined['Date'] == date_str].copy()
    
    if not data_at_date.empty:
        # Median for all funds combined
        overall_median = data_at_date['Nav Discount Percentage'].median()
        print(f"All Funds: {overall_median:.2f}%")
        
        # Filter for Infrastructure and Renewables only
        data_at_date_cat = data_at_date[data_at_date['Category'].isin(['Infrastructure', 'Renewables'])]
        
        # Calculate median for each category
        median_stats = data_at_date_cat.groupby('Category')['Nav Discount Percentage'].median().reset_index()
        median_stats.columns = ['Category', 'Median NAV Premium/Discount']
        
        # Print category medians
        for _, row in median_stats.iterrows():
            print(f"{row['Category']}: {row['Median NAV Premium/Discount']:.2f}%")
        
        # Print individual fund values with tickers, ensuring no duplicates
        print("\nIndividual Fund Values:")
        # Filter for unique fund names and then iterate
        unique_funds_at_date = data_at_date_cat['Fund Name'].unique()
        for fund_name in unique_funds_at_date:
            fund_data = data_at_date_cat[data_at_date_cat['Fund Name'] == fund_name].iloc[0] # Get the first entry for the fund
            ticker = fund_tickers.get(fund_name, 'N/A')
            print(f"{fund_name} ({ticker}): {fund_data['Nav Discount Percentage']:.2f}%")
    else:
        print("No data available for this date.")

# --- Market-Cap Weighted NAV Discount Time Series Plot ---
print("\nCreating Market-Cap Weighted NAV Discount chart...")

# Load the combined data
combined_data_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\COMBINED_NAV_CATEGORY_MARKETCAP.csv"
combined_df = pd.read_csv(combined_data_path)

# Convert Date column to datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Calculate Market-Cap Weighted NAV Discount for each date
# Group by Date and calculate the sum of (Discount * Market Cap) and sum of Market Cap
weighted_sum = (combined_df['Nav Discount Percentage'] * combined_df['Market Cap']).groupby(combined_df['Date']).sum()
total_market_cap = combined_df.groupby(combined_df['Date'])['Market Cap'].sum()

# Calculate the weighted average (handle cases where total market cap is zero or NaN)
market_cap_weighted_discount = weighted_sum / total_market_cap

# Convert to DataFrame for plotting
market_cap_weighted_discount_df = market_cap_weighted_discount.reset_index()
market_cap_weighted_discount_df.columns = ['Date', 'Weighted NAV Discount Percentage']

# Create the plot
plt.figure(figsize=(15, 8))

plt.plot(market_cap_weighted_discount_df['Date'], market_cap_weighted_discount_df['Weighted NAV Discount Percentage'], 
         label='Market-Cap Weighted NAV Premium/Discount', color='purple', linewidth=2)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

plt.xlabel('Date')
plt.ylabel('Market-Cap Weighted NAV Premium/Discount (%)')
plt.title('Market-Cap Weighted Monthly NAV Premium/Discount Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Set x-axis limits to match data range (remove free space)
min_date = market_cap_weighted_discount_df['Date'].min()
max_date = market_cap_weighted_discount_df['Date'].max()
if pd.notna(min_date) and pd.notna(max_date):
    plt.xlim(min_date, max_date)

# Save the plot
output_weighted_plot_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_marketcap_weighted.png"
plt.savefig(output_weighted_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# --- Market-Cap Weighted NAV Discount Time Series Plot for Infrastructure and Renewables ---
print("\n--- Starting Implied Rate of Return Calculation ---") # Debug print
print("\nCreating Market-Cap Weighted NAV Discount chart for Infrastructure and Renewables...")

# Load the combined data
combined_data_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\COMBINED_NAV_CATEGORY_MARKETCAP.csv"
combined_df = pd.read_csv(combined_data_path)

# Convert Date column to datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Filter for only Infrastructure and Renewables
infra_renew_weighted_df = combined_df[combined_df['Category'].isin(['Infrastructure', 'Renewables'])].copy()

# Calculate Market-Cap Weighted NAV Discount for each date and category
# Group by Date and Category, then calculate the sum of (Discount * Market Cap) and sum of Market Cap
weighted_sum_cat = (infra_renew_weighted_df['Nav Discount Percentage'] * infra_renew_weighted_df['Market Cap']).groupby([infra_renew_weighted_df['Date'], infra_renew_weighted_df['Category']]).sum()
total_market_cap_cat = infra_renew_weighted_df.groupby([infra_renew_weighted_df['Date'], infra_renew_weighted_df['Category']])['Market Cap'].sum()

# Calculate the weighted average for each date and category (handle cases where total market cap is zero or NaN)
market_cap_weighted_discount_cat = weighted_sum_cat / total_market_cap_cat

# Convert to DataFrame for plotting and unstack the category level
market_cap_weighted_discount_df_cat = market_cap_weighted_discount_cat.unstack(level='Category').reset_index()
market_cap_weighted_discount_df_cat.columns.name = None # Remove the category index name

# Rename columns for clarity
market_cap_weighted_discount_df_cat.rename(columns={'Date': 'Date'}, inplace=True)

# Create the plot
plt.figure(figsize=(15, 8))

# Plot scatter points for each category with distinct colors
plt.scatter(
    infra_renew_weighted_df[infra_renew_weighted_df['Category'] == 'Infrastructure']['Date'],
    infra_renew_weighted_df[infra_renew_weighted_df['Category'] == 'Infrastructure']['Nav Discount Percentage'],
    color='darkorange', alpha=0.4, s=20, label='Infrastructure')
plt.scatter(
    infra_renew_weighted_df[infra_renew_weighted_df['Category'] == 'Renewables']['Date'],
    infra_renew_weighted_df[infra_renew_weighted_df['Category'] == 'Renewables']['Nav Discount Percentage'],
    color='forestgreen', alpha=0.4, s=20, label='Renewables')

# Plot weighted discount for Infrastructure
if 'Infrastructure' in market_cap_weighted_discount_df_cat.columns:
    plt.plot(market_cap_weighted_discount_df_cat['Date'], market_cap_weighted_discount_df_cat['Infrastructure'], 
             label='Infrastructure (Market-Cap Weighted)', color='red', linewidth=3)

# Plot weighted discount for Renewables
if 'Renewables' in market_cap_weighted_discount_df_cat.columns:
    plt.plot(market_cap_weighted_discount_df_cat['Date'], market_cap_weighted_discount_df_cat['Renewables'], 
             label='Renewables (Market-Cap Weighted)', color='blue', linewidth=3)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

plt.xlabel('Date')
plt.ylabel('Market-Cap Weighted NAV Premium/Discount (%)')
plt.title('Market-Cap Weighted Monthly NAV Premium/Discount Time Series: Infrastructure vs Renewables')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Set x-axis limits to match data range
min_date = infra_renew_weighted_df['Date'].min()
max_date = infra_renew_weighted_df['Date'].max()
if pd.notna(min_date) and pd.notna(max_date):
    plt.xlim(min_date, max_date)

# Save the plot
output_weighted_infra_renew_plot_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_marketcap_weighted_infra_renew.png"
plt.savefig(output_weighted_infra_renew_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# --- Calculate Implied Rate of Return at Specific Dates ---
print("\nCalculating Implied Rate of Return for all funds combined...")

# Load the combined data if not already loaded
# combined_data_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\COMBINED_NAV_CATEGORY_MARKETCAP.csv"
# combined_df = pd.read_csv(combined_data_path)

# Convert Date column to datetime if not already done
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Define the average discount rates extracted from the CSV
discount_rate_averages = {
    2015: 0.0902,
    2016: 0.0887,
    2017: 0.0881,
    2018: 0.0877,
    2019: 0.0864,
    2020: 0.0824,
    2021: 0.0815,
    2022: 0.0864,
    2023: 0.0965,
    2024: 0.0973,
    2025: 0.0973,
}

results = []

# Iterate through years 2015 to 2025
for year in range(2015, 2026):
    # Filter data for April of the current year
    april_data = combined_df[(combined_df['Date'].dt.year == year) & (combined_df['Date'].dt.month == 4)].copy()
    
    # Debug: Print number of rows in april_data
    print(f"\nYear {year}: Number of rows in April data: {len(april_data)}")
    
    # Use all funds data for this year's April
    all_funds_april_data = april_data.copy()
    
    market_cap_weighted_avg_discount = np.nan # Initialize as NaN
    
    if not all_funds_april_data.empty:
        # Drop rows with missing Market Cap as they cannot be weighted
        all_funds_april_data_weighted = all_funds_april_data.dropna(subset=['Market Cap']).copy()
        
        # Debug: Print number of rows after dropping missing market cap
        print(f"Year {year}: Number of rows after dropping missing Market Cap: {len(all_funds_april_data_weighted)}")
        
        # Debug: Print head of weighted data if not empty
        if not all_funds_april_data_weighted.empty:
            print(f"Year {year}: Head of weighted data:")
            print(all_funds_april_data_weighted.head())
            
        if not all_funds_april_data_weighted.empty:
            # Calculate weighted average NAV Discount = SUM(Discount * Market Cap) / SUM(Market Cap)
            weighted_sum_discount = (all_funds_april_data_weighted['Nav Discount'] * all_funds_april_data_weighted['Market Cap']).sum()
            total_market_cap_discount = all_funds_april_data_weighted['Market Cap'].sum()
            
            if total_market_cap_discount != 0:
                 market_cap_weighted_avg_discount = weighted_sum_discount / total_market_cap_discount

    # Get the corresponding discount rate average for the year
    discount_rate_avg = discount_rate_averages.get(year, np.nan) # Get rate, default to NaN if year not found
    
    implied_required_rate = np.nan # Initialize as NaN
    
    # Calculate implied required rate if both discount rate and NAV discount are available
    if not pd.isna(discount_rate_avg) and not pd.isna(market_cap_weighted_avg_discount) and (1 + market_cap_weighted_avg_discount) != 0:
        implied_required_rate = discount_rate_avg / (1 + market_cap_weighted_avg_discount)
        
    # Append results
    results.append({
        'Year': year,
        'Market-Cap Weighted NAV Discount (April)': market_cap_weighted_avg_discount,
        'Average Discount Rate': discount_rate_avg,
        'Implied Required Rate': implied_required_rate
    })

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Format for display
results_df['Market-Cap Weighted NAV Discount (April)'] = results_df['Market-Cap Weighted NAV Discount (April)'].map('{:.2%}'.format)
results_df['Average Discount Rate'] = results_df['Average Discount Rate'].map('{:.2%}'.format)
results_df['Implied Required Rate'] = results_df['Implied Required Rate'].map('{:.2%}'.format)

# Print the results table
print("\nImplied Rate of Return Analysis (April Data, All Funds Combined):")
print(results_df.to_string(index=False))

