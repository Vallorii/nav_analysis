import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"
category_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\AllFunds_categorised.csv"

# Load NAV data, including 'Fund Name'
# ASSUMING 'Fund Name' COLUMN EXISTS IN THE COMBINED CSV
nav_df = pd.read_csv(nav_path)

# Load category data
category_df = pd.read_csv(category_path)

# Convert Date column to datetime
nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%d/%m/%Y')

# Add NAV Discount column to the dataframe which is the ratio of Price to NAV -1 (Premium/Discount)
# Changed from NAV/Price - 1 to Price/NAV - 1 for Premium/Discount
nav_df['Nav Discount'] = ( nav_df['Price'] / nav_df['NAV'] ) - 1

# Add NAV Discount Percentage column
nav_df['Nav Discount Percentage'] = nav_df['Nav Discount'] * 100

# Merge NAV data with category data
# ASSUMING 'Fund Name' IS THE JOIN KEY IN BOTH DATAFRAMES
nav_df = pd.merge(nav_df, category_df[['Fund Name', 'Category']], on='Fund Name', how='left')

# Drop rows where Category is missing after merge, if any funds are not in the category file
nav_df.dropna(subset=['Category'], inplace=True)

# Create monthly aggregation with quartiles and percentiles (still needed for time series plot)
monthly_stats = nav_df.groupby(nav_df['Date'].dt.to_period('M')).agg({
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
plt.xlim(monthly_stats['Date'].min(), monthly_stats['Date'].max())

# Save the time series plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_timeseries.png")

plt.show()

# --- Scatter Plot by Category --- #
plt.figure(figsize=(15, 8))

# Filter data for 2015-2025
nav_df_filtered_scatter = nav_df[(nav_df['Date'] >= '2015-01-01') & (nav_df['Date'] <= '2025-12-31')].copy()

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
                color=color_map[category], alpha=0.4, s=15, label=category) # Changed alpha to 0.95 for very slight transparency

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
plt.xlim(nav_df_filtered_scatter['Date'].min(), nav_df_filtered_scatter['Date'].max())

# Save the scatter plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_premium_discount_scatter_category.png")

plt.show()

# --- Report Subgroup Statistics at Specific Dates --- #
print("\nSubgroup Statistics at Specific Dates:")

specific_dates = ['2017-01-01', '2021-01-01', '2025-01-01']

for date_str in specific_dates:
    print(f"\nDate: {date_str}")
    # Filter data for the specific date
    data_at_date = nav_df[nav_df['Date'] == date_str].copy()
    
    if not data_at_date.empty:
        # Group by category and calculate median and IQR
        subgroup_stats = data_at_date.groupby('Category')['Nav Discount Percentage'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).reset_index()
        subgroup_stats.columns = ['Category', 'Median NAV Premium/Discount', 'IQR']
        print(subgroup_stats.to_string(index=False))
    else:
        print("No data available for this date.")

# save the dataframe to a csv file (optional - keeping the original save)
# nav_df.to_csv(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS_NAV_DISCOUNT_CATEGORISED.csv", index=False)


