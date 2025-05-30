import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"

# Load NAV data
nav_df = pd.read_csv(nav_path)

# Convert Date column to datetime
nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%d/%m/%Y')

# Add NAV Discount column to the dataframe which is the ratio of NAV to Price -1
nav_df['Nav Discount'] = nav_df['NAV'] / nav_df['Price'] - 1

# Add NAV Discount Percentage column to the dataframe which is the NAV Discount times 100
nav_df['Nav Discount Percentage'] = nav_df['Nav Discount'] * 100

# Create monthly aggregation with quartiles and percentiles
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

# Create the time series plot
plt.figure(figsize=(15, 8))

# Plot the median line
plt.plot(monthly_stats['Date'], monthly_stats[('Nav Discount Percentage', 'median')], 
         label='Median NAV Discount', color='blue', linewidth=2)

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
plt.ylabel('NAV Discount Percentage')
plt.title('Monthly NAV Discount Percentage with Quartile and Percentile Bands')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the time series plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_discount_timeseries.png")

plt.show()

# Create a new figure for the scatter plot
plt.figure(figsize=(15, 8))

# Filter data for 2015-2025
nav_df_filtered = nav_df[(nav_df['Date'] >= '2015-01-01') & (nav_df['Date'] <= '2025-12-31')]

# Create scatter plot
plt.scatter(nav_df_filtered['Date'], nav_df_filtered['Nav Discount Percentage'], 
           alpha=0.5, s=10, label='Individual Fund NAV Discounts')

# Add median line to scatter plot
plt.plot(monthly_stats['Date'], monthly_stats[('Nav Discount Percentage', 'median')], 
         color='red', linewidth=2, label='Median NAV Discount')

plt.xlabel('Date')
plt.ylabel('NAV Discount Percentage')
plt.title('NAV Discounts Scatter Plot (2015-2025)')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the scatter plot
plt.savefig(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\nav_discount_scatter.png")

plt.show()

# save the dataframe to a csv file
nav_df.to_csv(r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS_NAV_DISCOUNT.csv", index=False)


