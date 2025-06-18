import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from viz_config import set_viz_style

# Set the visualization style from viz_config.py
set_viz_style()

# File paths
nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"
category_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\AllFunds_categorised.csv"
market_cap_path = r"..\..\..\..\20_Knowledge_Data\40_MarketData\Infra fund data\Uk_InfraFund_marketcap.xlsx"

def load_and_process_data():
    """Load and process NAV, category, and market cap data"""
    
    print("Loading NAV data...")
    # Load NAV data
    nav_df = pd.read_csv(nav_path)
    print(f"Loaded {len(nav_df)} NAV records")
    
    print("Loading category data...")
    # Load category data
    category_df = pd.read_csv(category_path)
    print(f"Loaded {len(category_df)} category records")
    
    # Merge NAV data with category data
    nav_df = pd.merge(nav_df, category_df[['Fund Name', 'Category']], on='Fund Name', how='left')
    print(f"After merge: {len(nav_df)} records")
    
    print("Loading market cap data...")
    # Process market cap data
    market_cap_df = pd.read_excel(market_cap_path, sheet_name="Funds Market Cap")
    print(f"Loaded {len(market_cap_df)} market cap records")
    
    # Rename market cap columns to match expected names
    market_cap_df.rename(columns={
        'requestId': 'Ticker',
        'Market Cap': 'Market Cap',
        'date': 'Date'
    }, inplace=True)
    
    # Convert Date columns to datetime
    nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%d/%m/%Y')
    market_cap_df['Date'] = pd.to_datetime(market_cap_df['Date'], format='%Y-%m-%d')
    
    # Sort market cap data by Date and Ticker
    market_cap_df = market_cap_df.sort_values(['Ticker', 'Date'])
    
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
    
    # Merge nav_df with market_cap_df on Date and Ticker
    nav_df_combined = pd.merge(nav_df, market_cap_df[['Date', 'Ticker', 'Market Cap']], 
                             on=['Date', 'Ticker'], how='left')
    print(f"After market cap merge: {len(nav_df_combined)} records")
    
    # Calculate the maximum allowed date difference (30 days)
    max_date_diff = pd.Timedelta(days=30)
    
    # Function to get closest market cap data
    def get_closest_market_cap(row):
        if pd.isna(row['Market Cap']):
            ticker_data = market_cap_df[market_cap_df['Ticker'] == row['Ticker']]
            if len(ticker_data) == 0:
                return np.nan
            
            date_diffs = abs(ticker_data['Date'] - row['Date'])
            min_diff_idx = date_diffs.idxmin()
            if date_diffs[min_diff_idx] <= max_date_diff:
                return ticker_data.loc[min_diff_idx, 'Market Cap']
        return row['Market Cap']
    
    # Apply the function to get closest market cap data
    nav_df_combined['Market Cap'] = nav_df_combined.apply(get_closest_market_cap, axis=1)
    
    # Add NAV Discount column
    nav_df_combined['Nav Discount'] = (nav_df_combined['Price'] / nav_df_combined['NAV']) - 1
    
    # Add NAV Discount Percentage column
    nav_df_combined['Nav Discount Percentage'] = nav_df_combined['Nav Discount'] * 100
    
    print(f"Final dataset: {len(nav_df_combined)} records")
    print(f"Date range: {nav_df_combined['Date'].min()} to {nav_df_combined['Date'].max()}")
    
    return nav_df_combined

def create_market_cap_weighted_infra_renewables_plot(nav_df_combined):
    print("Filtering for Infrastructure and Renewables...")
    df = nav_df_combined[nav_df_combined['Category'].isin(['Infrastructure', 'Renewables'])].copy()
    print(f"Filtered dataset: {len(df)} records")

    # Calculate market-cap weighted NAV discount for each date and category
    weighted_sum = (df['Nav Discount Percentage'] * df['Market Cap']).groupby([df['Date'], df['Category']]).sum()
    total_market_cap = df.groupby([df['Date'], df['Category']])['Market Cap'].sum()
    market_cap_weighted_discount = weighted_sum / total_market_cap
    market_cap_weighted_discount_df = market_cap_weighted_discount.unstack(level='Category').reset_index()

    # Create the plot using seaborn
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create scatter plots using seaborn
    sns.scatterplot(data=df[df['Category'] == 'Infrastructure'], 
                   x='Date', y='Nav Discount Percentage',
                   color='darkorange', alpha=0.4, s=20, 
                   label='Infrastructure (individual)', ax=ax)
    sns.scatterplot(data=df[df['Category'] == 'Renewables'], 
                   x='Date', y='Nav Discount Percentage',
                   color='forestgreen', alpha=0.4, s=20, 
                   label='Renewables (individual)', ax=ax)

    # Seaborn lineplot for market-cap weighted time series
    if 'Infrastructure' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Infrastructure',
            color='darkorange', linewidth=3, 
            label='Infrastructure (Market-Cap Weighted)', ax=ax
        )
    if 'Renewables' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Renewables',
            color='forestgreen', linewidth=3, 
            label='Renewables (Market-Cap Weighted)', ax=ax
        )

    # Add horizontal line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Set labels and title
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    # Set x-axis limits
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        ax.set_xlim(min_date, max_date)
    
    # Save the plot
    output_path = "Market_Cap_Weighted_NAV_Premium_Discount_Infra_vs_Renewables-final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {output_path}")
    return market_cap_weighted_discount_df

def main():
    """Main function to execute the analysis"""
    try:
        print("Loading and processing data...")
        nav_df_combined = load_and_process_data()
        
        print("Creating Market-Cap Weighted NAV Premium/Discount Time Series: Infrastructure vs Renewables plot...")
        create_market_cap_weighted_infra_renewables_plot(nav_df_combined)
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
