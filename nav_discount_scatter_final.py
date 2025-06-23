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

# List of funds to hide/omit
FUNDS_TO_EXCLUDE = [
    "Digital 9 Infrastructure plc ORD NPV",
    "HydrogenOne Capital Growth plc ORD GBP0.01",
    "Aquila Energy Efficiency Trust plc ORD GBP0.01",
    "BBGI Global Infrastructure S.A. Ord NPV (DI)"
]

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

    # Create the plot using seaborn (original style, no extra labels/legend/title)
    fig, ax = plt.subplots(figsize=(15, 8))
    infra_scatter = sns.scatterplot(data=df[df['Category'] == 'Infrastructure'], 
                   x='Date', y='Nav Discount Percentage',
                   color='darkorange', alpha=0.6, s=40, 
                   label='Infrastructure', ax=ax, zorder=3)
    renew_scatter = sns.scatterplot(data=df[df['Category'] == 'Renewables'], 
                   x='Date', y='Nav Discount Percentage',
                   color='forestgreen', alpha=0.6, s=40, 
                   label='Renewables', ax=ax, zorder=3)
    # Market-cap weighted lines (no legend)
    if 'Infrastructure' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Infrastructure',
            color='darkorange', linewidth=3, 
            legend=False, ax=ax, zorder=2
        )
    if 'Renewables' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Renewables',
            color='forestgreen', linewidth=3, 
            legend=False, ax=ax, zorder=2
        )
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x')
    plt.tight_layout()
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        ax.set_xlim(min_date, max_date)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    handles, labels = ax.get_legend_handles_labels()
    # Only keep the first two (scatter) handles
    ax.legend(handles[:2], labels[:2], loc='upper right', frameon=True, facecolor='white', framealpha=0.7)
    output_path = "Market_Cap_Weighted_NAV_Premium_Discount_Infra_vs_Renewables-final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {output_path}")
    return market_cap_weighted_discount_df

def create_market_cap_weighted_infra_renewables_plot_hide_scatter(nav_df_combined):
    print("[A] Hiding scatter for specified funds, but including them in calculations...")
    df = nav_df_combined[nav_df_combined['Category'].isin(['Infrastructure', 'Renewables'])].copy()
    scatter_df = df[~df['Fund Name'].isin(FUNDS_TO_EXCLUDE)]
    weighted_sum = (df['Nav Discount Percentage'] * df['Market Cap']).groupby([df['Date'], df['Category']]).sum()
    total_market_cap = df.groupby([df['Date'], df['Category']])['Market Cap'].sum()
    market_cap_weighted_discount = weighted_sum / total_market_cap
    market_cap_weighted_discount_df = market_cap_weighted_discount.unstack(level='Category').reset_index()
    # Use the original plotting code (no extra labels/legend/title)
    fig, ax = plt.subplots(figsize=(15, 8))
    infra_scatter = sns.scatterplot(data=scatter_df[scatter_df['Category'] == 'Infrastructure'], 
                   x='Date', y='Nav Discount Percentage',
                   color='darkorange', alpha=0.6, s=40, 
                   label='Infrastructure', ax=ax, zorder=3)
    renew_scatter = sns.scatterplot(data=scatter_df[scatter_df['Category'] == 'Renewables'], 
                   x='Date', y='Nav Discount Percentage',
                   color='forestgreen', alpha=0.6, s=40, 
                   label='Renewables', ax=ax, zorder=3)
    if 'Infrastructure' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Infrastructure',
            color='darkorange', linewidth=3, 
            legend=False, ax=ax, zorder=2
        )
    if 'Renewables' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Renewables',
            color='forestgreen', linewidth=3, 
            legend=False, ax=ax, zorder=2
        )
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x')
    plt.tight_layout()
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        ax.set_xlim(min_date, max_date)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right', frameon=True, facecolor='white', framealpha=0.7)
    output_path = "Market_Cap_Weighted_NAV_Premium_Discount_Infra_vs_Renewables-hide_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[A] Plot saved as: {output_path}")
    # Only report May 2025 median
    may_2025 = (df['Date'].dt.year == 2025) & (df['Date'].dt.month == 5)
    may_2025_df = df[may_2025]
    if not may_2025_df.empty:
        may_2025_medians = may_2025_df.groupby('Category')['Nav Discount Percentage'].median()
        print("[A] Median NAV Discount Percentage by Category in May 2025:")
        print(may_2025_medians)
        # Also report May 2025 median for all funds combined
        may_2025_overall_median = may_2025_df['Nav Discount Percentage'].median()
        print(f"[A] May 2025 Median NAV Discount Percentage (All Funds Combined): {may_2025_overall_median}")
    else:
        print("[A] No data for May 2025.")
    return None

def create_market_cap_weighted_infra_renewables_plot_omit_funds(nav_df_combined):
    print("[B] Omitting specified funds from all calculations and scatter...")
    df = nav_df_combined[
        nav_df_combined['Category'].isin(['Infrastructure', 'Renewables']) &
        (~nav_df_combined['Fund Name'].isin(FUNDS_TO_EXCLUDE))
    ].copy()
    weighted_sum = (df['Nav Discount Percentage'] * df['Market Cap']).groupby([df['Date'], df['Category']]).sum()
    total_market_cap = df.groupby([df['Date'], df['Category']])['Market Cap'].sum()
    market_cap_weighted_discount = weighted_sum / total_market_cap
    market_cap_weighted_discount_df = market_cap_weighted_discount.unstack(level='Category').reset_index()
    # Use the original plotting code (no extra labels/legend/title)
    fig, ax = plt.subplots(figsize=(15, 8))
    infra_scatter = sns.scatterplot(data=df[df['Category'] == 'Infrastructure'], 
                   x='Date', y='Nav Discount Percentage',
                   color='darkorange', alpha=0.6, s=40, 
                   label='Infrastructure', ax=ax, zorder=3)
    renew_scatter = sns.scatterplot(data=df[df['Category'] == 'Renewables'], 
                   x='Date', y='Nav Discount Percentage',
                   color='forestgreen', alpha=0.6, s=40, 
                   label='Renewables', ax=ax, zorder=3)
    if 'Infrastructure' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Infrastructure',
            color='darkorange', linewidth=3, 
            legend=False, ax=ax, zorder=2
        )
    if 'Renewables' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Renewables',
            color='forestgreen', linewidth=3, 
            legend=False, ax=ax, zorder=2
        )
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x')
    plt.tight_layout()
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        ax.set_xlim(min_date, max_date)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right', frameon=True, facecolor='white', framealpha=0.7)
    output_path = "Market_Cap_Weighted_NAV_Premium_Discount_Infra_vs_Renewables-omit_funds.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[B] Plot saved as: {output_path}")
    # Only report May 2025 median
    may_2025 = (df['Date'].dt.year == 2025) & (df['Date'].dt.month == 5)
    may_2025_df = df[may_2025]
    if not may_2025_df.empty:
        may_2025_medians = may_2025_df.groupby('Category')['Nav Discount Percentage'].median()
        print("[B] Median NAV Discount Percentage by Category in May 2025:")
        print(may_2025_medians)
        # Also report May 2025 median for all funds combined
        may_2025_overall_median = may_2025_df['Nav Discount Percentage'].median()
        print(f"[B] May 2025 Median NAV Discount Percentage (All Funds Combined): {may_2025_overall_median}")
    else:
        print("[B] No data for May 2025.")
    return None

def save_monthly_nav_discount_B(nav_df_combined):
    # Filter for B (omit specified funds)
    df = nav_df_combined[
        nav_df_combined['Category'].isin(['Infrastructure', 'Renewables']) &
        (~nav_df_combined['Fund Name'].isin(FUNDS_TO_EXCLUDE))
    ].copy()
    df['Month'] = df['Date'].dt.to_period('M')
    # All funds combined
    all_funds = df.groupby('Month')['Nav Discount Percentage'].median().rename('All Funds')
    # Infrastructure only
    infra = df[df['Category'] == 'Infrastructure'].groupby('Month')['Nav Discount Percentage'].median().rename('Infrastructure')
    # Renewables only
    renew = df[df['Category'] == 'Renewables'].groupby('Month')['Nav Discount Percentage'].median().rename('Renewables')
    # Combine into one DataFrame
    result = pd.concat([all_funds, infra, renew], axis=1)
    result.index = result.index.to_timestamp()
    result.to_csv('monthly_nav_discount_B.csv', index_label='Month')
    print('Saved monthly NAV discount table for version B as monthly_nav_discount_B.csv')

def main():
    """Main function to execute the analysis"""
    try:
        print("Loading and processing data...")
        nav_df_combined = load_and_process_data()
        
        print("Creating Market-Cap Weighted NAV Premium/Discount Time Series: Infrastructure vs Renewables plot...")
        create_market_cap_weighted_infra_renewables_plot(nav_df_combined)
        
        print("Analysis complete!")
        # New plots for A and B
        print("\n---\nGenerating plot A (hide scatter for specified funds)...")
        create_market_cap_weighted_infra_renewables_plot_hide_scatter(nav_df_combined)
        print("\n---\nGenerating plot B (omit specified funds from all calculations)...")
        create_market_cap_weighted_infra_renewables_plot_omit_funds(nav_df_combined)
        # Save monthly NAV discount table for version B
        save_monthly_nav_discount_B(nav_df_combined)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
