import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def set_viz_style():
    """Set the visualization style for consistent plotting"""
    sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)
    plt.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'axes.grid': True,  # Enable gridlines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'figure.titlesize': 15,
        'savefig.transparent': True,
        'axes.edgecolor': 'white',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'font.weight': 'bold',  # Make all fonts bold
    })
    # Set Offshore Wind color globally for consistency
    sns.set_palette(sns.color_palette(["#1E88E5"]))  # Blue for Offshore Wind

def load_processed_data():
    """Load the processed data from CSV"""
    data_path = "processed_nav_data.csv"
    try:
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded {len(df)} records from {data_path}")
        return df
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run the data processing script first.")
        return None

def create_market_cap_weighted_infra_renewables_plot(df):
    """Create the Market-Cap Weighted Monthly NAV Premium/Discount Time Series: Infrastructure vs Renewables plot"""
    print("Filtering for Infrastructure and Renewables...")
    filtered_df = df[df['Category'].isin(['Infrastructure', 'Renewables'])].copy()
    print(f"Filtered dataset: {len(filtered_df)} records")

    # Calculate market-cap weighted NAV discount for each date and category
    weighted_sum = (filtered_df['Nav Discount Percentage'] * filtered_df['Market Cap']).groupby([filtered_df['Date'], filtered_df['Category']]).sum()
    total_market_cap = filtered_df.groupby([filtered_df['Date'], filtered_df['Category']])['Market Cap'].sum()
    market_cap_weighted_discount = weighted_sum / total_market_cap
    market_cap_weighted_discount_df = market_cap_weighted_discount.unstack(level='Category').reset_index()

    # Create the plot using seaborn
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create scatter plots using seaborn with labels for legend (transparent for plot, full color for legend)
    infra_scatter = sns.scatterplot(data=filtered_df[filtered_df['Category'] == 'Infrastructure'], 
                                   x='Date', y='Nav Discount Percentage',
                                   color='darkorange', alpha=0.4, s=40, 
                                   label='Infrastructure', ax=ax)
    renew_scatter = sns.scatterplot(data=filtered_df[filtered_df['Category'] == 'Renewables'], 
                                   x='Date', y='Nav Discount Percentage',
                                   color='forestgreen', alpha=0.4, s=40, 
                                   label='Renewables', ax=ax)

    # Seaborn lineplot for market-cap weighted time series (without labels to exclude from legend)
    if 'Infrastructure' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Infrastructure',
            color='darkorange', linewidth=3, 
            label='', ax=ax  # Empty label to exclude from legend
        )
    if 'Renewables' in market_cap_weighted_discount_df.columns:
        sns.lineplot(
            data=market_cap_weighted_discount_df,
            x='Date', y='Renewables',
            color='forestgreen', linewidth=3, 
            label='', ax=ax  # Empty label to exclude from legend
        )

    # Add horizontal line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Create custom legend with full color markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', 
               markersize=8, label='Infrastructure'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', 
               markersize=8, label='Renewables')
    ]
    
    # Use custom legend with full color markers
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Set x-axis limits
    min_date = filtered_df['Date'].min()
    max_date = filtered_df['Date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        ax.set_xlim(min_date, max_date)
    
    # Save the plot
    output_path = "Market_Cap_Weighted_NAV_Premium_Discount_Infra_vs_Renewables-final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {output_path}")
    return market_cap_weighted_discount_df

def main():
    """Main function to create the visualization"""
    # Set the visualization style
    set_viz_style()
    
    # Load the processed data
    df = load_processed_data()
    if df is None:
        return
    
    # Create the plot
    print("Creating Market-Cap Weighted NAV Premium/Discount Time Series: Infrastructure vs Renewables plot...")
    create_market_cap_weighted_infra_renewables_plot(df)
    print("Visualization complete!")

if __name__ == "__main__":
    main() 