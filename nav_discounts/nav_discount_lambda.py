import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constants (using constant equity disc rate for now)
LEVERED_EQUITY_DISCOUNT_RATE = 0.095  # 9.5%

def get_risk_free_rate(date, rf_data):
    """
    Get the risk-free rate for a given date from BOE yields data.
    Using 10-year gilt yield as proxy for risk-free rate.
    """
    # Find the closest date in the data
    if date in rf_data.index:
        return rf_data.loc[date, '10yr_Nominal_Zero_Coupon']
    else:
        # If exact date not found, find the closest previous date
        available_dates = rf_data.index
        closest_date = available_dates[available_dates <= date].max()
        return rf_data.loc[closest_date, '10yr_Nominal_Zero_Coupon']

def get_rolling_risk_free_rate(date, rf_data, window_years=3):
    """
    Get the average risk-free rate over the specified window period
    """
    window_start = date - pd.DateOffset(years=window_years)
    period_data = rf_data[window_start:date]
    return period_data['10yr_Nominal_Zero_Coupon'].mean()

def calculate_implied_required_rate(nav_discounts, weights=None):
    """
    Calculate implied required rate using the formula:
    Implied required rate = (levered equity discount rate) / (1 + Nav discount)
    Note: NAV discounts are already in the correct format (positive for premium, negative for discount)
    """
    if weights is None:
        weights = np.ones(len(nav_discounts)) / len(nav_discounts)
    
    # Calculate weighted average NAV discount
    weighted_discount = np.average(nav_discounts, weights=weights)
    
    # Calculate implied required rate
    implied_rate = LEVERED_EQUITY_DISCOUNT_RATE / (1 + weighted_discount)
    
    return implied_rate

def calculate_lambda(implied_rate, rf_rate):
    """
    Calculate lambda (equity premium) by subtracting risk-free rate from implied required rate
    """
    return implied_rate - rf_rate

def create_rolling_analysis(monthly_discounts, start_date, end_date, window_years=3, rf_data=None):
    """Create time series of lambda values using rolling window with uniform weighting"""
    results = []
    current_date = start_date
    
    while current_date <= end_date:
        window_start = current_date - pd.DateOffset(years=window_years)
        period_data = monthly_discounts[window_start:current_date]
        
        if len(period_data) > 0:
            # Use uniform weights
            weights = np.ones(len(period_data)) / len(period_data)
            
            # Calculate implied required rate
            implied_rate = calculate_implied_required_rate(period_data.values, weights)
            
            # Get rolling average risk-free rate
            rf_rate = get_rolling_risk_free_rate(current_date, rf_data, window_years)
            
            # Calculate lambda
            lambda_value = calculate_lambda(implied_rate, rf_rate)
            
            results.append({
                'Date': current_date,
                'Weighted_NAV_Discount': np.average(period_data.values, weights=weights),
                'Implied_Required_Rate': implied_rate,
                'Risk_Free_Rate': rf_rate,
                'Lambda': lambda_value
            })
        
        current_date += pd.DateOffset(months=1)
    
    return pd.DataFrame(results)

def main():
    # Load the NAV data
    nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"
    nav_df = pd.read_csv(nav_path)
    
    # Load risk-free rate data
    rf_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\boe_yields_data.csv"
    rf_data = pd.read_csv(rf_path)
    rf_data['Date'] = pd.to_datetime(rf_data['Date'], format='%d/%m/%Y')
    rf_data.set_index('Date', inplace=True)
    
    # Convert Date column to datetime
    nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%d/%m/%Y')
    
    # Calculate NAV discounts
    nav_df['Nav Discount'] = (nav_df['Price'] / nav_df['NAV']) - 1
    
    # Get monthly median NAV discounts
    monthly_discounts = nav_df.groupby(nav_df['Date'].dt.to_period('M'))['Nav Discount'].median()
    monthly_discounts.index = monthly_discounts.index.to_timestamp()
    
    # --- Point-in-Time Analysis ---
    point_in_time_results = []
    
    for date in monthly_discounts.index:
        # Get NAV discount for the current month
        current_discount = monthly_discounts[date]
        
        # Calculate implied required rate
        implied_rate = calculate_implied_required_rate([current_discount])
        
        # Get risk-free rate for the date
        rf_rate = get_risk_free_rate(date, rf_data)
        
        # Calculate lambda
        lambda_value = calculate_lambda(implied_rate, rf_rate)
        
        point_in_time_results.append({
            'Date': date,
            'NAV_Discount': current_discount,
            'Implied_Required_Rate': implied_rate,
            'Risk_Free_Rate': rf_rate,
            'Lambda': lambda_value
        })
    
    # Create point-in-time results DataFrame
    point_in_time_df = pd.DataFrame(point_in_time_results)
    
    # Calculate relative lambda using May 2015 as base
    base_date = pd.Timestamp('2015-05-01')
    base_lambda = point_in_time_df[point_in_time_df['Date'] == base_date]['Lambda'].iloc[0]
    point_in_time_df['Relative_Lambda'] = point_in_time_df['Lambda'] / base_lambda
    
    # --- 3-Year Rolling Analysis ---
    end_date = pd.Timestamp('2025-05-01')
    start_date = pd.Timestamp('2018-05-01')  # 3 years before end date
    rolling_df = create_rolling_analysis(monthly_discounts, start_date, end_date, window_years=3, rf_data=rf_data)
    
    # --- Create Comprehensive Dataset ---
    # Merge point-in-time and rolling data
    comprehensive_df = pd.merge(
        point_in_time_df,
        rolling_df,
        on='Date',
        how='left',
        suffixes=('_Point', '_Rolling')
    )
    
    # Ensure output directories exist
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(os.path.join('..', 'output', 'data'), exist_ok=True)
    os.makedirs(os.path.join('..', 'output', 'charts'), exist_ok=True)
    # Save all results to CSV
    point_in_time_df.to_csv(os.path.join('..', 'output', 'data', f'lambda_point_in_time_results_{timestamp}.csv'), index=False)
    rolling_df.to_csv(os.path.join('..', 'output', 'data', f'lambda_rolling_3y_results_{timestamp}.csv'), index=False)
    comprehensive_df.to_csv(os.path.join('..', 'output', 'data', f'lambda_comprehensive_results_{timestamp}.csv'), index=False)
    
    # --- Visualizations ---
    # 1. Point-in-Time Analysis
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Lambda
    ax1.plot(point_in_time_df['Date'], point_in_time_df['Lambda'], color='blue', linewidth=2)
    ax1.set_title('Point-in-Time Lambda (Equity Premium)', pad=20, fontsize=14)
    ax1.set_ylabel('Lambda (Implied Required Rate - Risk-Free Rate)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    # Set y-axis range and format
    ax1.set_ylim(0, 0.15)  # Adjust range as needed
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    
    # Plot Relative Lambda
    ax2.plot(point_in_time_df['Date'], point_in_time_df['Relative_Lambda'], color='green', linewidth=2)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Base Level (May 2015)')
    ax2.set_title('Relative Lambda (vs May 2015)', pad=20, fontsize=14)
    ax2.set_ylabel('Relative Lambda', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    # Set y-axis range and format
    ax2.set_ylim(0.5, 1.5)  # Adjust range as needed
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'charts', f'lambda_point_in_time_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 3-Year Rolling Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_df['Date'], rolling_df['Lambda'], color='purple', linewidth=2)
    plt.title('3-Year Rolling Lambda (Equity Premium)\nUniform Weighting', pad=20, fontsize=14)
    plt.ylabel('Lambda (Implied Required Rate - Risk-Free Rate)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    # Set y-axis range and format
    plt.ylim(0.06, 0.09)  # Adjust range as needed
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'charts', f'lambda_rolling_3y_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Risk-Free Rate and Implied Required Rate Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(point_in_time_df['Date'], point_in_time_df['Risk_Free_Rate'], 
             color='blue', linewidth=2, label='Risk-Free Rate')
    plt.plot(point_in_time_df['Date'], point_in_time_df['Implied_Required_Rate'], 
             color='red', linewidth=2, label='Implied Required Rate')
    plt.title('Risk-Free Rate and Implied Required Rate (2015-2025)', pad=20, fontsize=14)
    plt.ylabel('Rate', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    # Set y-axis range and format
    plt.ylim(0, 0.15)  # Adjust range as needed
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'charts', f'rates_time_series_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print specific comparisons
    may_2024 = pd.Timestamp('2024-05-01')
    point_in_time_2024 = point_in_time_df[point_in_time_df['Date'] == may_2024]['Lambda'].iloc[0]
    rolling_2024 = rolling_df[rolling_df['Date'] == may_2024]['Lambda'].iloc[0]
    
    print("\nLambda Comparison (May 2024):")
    print(f"Point-in-Time Lambda: {point_in_time_2024:.4f}")
    print(f"3-Year Rolling Lambda: {rolling_2024:.4f}")
    print(f"\nRelative Change from May 2015 (Point-in-Time): {point_in_time_2024/base_lambda:.4f}")
    print("\nResults have been saved to:")
    print("- lambda_point_in_time_results.csv")
    print("- lambda_rolling_3y_results.csv")
    print("- lambda_comprehensive_results.csv")
    print("\nVisualizations have been saved to:")
    print("- lambda_point_in_time.png")
    print("- lambda_rolling_3y.png")
    print("- rates_time_series.png")

if __name__ == "__main__":
    main()
