import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- File paths (update if needed) ---
nav_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\20250529_235935_COMBINED_ALL_FUNDS.csv"
category_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\AllFunds_categorised.csv"
discount_rate_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\NAV\NAV data excluding extremes\250623_discount-rates_nav-discounts.xlsx"
rf_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\boe_yields_data.csv"

# --- Funds to exclude for version B ---
FUNDS_TO_EXCLUDE = [
    "Digital 9 Infrastructure plc ORD NPV",
    "HydrogenOne Capital Growth plc ORD GBP0.01",
    "Aquila Energy Efficiency Trust plc ORD GBP0.01",
    "BBGI Global Infrastructure S.A. Ord NPV (DI)"
]

# --- Load NAV and category data ---
nav_df = pd.read_csv(nav_path)
category_df = pd.read_csv(category_path)
nav_df = pd.merge(nav_df, category_df[['Fund Name', 'Category']], on='Fund Name', how='left')
nav_df['Date'] = pd.to_datetime(nav_df['Date'], format='%d/%m/%Y')

# --- Filter for version B (exclude specified funds, only infra/renewables) ---
nav_df = nav_df[
    nav_df['Category'].isin(['Infrastructure', 'Renewables']) &
    (~nav_df['Fund Name'].isin(FUNDS_TO_EXCLUDE))
].copy()
# Save filtered nav_df as CSV
nav_df.to_csv('nav_df_B.csv', index=False)
print('Saved filtered nav_df as nav_df_B.csv')

# --- Calculate NAV Discount Percentage ---
nav_df['Nav Discount'] = (nav_df['Price'] / nav_df['NAV']) - 1
nav_df['Nav Discount Percentage'] = nav_df['Nav Discount'] * 100
nav_df['Month'] = nav_df['Date'].dt.to_period('M')

# --- Monthly median NAV discount for all, infra, renewables ---
monthly = nav_df.groupby('Month').agg(
    All_Funds_Median_Nav_Discount=('Nav Discount', 'median'),
).reset_index()
monthly_infra = nav_df[nav_df['Category'] == 'Infrastructure'].groupby('Month').agg(
    Infra_Median_Nav_Discount=('Nav Discount', 'median'),
).reset_index()
monthly_renew = nav_df[nav_df['Category'] == 'Renewables'].groupby('Month').agg(
    Renew_Median_Nav_Discount=('Nav Discount', 'median'),
).reset_index()

# --- Merge all medians into one DataFrame ---
monthly = monthly.merge(monthly_infra, on='Month', how='left')
monthly = monthly.merge(monthly_renew, on='Month', how='left')
monthly['Date'] = monthly['Month'].dt.to_timestamp()

# --- Load discount rate and risk free rate data ---
discount_df = pd.read_excel(discount_rate_path)
discount_df['Date'] = pd.to_datetime(discount_df['Date'])
riskfree_df = pd.read_csv(rf_path)
riskfree_df['Date'] = pd.to_datetime(riskfree_df['Date'], format='%d/%m/%Y')

# --- For each month, get the closest discount rate and risk free rate ---
def get_closest_value(date, df, col):
    idx = (df['Date'] - date).abs().idxmin()
    return df.loc[idx, col]

monthly['discount_rate_average'] = monthly['Date'].apply(lambda d: get_closest_value(d, discount_df, 'discount_rate_average'))
monthly['risk_free_rate'] = monthly['Date'].apply(lambda d: get_closest_value(d, riskfree_df, '10yr_Nominal_Zero_Coupon'))

# --- Calculate implied required rate and implied equity risk premium for each group ---
def implied_required_rate(discount_rate, nav_discount):
    return discount_rate / (1 + nav_discount)

def implied_equity_risk_premium(implied_rate, risk_free):
    return implied_rate - risk_free

for group, col in [
    ('All_Funds', 'All_Funds_Median_Nav_Discount'),
    ('Infra', 'Infra_Median_Nav_Discount'),
    ('Renew', 'Renew_Median_Nav_Discount')
]:
    monthly[f'{group}_Implied_Required_Rate'] = implied_required_rate(monthly['discount_rate_average'], monthly[col])
    monthly[f'{group}_Implied_Equity_Risk_Premium'] = implied_equity_risk_premium(monthly[f'{group}_Implied_Required_Rate'], monthly['risk_free_rate'])

# --- Restrict to 2015-05-01 to 2025-05-01 ---
mask = (monthly['Date'] >= '2015-05-01') & (monthly['Date'] <= '2025-05-01')
monthly = monthly.loc[mask].copy()

# --- Save intermediate monthly DataFrame to CSV for further analysis ---
monthly.to_csv('monthly_intermediate.csv', index=False)
print('Saved intermediate monthly DataFrame as monthly_intermediate.csv')

# --- Save to CSV ---
monthly.to_csv('implied_equity_risk_premium_B.csv', index=False)
print('Saved implied equity risk premium table as implied_equity_risk_premium_B.csv')

# --- Plot line plot for all funds implied equity risk premium ---
plt.figure(figsize=(12,6))
plt.plot(monthly['Date'], monthly['All_Funds_Implied_Equity_Risk_Premium'], color='blue', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Implied Equity Risk Premium (All Funds, Median)')
plt.title('Implied Equity Risk Premium (All Funds, Median) Over Time')
# Set x-axis to fit exactly from 2015-05-01 to 2025-05-01 with yearly ticks, no duplicate 2025
start_date = pd.Timestamp('2015-05-01')
end_date = pd.Timestamp('2025-05-01')
plt.xlim(start_date, end_date)
ax = plt.gca()
years = pd.date_range(start=start_date, end=end_date, freq='YS')
ax.set_xticks(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Set y-axis to start at 0.00
ax.set_ylim(bottom=0.00)
plt.tight_layout()
plt.savefig('implied_equity_risk_premium_B.png', dpi=300)
plt.close()
print('Saved implied equity risk premium plot as implied_equity_risk_premium_B.png')

# --- Plot relative change since 01/05/2015 ---
base_value = monthly.loc[monthly['Date'] == pd.Timestamp('2015-05-01'), 'All_Funds_Implied_Equity_Risk_Premium'].values
if len(base_value) > 0 and base_value[0] != 0:
    rel = monthly['All_Funds_Implied_Equity_Risk_Premium'] / base_value[0]
    plt.figure(figsize=(12,6))
    plt.plot(monthly['Date'], rel, color='purple', linewidth=2)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Relative Change (vs 01/05/2015)')
    plt.title('Relative Change in Implied Equity Risk Premium (All Funds, Median)')
    plt.xlim(start_date, end_date)
    ax = plt.gca()
    ax.set_xticks(years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('implied_equity_risk_premium_relative_B.png', dpi=300)
    plt.close()
    print('Saved relative implied equity risk premium plot as implied_equity_risk_premium_relative_B.png')

# --- Load market cap data ---
market_cap_path = r"C:\Users\CB - Vallorii\Vallorii\Vallorii - Vallorii Team\20_Knowledge_Data\40_MarketData\Infra fund data\Uk_InfraFund_marketcap.xlsx"
market_cap_df = pd.read_excel(market_cap_path, sheet_name="Funds Market Cap")
market_cap_df.rename(columns={
    'requestId': 'Ticker',
    'Market Cap': 'Market Cap',
    'date': 'Date'
}, inplace=True)
market_cap_df['Date'] = pd.to_datetime(market_cap_df['Date'], format='%Y-%m-%d')

# Add fund ticker mapping (same as in nav_discount_scatter_final.py)
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
nav_df['Ticker'] = nav_df['Fund Name'].map(fund_tickers)

# Merge nav_df with market_cap_df on Date and Ticker
nav_df = pd.merge(nav_df, market_cap_df[['Date', 'Ticker', 'Market Cap']], on=['Date', 'Ticker'], how='left')

# For missing market cap, get closest within 30 days
max_date_diff = pd.Timedelta(days=30)
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
nav_df['Market Cap'] = nav_df.apply(get_closest_market_cap, axis=1)

# Save nav_df with market cap as CSV
nav_df.to_csv('nav_df_with_marketcap.csv', index=False)
print('Saved nav_df with market cap as nav_df_with_marketcap.csv')

# --- Market cap weighted NAV discount for each month and group ---
def weighted_nav_discount(subdf):
    valid = subdf.dropna(subset=['Market Cap', 'Nav Discount'])
    if valid['Market Cap'].sum() == 0:
        return np.nan
    return (valid['Nav Discount'] * valid['Market Cap']).sum() / valid['Market Cap'].sum()

weighted = nav_df.groupby('Month').apply(weighted_nav_discount).rename('All_Funds_Weighted_Nav_Discount')
weighted_infra = nav_df[nav_df['Category'] == 'Infrastructure'].groupby('Month').apply(weighted_nav_discount).rename('Infra_Weighted_Nav_Discount')
weighted_renew = nav_df[nav_df['Category'] == 'Renewables'].groupby('Month').apply(weighted_nav_discount).rename('Renew_Weighted_Nav_Discount')

# Merge into monthly
monthly = monthly.set_index('Month')
monthly = monthly.join(weighted)
monthly = monthly.join(weighted_infra)
monthly = monthly.join(weighted_renew)
monthly = monthly.reset_index()

# --- Calculate implied required rate and equity risk premium for weighted ---
for group, col in [
    ('All_Funds_Weighted', 'All_Funds_Weighted_Nav_Discount'),
    ('Infra_Weighted', 'Infra_Weighted_Nav_Discount'),
    ('Renew_Weighted', 'Renew_Weighted_Nav_Discount')
]:
    monthly[f'{group}_Implied_Required_Rate'] = implied_required_rate(monthly['discount_rate_average'], monthly[col])
    monthly[f'{group}_Implied_Equity_Risk_Premium'] = implied_equity_risk_premium(monthly[f'{group}_Implied_Required_Rate'], monthly['risk_free_rate'])

# --- Plot market cap weighted implied equity risk premium (level) ---
plt.figure(figsize=(12,6))
plt.plot(monthly['Date'], monthly['All_Funds_Weighted_Implied_Equity_Risk_Premium'], color='red', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Implied Equity Risk Premium (All Funds, Weighted)')
plt.title('Implied Equity Risk Premium (All Funds, Market Cap Weighted) Over Time')
plt.xlim(start_date, end_date)
ax = plt.gca()
ax.set_xticks(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(bottom=0.00)
plt.tight_layout()
plt.savefig('implied_equity_risk_premium_weighted_B.png', dpi=300)
plt.close()
print('Saved implied equity risk premium plot (market cap weighted) as implied_equity_risk_premium_weighted_B.png')

# --- Plot relative change for weighted ---
base_value_w = monthly.loc[monthly['Date'] == pd.Timestamp('2015-05-01'), 'All_Funds_Weighted_Implied_Equity_Risk_Premium'].values
if len(base_value_w) > 0 and base_value_w[0] != 0:
    rel_w = monthly['All_Funds_Weighted_Implied_Equity_Risk_Premium'] / base_value_w[0]
    plt.figure(figsize=(12,6))
    plt.plot(monthly['Date'], rel_w, color='green', linewidth=2)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Relative Change (vs 01/05/2015)')
    plt.title('Relative Change in Implied Equity Risk Premium (All Funds, Weighted)')
    plt.xlim(start_date, end_date)
    ax = plt.gca()
    ax.set_xticks(years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('implied_equity_risk_premium_weighted_relative_B.png', dpi=300)
    plt.close()
    print('Saved relative implied equity risk premium plot (market cap weighted) as implied_equity_risk_premium_weighted_relative_B.png')

# --- Plot by category (normal, level) ---
plt.figure(figsize=(12,6))
plt.plot(monthly['Date'], monthly['Infra_Implied_Equity_Risk_Premium'], color='darkorange', linewidth=2, label='Infrastructure')
plt.plot(monthly['Date'], monthly['Renew_Implied_Equity_Risk_Premium'], color='forestgreen', linewidth=2, label='Renewables')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Implied Equity Risk Premium')
plt.title('Implied Equity Risk Premium by Category (Median) Over Time')
plt.xlim(start_date, end_date)
ax = plt.gca()
ax.set_xticks(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(bottom=0.00)
plt.legend()
plt.tight_layout()
plt.savefig('implied_equity_risk_premium_by_category_B.png', dpi=300)
plt.close()
print('Saved implied equity risk premium by category plot as implied_equity_risk_premium_by_category_B.png')

# --- Plot by category (normal, relative) ---
base_infra = monthly.loc[monthly['Date'] == pd.Timestamp('2015-05-01'), 'Infra_Implied_Equity_Risk_Premium'].values
base_renew = monthly.loc[monthly['Date'] == pd.Timestamp('2015-05-01'), 'Renew_Implied_Equity_Risk_Premium'].values
if len(base_infra) > 0 and base_infra[0] != 0 and len(base_renew) > 0 and base_renew[0] != 0:
    rel_infra = monthly['Infra_Implied_Equity_Risk_Premium'] / base_infra[0]
    rel_renew = monthly['Renew_Implied_Equity_Risk_Premium'] / base_renew[0]
    plt.figure(figsize=(12,6))
    plt.plot(monthly['Date'], rel_infra, color='darkorange', linewidth=2, label='Infrastructure')
    plt.plot(monthly['Date'], rel_renew, color='forestgreen', linewidth=2, label='Renewables')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Relative Change (vs 01/05/2015)')
    plt.title('Relative Change in Implied Equity Risk Premium by Category (Median)')
    plt.xlim(start_date, end_date)
    ax = plt.gca()
    ax.set_xticks(years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend()
    plt.tight_layout()
    plt.savefig('implied_equity_risk_premium_relative_by_category_B.png', dpi=300)
    plt.close()
    print('Saved relative implied equity risk premium by category plot as implied_equity_risk_premium_relative_by_category_B.png')

# --- Plot by category (weighted, level) ---
plt.figure(figsize=(12,6))
plt.plot(monthly['Date'], monthly['Infra_Weighted_Implied_Equity_Risk_Premium'], color='darkorange', linewidth=2, label='Infrastructure (Weighted)')
plt.plot(monthly['Date'], monthly['Renew_Weighted_Implied_Equity_Risk_Premium'], color='forestgreen', linewidth=2, label='Renewables (Weighted)')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Implied Equity Risk Premium (Weighted)')
plt.title('Implied Equity Risk Premium by Category (Market Cap Weighted) Over Time')
plt.xlim(start_date, end_date)
ax = plt.gca()
ax.set_xticks(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(bottom=0.00)
plt.legend()
plt.tight_layout()
plt.savefig('implied_equity_risk_premium_weighted_by_category_B.png', dpi=300)
plt.close()
print('Saved implied equity risk premium by category (weighted) plot as implied_equity_risk_premium_weighted_by_category_B.png')

# --- Plot by category (weighted, relative) ---
base_infra_w = monthly.loc[monthly['Date'] == pd.Timestamp('2015-05-01'), 'Infra_Weighted_Implied_Equity_Risk_Premium'].values
base_renew_w = monthly.loc[monthly['Date'] == pd.Timestamp('2015-05-01'), 'Renew_Weighted_Implied_Equity_Risk_Premium'].values
if len(base_infra_w) > 0 and base_infra_w[0] != 0 and len(base_renew_w) > 0 and base_renew_w[0] != 0:
    rel_infra_w = monthly['Infra_Weighted_Implied_Equity_Risk_Premium'] / base_infra_w[0]
    rel_renew_w = monthly['Renew_Weighted_Implied_Equity_Risk_Premium'] / base_renew_w[0]
    plt.figure(figsize=(12,6))
    plt.plot(monthly['Date'], rel_infra_w, color='darkorange', linewidth=2, label='Infrastructure (Weighted)')
    plt.plot(monthly['Date'], rel_renew_w, color='forestgreen', linewidth=2, label='Renewables (Weighted)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Relative Change (vs 01/05/2015)')
    plt.title('Relative Change in Implied Equity Risk Premium by Category (Weighted)')
    plt.xlim(start_date, end_date)
    ax = plt.gca()
    ax.set_xticks(years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend()
    plt.tight_layout()
    plt.savefig('implied_equity_risk_premium_weighted_relative_by_category_B.png', dpi=300)
    plt.close()
    print('Saved relative implied equity risk premium by category (weighted) plot as implied_equity_risk_premium_weighted_relative_by_category_B.png')

# --- Save market cap weighted results to CSV ---
weighted_cols = [
    'Date',
    'All_Funds_Weighted_Nav_Discount',
    'Infra_Weighted_Nav_Discount',
    'Renew_Weighted_Nav_Discount',
    'All_Funds_Weighted_Implied_Required_Rate',
    'Infra_Weighted_Implied_Required_Rate',
    'Renew_Weighted_Implied_Required_Rate',
    'All_Funds_Weighted_Implied_Equity_Risk_Premium',
    'Infra_Weighted_Implied_Equity_Risk_Premium',
    'Renew_Weighted_Implied_Equity_Risk_Premium',
    'discount_rate_average',
    'risk_free_rate'
]
weighted_df = monthly[weighted_cols].copy()
weighted_df.to_csv('implied_equity_risk_premium_weighted_B.csv', index=False)
print('Saved market cap weighted implied equity risk premium table as implied_equity_risk_premium_weighted_B.csv')

# --- Save full monthly DataFrame to CSV (including market cap weighted columns) ---
monthly.to_csv('implied_equity_risk_premium_full_B.csv', index=False)
print('Saved full implied equity risk premium table as implied_equity_risk_premium_full_B.csv') 