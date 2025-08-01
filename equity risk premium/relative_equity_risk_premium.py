import pandas as pd
import os
import datetime

# Load the intermediate monthly data
monthly = pd.read_csv(os.path.join('..', 'output', 'data', 'monthly_intermediate.csv'))
monthly['Date'] = pd.to_datetime(monthly['Date'])

# Get base values for 2015-05-01
base_date = pd.Timestamp('2015-05-01')
base_infra = monthly.loc[monthly['Date'] == base_date, 'Infra_Implied_Equity_Risk_Premium'].values
base_renew = monthly.loc[monthly['Date'] == base_date, 'Renew_Implied_Equity_Risk_Premium'].values

if len(base_infra) == 0 or base_infra[0] == 0:
    raise ValueError('Base value for Infrastructure not found or is zero.')
if len(base_renew) == 0 or base_renew[0] == 0:
    raise ValueError('Base value for Renewables not found or is zero.')

# Compute relative change for all months
monthly['Infra_Relative_Equity_Risk_Premium'] = monthly['Infra_Implied_Equity_Risk_Premium'] / base_infra[0]
monthly['Renew_Relative_Equity_Risk_Premium'] = monthly['Renew_Implied_Equity_Risk_Premium'] / base_renew[0]

# Save to CSV
os.makedirs('output/data', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
relative_csv_path = os.path.join('..', 'output', 'data', f'relative_equity_risk_premium_{timestamp}.csv')
relative_cols = ['Date', 'Infra_Relative_Equity_Risk_Premium', 'Renew_Relative_Equity_Risk_Premium']
monthly[relative_cols].to_csv(relative_csv_path, index=False)
print(f'Saved relative equity risk premium for infra and renewables to {relative_csv_path}')

# Print the relative change for each month
for _, row in monthly.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')
    infra_rel = row['Infra_Relative_Equity_Risk_Premium']
    renew_rel = row['Renew_Relative_Equity_Risk_Premium']
    print(f"{date_str}: Infra = {infra_rel:.3f}, Renew = {renew_rel:.3f}") 