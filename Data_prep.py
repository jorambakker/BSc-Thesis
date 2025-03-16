########################################################
# data_preparation.py
# ------------------------------------------------------
# This script contains all data loading, cleaning,
# merging, and feature-engineering logic.
########################################################

import pandas as pd
import numpy as np

def main():
    """
    Main function orchestrating data cleaning and preparation steps.
    Adjust file paths according to your local environment.
    """
    # --------------------------------------------------
    # 1. Load the datasets
    # --------------------------------------------------
    vol_surface_df = pd.read_csv('/Users/jorambakker/Downloads/Vol-surface 30and60.csv')
    sp500_df = pd.read_csv('/Users/jorambakker/Downloads/SP500.csv')

    vol_surface_df['date'] = pd.to_datetime(vol_surface_df['date'])
    sp500_df['date'] = pd.to_datetime(sp500_df['date'])

    # --------------------------------------------------
    # 2. Merge the vol-surface with SP500 and compute moneyness
    # --------------------------------------------------
    combined_df = pd.merge(vol_surface_df, sp500_df, on='date', how='left')
    combined_df['mnes'] = combined_df.apply(lambda row: row['close'] / row['impl_strike'], axis=1)

    # Save the merged dataset
    combined_df.to_csv('Thesis data.csv', index=False)

    # --------------------------------------------------
    # 3. Create Vilkov30.csv and Vilkov60.csv
    # --------------------------------------------------
    vilkov_df = pd.DataFrame()
    vilkov_df["id"] = combined_df["secid_x"]
    vilkov_df["date"] = combined_df["date"]
    vilkov_df["days"] = combined_df["days"]
    vilkov_df["impl_volatility"] = combined_df["impl_volatility"]
    vilkov_df["mnes"] = combined_df["mnes"]
    vilkov_df["prem"] = combined_df["impl_premium"] / combined_df["close"]
    vilkov_df["delta"] = combined_df["delta"]

    vilkov_df_30 = vilkov_df[vilkov_df["days"] == 30]
    vilkov_df_60 = vilkov_df[vilkov_df["days"] == 60]

    vilkov_df_30.to_csv('Vilkov30.csv', index=False)
    vilkov_df_60.to_csv('Vilkov60.csv', index=False)

    # --------------------------------------------------
    # 4. Filter for days = 30 or 60 and pick near-ATM for each date/cp_flag
    #    Then compute average implied variance for 30 and 60 days
    # --------------------------------------------------
    filtered_data = combined_df[combined_df['days'].isin([30, 60])]

    def closest_moneyness(group):
        return group.iloc[(group['mnes'] - 1).abs().argmin()]

    grouped_data = filtered_data.groupby(['date', 'cp_flag', 'days']) \
                                .apply(closest_moneyness) \
                                .reset_index(drop=True)
    grouped_data.to_csv('Filtered_Thesis_Data.csv', index=False)

    # Compute implied variance, average by date & days
    grouped_data['impl_variance'] = grouped_data['impl_volatility'] ** 2
    average_variance = grouped_data.groupby(['date', 'days'])['impl_variance'].mean().reset_index()

    # Pivot to have columns: avg_impl_variance30, avg_impl_variance60
    pivot_avg_variance = average_variance.pivot(index='date', columns='days', values='impl_variance')
    pivot_avg_variance.columns = [f'avg_impl_variance{col}' for col in pivot_avg_variance.columns]
    final_data = pivot_avg_variance.reset_index(drop=False)
    final_data.to_csv('output_avg_variance.csv', index=False)

    # --------------------------------------------------
    # 5. Convert to only first (or last) trading day of the month
    #    Then compute Forecast(t-1) for CG model
    # --------------------------------------------------
    df = final_data.copy()
    df.sort_values('date', inplace=True)

    # Group by year-month; you used last() in your code:
    first_of_month = df.groupby([df['date'].dt.year, df['date'].dt.month]).last().reset_index(drop=True)

    # h-step forecast from your formula:
    first_of_month['Forecast,t-1 of X(t,t+h)'] = (
        first_of_month['avg_impl_variance60'] * 60 / 30
        - first_of_month['avg_impl_variance30'] * 30 / 30
    ).shift(1)

    first_of_month.to_csv('CG data thesis.csv', index=False)

    # --------------------------------------------------
    # 6. Calculate annualized variance yourself
    # --------------------------------------------------
    sp500_df.sort_values('date', inplace=True)  # ensure sorted
    sp500_df.reset_index(drop=True, inplace=True)

    def calculate_annualized_variance(df_):
        # Create a copy to avoid modifying original
        temp_df = df_.copy()
        temp_df.set_index('date', inplace=True)
        temp_df['Log Returns'] = np.log(temp_df['close'] / temp_df['close'].shift(1))

        temp_df['Realized Variance'] = np.nan
        temp_df['Annualized Variance'] = np.nan

        for idx in temp_df.index:
            target_date = idx + pd.Timedelta(days=30)
            window_data = temp_df['Log Returns'][(temp_df.index > idx) & (temp_df.index <= target_date)]
            if not window_data.empty:
                realized_variance = window_data.var()
                temp_df.at[idx, 'Realized Variance'] = realized_variance
                temp_df.at[idx, 'Annualized Variance'] = realized_variance * 252
        return temp_df.reset_index()

    new_df = calculate_annualized_variance(sp500_df)
    # Merge self-calculated variance with the monthly dataset
    cg = pd.merge(first_of_month, new_df[['date','Annualized Variance']], on='date', how='left')
    cg.to_csv('CG data thesis.csv', index=False)

    # --------------------------------------------------
    # 7. Merge historical volatility from OptionMetrics
    # --------------------------------------------------
    hisvol = pd.read_csv('/Users/jorambakker/Downloads/HisVol.csv')
    hisvol['date'] = pd.to_datetime(hisvol['date'])
    cg1 = pd.merge(cg, hisvol[['date','volatility']], on='date', how='left')
    cg1['historical variance'] = (cg1['volatility'] ** 2).shift(-1)

    # --------------------------------------------------
    # 8. Merge macro-finance uncertainty data (RealUncertaintyToCirculate.xlsx)
    # --------------------------------------------------
    uncertainty_df = pd.read_excel('/Users/jorambakker/Downloads/MacroFinanceUncertainty_202402Update/RealUncertaintyToCirculate.xlsx')
    uncertainty_df['Date'] = pd.to_datetime(uncertainty_df['Date'])

    cg1['month_year'] = cg1['date'].dt.to_period('M')
    uncertainty_df['month_year'] = uncertainty_df['Date'].dt.to_period('M')

    cg_unc = pd.merge(cg1, uncertainty_df, on='month_year', how='left')
    cg_unc["lag_h1"] = cg_unc["h=1"].shift(1)
    cg_unc['h1_change'] = np.log(cg_unc['h=1'] / cg_unc['h=1'].shift(1))

    cg_unc.to_csv('CG Uncertainty.csv', index=False)

    # --------------------------------------------------
    # 9. Stein data prep with monthly or daily approach
    # --------------------------------------------------
    df_stein = pd.merge(final_data, hisvol[['date','volatility']], on='date', how='left')
    df_stein['historical variance'] = (df_stein['volatility']**2)
    df_stein['var30_lag1w'] = df_stein['avg_impl_variance30'].shift(5)
    df_stein['var30_lag2w'] = df_stein['avg_impl_variance30'].shift(10)
    df_stein['var30_lag3w'] = df_stein['avg_impl_variance30'].shift(15)
    df_stein['var30_lag4w'] = df_stein['avg_impl_variance30'].shift(20)

    df_stein.to_csv('Stein data thesis.csv', index=False)
    print("Data preparation completed successfully.")

if __name__ == "__main__":
    main()
