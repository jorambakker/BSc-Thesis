########################################################
# methodology_results.py
# ------------------------------------------------------
# Translates the R-based methodology (Bachelor Thesis.Rmd)
# into Python (pandas + statsmodels).
########################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm

def cg_part_analysis(csv_path="CG Uncertainty.csv"):
    """
    Perform the CG model regressions (equations in your R code).
    """
    # Load data
    d = pd.read_csv(csv_path, parse_dates=['date'])

    # Filter date 1996-2022
    d = d[(d['date'].dt.year >= 1996) & (d['date'].dt.year <= 2022)].copy()

    # For_error and For_revision with Annualized Variance
    d['For_error'] = d['Annualized Variance'] - d['avg_impl_variance30']
    d['For_revision'] = d['avg_impl_variance30'] - d['Forecast,t-1 of X(t,t+h)']

    print("=== CG Model: Regress For_error on For_revision ===")
    # Model 1
    X1 = sm.add_constant(d['For_revision'])
    y1 = d['For_error']
    model1 = sm.OLS(y1, X1).fit()
    print(model1.summary(), "\n")

    # Model 2 (with an interaction: For_revision * h=1 + h=1)
    # Ensure we have column named 'h=1' from the uncertainty dataset
    if 'h=1' in d.columns:
        d['For_error'] = d['Annualized Variance'] - d['avg_impl_variance30']
        d['For_revision'] = d['avg_impl_variance30'] - d['Forecast,t-1 of X(t,t+h)']

        print("=== CG Model with uncertainty extension (equation 6?) ===")
        d['interaction'] = d['For_revision'] * d['h=1']
        X2 = sm.add_constant(d[['For_revision', 'h=1', 'interaction']])
        y2 = d['For_error']
        model2 = sm.OLS(y2, X2).fit()
        print(model2.summary(), "\n")

        # Another version with log-lag or h1_change if needed
        # Example:
        print("=== Another extension: For_error ~ For_revision + For_revision*lag_h1 ===")
        if 'lag_h1' in d.columns:
            d['interaction2'] = d['For_revision'] * d['lag_h1']
            X3 = sm.add_constant(d[['For_revision', 'lag_h1', 'interaction2']])
            y3 = d['For_error']
            model3 = sm.OLS(y3, X3).fit()
            print(model3.summary(), "\n")

    # Also do the same approach with the historical variance
    # that you called "NEW approach of CG with historical volatility"
    if 'historical variance' in d.columns:
        d['For_error'] = d['historical variance'] - d['avg_impl_variance30']
        d['For_revision'] = d['avg_impl_variance30'] - d['Forecast,t-1 of X(t,t+h)']

        print("=== CG Model with historical variance ===")
        X4 = sm.add_constant(d['For_revision'])
        y4 = d['For_error']
        model4 = sm.OLS(y4, X4).fit()
        print(model4.summary(), "\n")

        # CG model with uncertainty extension
        if 'h=1' in d.columns:
            d['interaction_hist'] = d['For_revision'] * d['h=1']
            X5 = sm.add_constant(d[['For_revision', 'interaction_hist']])
            y5 = d['For_error']
            model5 = sm.OLS(y5, X5).fit()
            print("=== CG model with historical variance + uncertainty extension ===")
            print(model5.summary(), "\n")


def stein_part_analysis_monthly(csv_path="Stein data thesis.csv"):
    """
    Perform the Stein model regressions on monthly data.
    The R code included an AR(1) approach and 'spread' calculations.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[(df['date'].dt.year >= 1996) & (df['date'].dt.year <= 2022)].copy()

    # Possibly the monthly version
    # eq. 3: AR(1) on avg_impl_variance30
    df['lag_var30'] = df['avg_impl_variance30'].shift(1)
    df.dropna(subset=['avg_impl_variance30','lag_var30'], inplace=True)

    print("=== STEIN AR(1) with monthly data (equation 3) ===")
    X = sm.add_constant(df['lag_var30'])
    y = df['avg_impl_variance30']
    model_ar1 = sm.OLS(y, X).fit()
    print(model_ar1.summary(), "\n")

    # eq. 1: spread between short & long
    # We'll define mean of historical variance
    mean_variance = df['historical variance'].mean(skipna=True)
    df['spread_short'] = df['avg_impl_variance30'] - mean_variance
    df['spread_long'] = df['avg_impl_variance60'] - mean_variance

    print("=== STEIN eq.1: spread_long ~ spread_short ===")
    X2 = sm.add_constant(df['spread_short'])
    y2 = df['spread_long']
    model_spread = sm.OLS(y2, X2).fit()
    print(model_spread.summary(), "\n")


def stein_part_analysis_daily(csv_path="Stein data thesis.csv"):
    """
    Perform the Stein model regressions on daily data, 
    including the lag(1w,2w,3w,4w) approach for var and vol.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[(df['date'].dt.year >= 1996) & (df['date'].dt.year <= 2022)].copy()

    # Regress avg_impl_variance30 on var30_lag1w, var30_lag2w, etc.
    # 1 week lag -> power(1/1)
    print("=== Stein daily: p from 1-week lag ===")
    df_1w = df.dropna(subset=['avg_impl_variance30','var30_lag1w'])
    X_1w = sm.add_constant(df_1w['var30_lag1w'])
    y_1w = df_1w['avg_impl_variance30']
    model_1w = sm.OLS(y_1w, X_1w).fit()
    print(model_1w.summary(), "\n")

    # 2 week lag -> we interpret the slope^(1/2)
    print("=== Stein daily: p from 2-week lag => slope^(1/2) ===")
    df_2w = df.dropna(subset=['avg_impl_variance30','var30_lag2w'])
    X_2w = sm.add_constant(df_2w['var30_lag2w'])
    y_2w = df_2w['avg_impl_variance30']
    model_2w = sm.OLS(y_2w, X_2w).fit()
    print(model_2w.summary())
    print(f"p = {model_2w.params['var30_lag2w'] ** (1/2)} \n")

    # 3 week lag
    print("=== Stein daily: p from 3-week lag => slope^(1/3) ===")
    df_3w = df.dropna(subset=['avg_impl_variance30','var30_lag3w'])
    X_3w = sm.add_constant(df_3w['var30_lag3w'])
    y_3w = df_3w['avg_impl_variance30']
    model_3w = sm.OLS(y_3w, X_3w).fit()
    print(model_3w.summary())
    print(f"p = {model_3w.params['var30_lag3w'] ** (1/3)} \n")

    # 4 week lag
    print("=== Stein daily: p from 4-week lag => slope^(1/4) ===")
    df_4w = df.dropna(subset=['avg_impl_variance30','var30_lag4w'])
    X_4w = sm.add_constant(df_4w['var30_lag4w'])
    y_4w = df_4w['avg_impl_variance30']
    model_4w = sm.OLS(y_4w, X_4w).fit()
    print(model_4w.summary())
    print(f"p = {model_4w.params['var30_lag4w'] ** (1/4)} \n")

    # Equation 1 with daily data
    mean_variance2 = df['historical variance'].mean(skipna=True)
    df['spread_short'] = df['avg_impl_variance30'] - mean_variance2
    df['spread_long'] = df['avg_impl_variance60'] - mean_variance2

    print("=== Stein eq.1 (daily): spread_long ~ spread_short ===")
    df_spread = df.dropna(subset=['spread_short','spread_long'])
    X_spr = sm.add_constant(df_spread['spread_short'])
    y_spr = df_spread['spread_long']
    model_spr = sm.OLS(y_spr, X_spr).fit()
    print(model_spr.summary(), "\n")

    # Also do the “volatility” version (sqrt of the above)
    df['avg_vol30'] = np.sqrt(df['avg_impl_variance30'])
    df['avg_vol60'] = np.sqrt(df['avg_impl_variance60'])
    df['vol30_lag1w'] = np.sqrt(df['var30_lag1w'])
    df['vol30_lag2w'] = np.sqrt(df['var30_lag2w'])
    df['vol30_lag3w'] = np.sqrt(df['var30_lag3w'])
    df['vol30_lag4w'] = np.sqrt(df['var30_lag4w'])

    # Re-do the slope checks for volatility
    print("=== Stein daily with vol (1w lag) ===")
    df_1w_vol = df.dropna(subset=['avg_vol30','vol30_lag1w'])
    Xv_1w = sm.add_constant(df_1w_vol['vol30_lag1w'])
    yv_1w = df_1w_vol['avg_vol30']
    modelv_1w = sm.OLS(yv_1w, Xv_1w).fit()
    print(modelv_1w.summary(), "\n")

    # etc. for 2, 3, 4 weeks...
    # You can replicate as above for p = slope^(1/2), etc.

    mean_volatility = df['volatility'].mean(skipna=True)
    df['spread_short1'] = df['avg_vol30'] - mean_volatility
    df['spread_long1'] = df['avg_vol60'] - mean_volatility

    print("=== Stein eq.1 (daily) in volatility terms: spread_long1 ~ spread_short1 ===")
    df_spread_vol = df.dropna(subset=['spread_short1','spread_long1'])
    Xsv = sm.add_constant(df_spread_vol['spread_short1'])
    ysv = df_spread_vol['spread_long1']
    model_spr_vol = sm.OLS(ysv, Xsv).fit()
    print(model_spr_vol.summary(), "\n")


def main():
    # CG regressions
    cg_part_analysis("CG Uncertainty.csv")

    # Stein monthly analysis
    stein_part_analysis_monthly("Stein data thesis.csv")

    # Stein daily analysis
    stein_part_analysis_daily("Stein data thesis.csv")


if __name__ == "__main__":
    main()
