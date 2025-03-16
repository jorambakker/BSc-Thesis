########################################################
# Methodology.py (updated)
# ------------------------------------------------------
# Translates your R-based methodology into Python
# using pandas and statsmodels.
########################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm

def cg_part_analysis(csv_path="CG Uncertainty.csv"):
    """
    Perform the CG model regressions.
    """
    # Load data and parse the date column
    d = pd.read_csv(csv_path, parse_dates=['date'])
    d = d[(d['date'].dt.year >= 1996) & (d['date'].dt.year <= 2022)].copy()

    # Calculate For_error and For_revision (using Annualized Variance)
    d['For_error'] = d['Annualized Variance'] - d['avg_impl_variance30']
    d['For_revision'] = d['avg_impl_variance30'] - d['Forecast,t-1 of X(t,t+h)']

    # ---- Model 1: Basic regression ---------------------
    print("=== CG Model: Regress For_error on For_revision ===")
    d_model1 = d[['For_revision', 'For_error']].replace([np.inf, -np.inf], np.nan).dropna()
    X1 = sm.add_constant(d_model1['For_revision'])
    y1 = d_model1['For_error']
    model1 = sm.OLS(y1, X1).fit()
    print(model1.summary(), "\n")

    # ---- Model 2: CG Model with uncertainty extension ---------------------
    if 'h=1' in d.columns:
        # Recalculate For_error and For_revision in case they need to be reset
        d['For_error'] = d['Annualized Variance'] - d['avg_impl_variance30']
        d['For_revision'] = d['avg_impl_variance30'] - d['Forecast,t-1 of X(t,t+h)']
        print("=== CG Model with uncertainty extension (equation 6?) ===")
        d['interaction'] = d['For_revision'] * d['h=1']
        # Clean data: drop any rows with NaN or inf in the relevant columns
        d_model2 = d[['For_revision', 'h=1', 'interaction', 'For_error']].replace([np.inf, -np.inf], np.nan).dropna()
        X2 = sm.add_constant(d_model2[['For_revision', 'h=1', 'interaction']])
        y2 = d_model2['For_error']
        model2 = sm.OLS(y2, X2).fit()
        print(model2.summary(), "\n")

        # ---- Model 3: Extension with lag_h1 if available ---------------------
        if 'lag_h1' in d.columns:
            print("=== Another extension: For_error ~ For_revision + For_revision * lag_h1 ===")
            d['interaction2'] = d['For_revision'] * d['lag_h1']
            d_model3 = d[['For_revision', 'lag_h1', 'interaction2', 'For_error']].replace([np.inf, -np.inf], np.nan).dropna()
            X3 = sm.add_constant(d_model3[['For_revision', 'lag_h1', 'interaction2']])
            y3 = d_model3['For_error']
            model3 = sm.OLS(y3, X3).fit()
            print(model3.summary(), "\n")

    # ---- Model 4: CG Model with historical variance ---------------------
    if 'historical variance' in d.columns:
        d['For_error'] = d['historical variance'] - d['avg_impl_variance30']
        d['For_revision'] = d['avg_impl_variance30'] - d['Forecast,t-1 of X(t,t+h)']
        print("=== CG Model with historical variance ===")
        d_model4 = d[['For_revision', 'For_error']].replace([np.inf, -np.inf], np.nan).dropna()
        X4 = sm.add_constant(d_model4['For_revision'])
        y4 = d_model4['For_error']
        model4 = sm.OLS(y4, X4).fit()
        print(model4.summary(), "\n")

        # ---- Model 5: CG model with historical variance + uncertainty extension ---------------------
        if 'h=1' in d.columns:
            d['interaction_hist'] = d['For_revision'] * d['h=1']
            d_model5 = d[['For_revision', 'interaction_hist', 'For_error']].replace([np.inf, -np.inf], np.nan).dropna()
            X5 = sm.add_constant(d_model5[['For_revision', 'interaction_hist']])
            y5 = d_model5['For_error']
            model5 = sm.OLS(y5, X5).fit()
            print("=== CG model with historical variance + uncertainty extension ===")
            print(model5.summary(), "\n")


def stein_part_analysis_monthly(csv_path="Stein data thesis.csv"):
    """
    Perform the Stein model regressions on monthly data.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[(df['date'].dt.year >= 1996) & (df['date'].dt.year <= 2022)].copy()

    # AR(1) regression on avg_impl_variance30
    df['lag_var30'] = df['avg_impl_variance30'].shift(1)
    df_model = df[['avg_impl_variance30', 'lag_var30']].replace([np.inf, -np.inf], np.nan).dropna()
    print("=== STEIN AR(1) with monthly data (equation 3) ===")
    X = sm.add_constant(df_model['lag_var30'])
    y = df_model['avg_impl_variance30']
    model_ar1 = sm.OLS(y, X).fit()
    print(model_ar1.summary(), "\n")

    # Equation 1: spread between short & long
    mean_variance = df['historical variance'].mean(skipna=True)
    df['spread_short'] = df['avg_impl_variance30'] - mean_variance
    df['spread_long'] = df['avg_impl_variance60'] - mean_variance
    df_model2 = df[['spread_short', 'spread_long']].replace([np.inf, -np.inf], np.nan).dropna()
    print("=== STEIN eq.1: spread_long ~ spread_short ===")
    X2 = sm.add_constant(df_model2['spread_short'])
    y2 = df_model2['spread_long']
    model_spread = sm.OLS(y2, X2).fit()
    print(model_spread.summary(), "\n")


def stein_part_analysis_daily(csv_path="Stein data thesis.csv"):
    """
    Perform the Stein model regressions on daily data.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[(df['date'].dt.year >= 1996) & (df['date'].dt.year <= 2022)].copy()

    # 1-week lag regression
    print("=== Stein daily: p from 1-week lag ===")
    df_1w = df[['avg_impl_variance30','var30_lag1w']].replace([np.inf, -np.inf], np.nan).dropna()
    X_1w = sm.add_constant(df_1w['var30_lag1w'])
    y_1w = df_1w['avg_impl_variance30']
    model_1w = sm.OLS(y_1w, X_1w).fit()
    print(model_1w.summary(), "\n")

    # 2-week lag regression: slope^(1/2)
    print("=== Stein daily: p from 2-week lag => slope^(1/2) ===")
    df_2w = df[['avg_impl_variance30','var30_lag2w']].replace([np.inf, -np.inf], np.nan).dropna()
    X_2w = sm.add_constant(df_2w['var30_lag2w'])
    y_2w = df_2w['avg_impl_variance30']
    model_2w = sm.OLS(y_2w, X_2w).fit()
    print(model_2w.summary())
    print(f"p = {model_2w.params['var30_lag2w'] ** (1/2)} \n")

    # 3-week lag regression: slope^(1/3)
    print("=== Stein daily: p from 3-week lag => slope^(1/3) ===")
    df_3w = df[['avg_impl_variance30','var30_lag3w']].replace([np.inf, -np.inf], np.nan).dropna()
    X_3w = sm.add_constant(df_3w['var30_lag3w'])
    y_3w = df_3w['avg_impl_variance30']
    model_3w = sm.OLS(y_3w, X_3w).fit()
    print(model_3w.summary())
    print(f"p = {model_3w.params['var30_lag3w'] ** (1/3)} \n")

    # 4-week lag regression: slope^(1/4)
    print("=== Stein daily: p from 4-week lag => slope^(1/4) ===")
    df_4w = df[['avg_impl_variance30','var30_lag4w']].replace([np.inf, -np.inf], np.nan).dropna()
    X_4w = sm.add_constant(df_4w['var30_lag4w'])
    y_4w = df_4w['avg_impl_variance30']
    model_4w = sm.OLS(y_4w, X_4w).fit()
    print(model_4w.summary())
    print(f"p = {model_4w.params['var30_lag4w'] ** (1/4)} \n")

    # Equation 1 (daily) for spread in variance terms
    mean_variance2 = df['historical variance'].mean(skipna=True)
    df['spread_short'] = df['avg_impl_variance30'] - mean_variance2
    df['spread_long'] = df['avg_impl_variance60'] - mean_variance2
    df_model_spread = df[['spread_short', 'spread_long']].replace([np.inf, -np.inf], np.nan).dropna()
    print("=== Stein eq.1 (daily): spread_long ~ spread_short ===")
    X_spr = sm.add_constant(df_model_spread['spread_short'])
    y_spr = df_model_spread['spread_long']
    model_spr = sm.OLS(y_spr, X_spr).fit()
    print(model_spr.summary(), "\n")

    # Volatility approach: take square roots of variances
    df['avg_vol30'] = np.sqrt(df['avg_impl_variance30'])
    df['avg_vol60'] = np.sqrt(df['avg_impl_variance60'])
    df['vol30_lag1w'] = np.sqrt(df['var30_lag1w'])
    df['vol30_lag2w'] = np.sqrt(df['var30_lag2w'])
    df['vol30_lag3w'] = np.sqrt(df['var30_lag3w'])
    df['vol30_lag4w'] = np.sqrt(df['var30_lag4w'])
    
    print("=== Stein daily with volatility (1w lag) ===")
    df_1w_vol = df[['avg_vol30','vol30_lag1w']].replace([np.inf, -np.inf], np.nan).dropna()
    Xv_1w = sm.add_constant(df_1w_vol['vol30_lag1w'])
    yv_1w = df_1w_vol['avg_vol30']
    modelv_1w = sm.OLS(yv_1w, Xv_1w).fit()
    print(modelv_1w.summary(), "\n")

    mean_volatility = df['volatility'].mean(skipna=True)
    df['spread_short1'] = df['avg_vol30'] - mean_volatility
    df['spread_long1'] = df['avg_vol60'] - mean_volatility
    df_model_spread_vol = df[['spread_short1', 'spread_long1']].replace([np.inf, -np.inf], np.nan).dropna()
    print("=== Stein eq.1 (daily) in volatility terms: spread_long1 ~ spread_short1 ===")
    Xsv = sm.add_constant(df_model_spread_vol['spread_short1'])
    ysv = df_model_spread_vol['spread_long1']
    model_spr_vol = sm.OLS(ysv, Xsv).fit()
    print(model_spr_vol.summary(), "\n")


def main():
    cg_part_analysis("CG Uncertainty.csv")
    stein_part_analysis_monthly("Stein data thesis.csv")
    stein_part_analysis_daily("Stein data thesis.csv")


if __name__ == "__main__":
    main()
