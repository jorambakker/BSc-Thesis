########################################################
# backtest.py
# ------------------------------------------------------
# Contains logic from "Backtest.ipynb" (Code 3).
#  - Load the Optionprices + SP500
#  - Filter to near-ATM
#  - Identify the "third Friday" or "day after" logic
#  - Merge with CG data (For_error, For_revision)
#  - Run a straddle backtest
########################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class BacktestStraddle:
    """
    A class that runs a simple straddle backtest based on the 'For_revision' signal and 'beta' coefficient.
    """
    def __init__(self):
        self.df = pd.DataFrame()
        self.results_df = pd.DataFrame()

    def get_results_df(self):
        return self.results_df

    def load_data(self, path):
        """
        Load pre-merged dataset (the 'strategy.csv' you created),
        filter from 2009 onward, set index to date.
        """
        self.df = pd.read_csv(path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df[self.df["date"].dt.year >= 2009]
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        self.df.set_index('date', inplace=True)

    def populate_signals(self):
        """
        Identify the 1st and 2nd observations each month, set 'position'
        based on 'For_revision' and 'beta'.
        """
        self.df['observation_rank'] = self.df.groupby('year_month').cumcount() + 1
        self.df['position'] = None

        # If beta < 0 => invert the usual logic
        # Otherwise => normal logic
        def determine_position(for_revision, beta):
            if pd.isna(for_revision) or pd.isna(beta):
                return None
            if beta < 0:
                # Invert positions
                if for_revision > 0:
                    return 'short'
                elif for_revision < 0:
                    return 'long'
            else:
                # Normal logic
                if for_revision > 0:
                    return 'long'
                elif for_revision < 0:
                    return 'short'
            return None

        # Apply position on the 2nd observation each month
        idx_filter = (self.df['observation_rank'] == 2)
        self.df.loc[idx_filter, 'position'] = self.df[idx_filter].apply(
            lambda row: determine_position(row['For_revision'], row['beta']), axis=1
        )

    def calculate_option_payoff(self, close_price, strike_price, position):
        """
        Basic payoff for a straddle at expiration:
         - Long straddle payoff = (long call + long put)
         - Short straddle payoff = -(long call + long put)
        """
        call_payoff = max(0, close_price - strike_price)
        put_payoff = max(0, strike_price - close_price)
        if position == 'long':
            return call_payoff + put_payoff
        elif position == 'short':
            return - (call_payoff + put_payoff)
        else:
            return 0

    def run_backtest(self):
        """
        Iterates over each date row. On the second observation, we open a position.
        On the next first observation, we close it (simple approach).
        """
        balance = 0.0
        position = None
        strike_price = None
        trade_cost = 0.0
        open_date = None
        self.results_df = pd.DataFrame()

        for i, (index, row) in enumerate(self.df.iterrows()):
            obs_rank = row['observation_rank']

            # We close position on the first observation, if we have an open position
            if position and obs_rank == 1:
                days_held = (index - open_date).days
                payoff = self.calculate_option_payoff(row['close'], strike_price, position)

                if position == 'long':
                    profit = payoff - trade_cost
                    return_percentage = (profit / trade_cost) if trade_cost != 0 else 0
                else:
                    profit = payoff + trade_cost
                    # For a short position, the "cost" was actually credit to us
                    return_percentage = (profit / abs(trade_cost)) if trade_cost != 0 else 0

                balance += profit
                # Log the trade
                new_row = {
                    'Date': index.date(),
                    'Position': position,
                    'Strike': strike_price,
                    'Close Price': row['close'],
                    'Days Held': days_held,
                    'Payoff': payoff,
                    'Initial Cost/Revenue': trade_cost,
                    'Profit_Loss': profit,
                    'Return_Percentage': return_percentage,
                    'Balance': balance
                }
                self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)

                # Reset position
                position = None

            # We open a new position on the second observation if row['position'] is set
            elif row['position'] and obs_rank == 2:
                open_date = index
                position = row['position']
                strike_price = row['strike_price']
                # Approx. mid price for the straddle
                call_mid = (row['best_bid_C'] + row['best_offer_C']) / 2.0
                put_mid = (row['best_bid_P'] + row['best_offer_P']) / 2.0
                trade_cost = call_mid + put_mid

                if position == 'long':
                    balance -= trade_cost  # Paying for the options
                else:
                    balance += trade_cost  # Receiving premium from shorting the options

        print("Final balance:", balance)
        return self.results_df

def main():
    """
    Main function that:
      - Loads the near-ATM dataset with signals (strategy.csv)
      - Populates signals
      - Runs the backtest
      - Exports results
    """

    # 1. Merge the options & sp500 to get near-ATM data
    #    (Already done in your code snippet; we assume you have 'strategy.csv'.)
    strategy_df = pd.read_csv('strategy.csv')
    # If you prefer, you can replicate the near-ATM filtering logic here:
    #   - read Optionprices(1).csv, sp500.csv
    #   - merge, filter to third Friday, etc.
    #   - finally produce 'strategy.csv'

    # 2. Run the straddle backtest class
    backtester = BacktestStraddle()
    backtester.load_data('strategy.csv')  # or pass strategy_df, but the class expects a path
    backtester.populate_signals()
    results = backtester.run_backtest()

    # 3. Analyze performance
    # Example: simple stats, Sharpe ratio, etc.
    results['SP500_Return'] = results['Close Price'].pct_change()
    # If you have a risk-free rate timeseries, merge it similarly

    results.to_csv('results_strategy.csv', index=False)
    print("Saved backtest results to results_strategy.csv")

    # Optionally plot results
    # NOTE: Single-plot usage only (not subplots), to comply with instructions
    plt.figure()
    plt.scatter(results.index, results['Return_Percentage'], alpha=0.5)
    plt.title('Trade-by-Trade Return (%)')
    plt.xlabel('Trade Number')
    plt.ylabel('Return Percentage')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
