import sys
sys.path.append(".")
import csv
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

# function should always be named conditional_probability
from contents.conditional_probability_obj import Conditional_Probability
from contents.simple_return import Simple_Return
# Goal of this function
# 1.) Take in a conditonal probability object
# 2.) Set fixed paramters such as target probability, ticker, interval
# 3.) Test the model an N number of times based on the fixed parameters
# 4.) Export the results to a CSV file, to be used in futher analysis

class Backtest():
    def __init__(self):
        #Argument Checks
        if len(sys.argv) > 1:
            # Checkiing if inputs are valid tickers and integers
            number_check = None
            try:
                number_check = int(sys.argv[1])
            except ValueError:
                pass
            if isinstance(number_check, int):
                print("-" * 50)
                print(f"Argument: {sys.argv[1]} -> provided integer, not ticker")
                print("Usage: python backtest.py <ticker symbol> <number_of_iterations>")
                print("-" * 50)
                sys.exit(1)
        if len(sys.argv) >= 4:
            # checking amount of inputs
            print("-" * 50)
            print("Invalid argument. Too many inputs")
            print("Usage: python backtest.py <ticker symbol> <number_of_iterations>")
            print("-" * 50)
            sys.exit(1)
        if len(sys.argv) == 2:
            self.arg = str(sys.argv[1])
            self.arg2 = 1
        elif len(sys.argv) > 2:
            try:
                self.arg = str(sys.argv[1])
                self.arg2 = int(sys.argv[2])
            except ValueError:
                print("-" * 50)
                print("Invalid argument. Please provide a valid ticker and integer for the number of iterations. This should be a positive integer")
                print("Usage: python backtest.py <ticker symbol> <number_of_iterations>")
                print("-" * 50)
                sys.exit(1)
        else:
            print("-" * 50)
            print("No argument provided. Please provide a valid ticker and integer for the number of iterations. This should be a positive integer")
            print("Usage: python backtest.py <ticker symbol> <number_of_iterations>")
            print("-" * 50)
            sys.exit(1)

# Inputs
        self.universe = datetime.strptime("2000-01-01", "%Y-%m-%d").date()
        self.today = date.today()
        self.end_date_range = self.today - timedelta(days=1)  # today
        self.ticker = self.arg
        self.interval = "1d"
        self.target_probability = 0.55
# Declare df outside of the loop to avoid re-downloading data each iteration
        print(f"Downloading {self.ticker}...")
        self.df = yf.download(self.ticker, start = self.universe, end = str(date.today() - timedelta(1)), interval = self.interval, multi_level_index=False)
        #Check if input ticker is a valid ticker
        if self.df.empty:
            print("-" * 50)
            print("-" * 50)
            print(f'Job halted: {self.ticker} is an invalid Ticker')
            print("-" * 50)
            print("-" * 50)
            sys.exit(1)
        if self.ticker != 'SPY':
            print(f"Downloading SPY...")
            self.spydf = yf.download('SPY', start = self.universe, end = str(date.today() - timedelta(1)), interval = self.interval, multi_level_index=False)

    def backtest(self):
        for i in range(self.arg2):
            # Default date range
            # use a random date from a range of 25 years ago to today 

            print(f"Backtest {i + 1} of {self.arg2}...")
            # Generate a random start date within the range
            random_days = random.randint(0, (self.end_date_range - self.universe).days)
            input_start_date = pd.to_datetime(self.universe + timedelta(days=random_days))
            input_end_date = pd.to_datetime(input_start_date + timedelta(days=365) ) # Default end date is 1 year later
            
            # Check if input_end_date is valid
            if input_end_date.date() < self.today:
                # Testing Modules for a specific date
                #input_start_date = "2000-07-29"
                #input_end_date = "2001-07-29"

                model = Conditional_Probability(ticker=self.ticker, interval=self.interval, start=self.universe, optional_df=self.df)
                model.run_algo(target_probability=self.target_probability, start_date=input_start_date, end_date=input_end_date, return_table=True)
                real_start_date = model.df.index[0]  # Get the first date in the DataFrame
                real_end_date = model.df.index[-1]  # Get the last date in the DataFrame
                if self.ticker != 'SPY':
                    spy_model = Simple_Return(ticker=self.ticker, interval=self.interval, start=input_start_date, end=real_end_date, optional_df=self.spydf)
                backtest_result = model.backtest(return_table=False, print_statement=False, model_return=True)
                buy_hold_result = model.backtest(return_table=False, buy_hold=True)

                #Sharpe Ratios
                backtest_sharpe = model.sharpe_ratio(return_model=True)
                buy_hold_sharpe = model.sharpe_ratio(return_buy_hold=True)
                if self.ticker != 'SPY':
                    spy_result = spy_model.get_return()
                    spy_sharpe = spy_model.get_sharpe()
                    spy_delta = backtest_result - spy_result
                    print(f"SPY Buy/Hold Result: {spy_result}")
                delta = backtest_result - buy_hold_result

                # Export to CSV
                def export_to_csv(backtest_result, buy_hold_result, filename=f"{self.ticker}_backtest_results.csv"):
                    #Check for Overload Error
                    if np.isnan(backtest_sharpe):
                        print(f"Error: Errors found in backtest do to overload. Backtest #{i + 1} scrapped.")
                        return
                    else:
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            if csvfile.tell() == 0:  # Check if file is empty
                                # Write header only if the file is empty
                                if self.ticker != 'SPY':
                                    writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe', 'SPY Buy/Hold Result', 'SPY Sharpe', 'SPY Delta'])
                                else:
                                    writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe'])    # header
                            if self.ticker != 'SPY':
                                writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe, spy_result, spy_sharpe, round(spy_delta,2)]) # data
                            else:
                                writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe]) # data
                print("Done")
                export_to_csv(backtest_result, buy_hold_result)
            elif input_end_date.date() >= self.today:
                print(f"End Date is not valid, no entry recorded")
                print(f"{input_end_date}")
        print("-" * 50)
        print("-" * 50)
        print("Backtest completed")
        print("-" * 50)
        print("-" * 50)

if __name__ == "__main__":
    test = Backtest()
    test.backtest()