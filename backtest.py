import sys
sys.path.append(".")
import csv
import random
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from datetime import date, timedelta, datetime


#BUG: end dates that are past todays date - Fixed

# function should always be named conditional_probability
from contents.conditional_probability_obj import Conditional_Probability

# Goal of this function
# 1.) Take in a conditonal probability object
# 2.) Set fixed paramters such as target probability, ticker, interval
# 3.) Test the model an N number of times based on the fixed parameters
# 4.) Export the results to a CSV file, to be used in futher analysis

# Inputs
universe = datetime.strptime("2000-01-01", "%Y-%m-%d").date()
today = date.today()
end_date_range = today - timedelta(days=1)  # today
ticker = "SPY"
interval = "1d"
target_probability = 0.55
# Declare df outside of the loop to avoid re-downloading data each iteration
df = yf.download(ticker, start = universe, end = str(date.today() - timedelta(1)), interval = interval, multi_level_index=False)

def backtest():
    if len(sys.argv) == 1:
        arg = 1
    elif len(sys.argv) > 1:
        try:
            arg = int(sys.argv[1])
        except ValueError:
            print("-" * 50)
            print("Invalid argument. Please provide a valid integer for the number of iterations. This should be a positive integer")
            print("Usage: python backtest.py <number_of_iterations>")
            print("-" * 50)
            sys.exit(1)
    else:
        print("-" * 50)
        print("No argument provided. Please provide a valid integer for the number of iterations. This should be a positive integer")
        print("Usage: python backtest.py <number_of_iterations>")
        print("-" * 50)
        sys.exit(1)

    for i in range(arg):
        # Default date range
        # use a random date from a range of 25 years ago to today 

        print(f"Backtest {i + 1} of {arg}...")
        # Generate a random start date within the range
        random_days = random.randint(0, (end_date_range - universe).days)
        input_start_date = pd.to_datetime(universe + timedelta(days=random_days))
        input_end_date = pd.to_datetime(input_start_date + timedelta(days=365) ) # Default end date is 1 year later
        # Check if input_end_date is valid
        if input_end_date.date() < today:
            model = Conditional_Probability(ticker=ticker, interval=interval, start=universe, optional_df=df)
            model.run_algo(target_probability=target_probability, start_date=input_start_date, end_date=input_end_date, print_table=True)
            real_start_date = model.df.index[0]  # Get the first date in the DataFrame
            real_end_date = model.df.index[-1]  # Get the last date in the DataFrame
            backtest_result = model.backtest(print_table=False, print_statement=False, model_return=True)
            buy_hold_result = model.backtest(print_table=False, buy_hold=True)

            #Sharpe Ratios
            backtest_sharpe = model.sharpe_ratio(return_model=True)
            buy_hold_sharpe = model.sharpe_ratio(return_buy_hold=True)

            delta = backtest_result - buy_hold_result

            # Export to CSV
            def export_to_csv(backtest_result, buy_hold_result, filename="backtest_results.csv"): 
                with open(filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if csvfile.tell() == 0:  # Check if file is empty
                        # Write header only if the file is empty
                        writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe']) # header
                    writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe]) # data

            print("Done")
            export_to_csv(backtest_result, buy_hold_result)
        elif input_end_date.date() >= today:
            print(f"End Date is not valid, no entry recorded")
            print(f"{input_end_date}")
    print("-" * 50)
    print("-" * 50)
    print("Backtest completed")
    print("-" * 50)
    print("-" * 50)

def return_hist(csv_file, backtest=True, buy_hold=False, both=False):
    fig1 = plt.figure(figsize=(12, 6))
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. Please run the backtest first.")
        return

    backtest_mini, backtest_maxi = df['Model Result'].min(), df['Model Result'].max()
    buy_hold_mini, buy_hold_maxi = df['Buy/Hold Result'].min(), df['Buy/Hold Result'].max()
    backtest_mean = df['Model Result'].mean()
    buy_hold_mean = df['Buy/Hold Result'].mean()
    backtest_std = df['Model Result'].std()
    buy_hold_std = df['Buy/Hold Result'].std()
    backtest_overlay = np.linspace(backtest_mini, backtest_maxi, 100)
    buy_hold_overlay = np.linspace(buy_hold_mini, buy_hold_maxi, 100)
    backtest_p = norm.pdf(backtest_overlay, backtest_mean, backtest_std)
    buy_hold_p = norm.pdf(buy_hold_overlay, buy_hold_mean, buy_hold_std)
    print("-" * 50)
    print(f"n = {len(df)}")
    print("-" * 50)

    if both:
        backtest = True
        buy_hold = True

    if backtest:
        plt.hist(df['Model Result'], bins=50, density=True, color='blue', alpha=0.5, label='Model Result')
        plt.xlim(backtest_mini, backtest_maxi)
        plt.plot(backtest_overlay, backtest_p, 'k', label='Model PDF')
        plt.axvline(backtest_mean, color='blue', linestyle='dashed', label='Model Mean')
        plt.text(backtest_mean, plt.ylim()[1] * .9, f'{round(backtest_mean, 2)}%', color='black', ha='center')
        plt.title('Normal Distribution of Model Results')
        if buy_hold is False:
            #Standard Deviation Plots
            plt.axvline(backtest_mean + backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean - backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean + 2 * backtest_std, color='purple', linestyle='dashed')
            plt.axvline(backtest_mean - 2 * backtest_std, color='purple', linestyle='dashed')

            #labels
            plt.text(backtest_mean + backtest_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(backtest_mean + 2 * backtest_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(backtest_mean - 2 * backtest_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(backtest_mean - backtest_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("Model Results:")
        print(f"Model Mean: {round(backtest_mean, 2)}%")
        print(f"Model Std: {round(backtest_std, 2)}%")
    
    if buy_hold:
        plt.hist(df['Buy/Hold Result'], bins=50, density=True, color='orange', alpha=0.5, label='Buy/Hold Result')
        plt.xlim(buy_hold_mini, buy_hold_maxi)
        plt.plot(buy_hold_overlay, buy_hold_p, 'r', label='Buy/Hold PDF')
        plt.axvline(buy_hold_mean, color='orange', linestyle='dashed', label='Buy/Hold Mean')
        plt.text(buy_hold_mean, plt.ylim()[1] * .9, f'{round(buy_hold_mean, 2)}%', color='black', ha='center')
        if backtest is False:
            #Standard Deviation Plots
            plt.axvline(buy_hold_mean + buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean - buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean + 2 * buy_hold_std, color='purple', linestyle='dashed')
            plt.axvline(buy_hold_mean - 2 * buy_hold_std, color='purple', linestyle='dashed')

            #labels

            plt.text(buy_hold_mean + buy_hold_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(buy_hold_mean + 2 * buy_hold_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(buy_hold_mean - 2 * buy_hold_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(buy_hold_mean - buy_hold_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("-" * 50)
        print("Buy/Hold Results:")
        print(f"Buy/Hold Mean: {round(buy_hold_mean, 2)}%")
        print(f"Buy/Hold Std: {round(buy_hold_std, 2)}%")

    if both:
        plt.title('Normal Distribution of Backtest and Buy/Hold Results')
        plt.xlim(min(backtest_mini, buy_hold_mini), max(backtest_maxi, buy_hold_maxi))

    plt.xlabel('Return (%)')
    plt.ylabel('Density')
    plt.legend()

    print("-" * 50)

def sharpe_hist(csv_file, backtest=True, buy_hold=False, both=False):
    fig1 = plt.figure(figsize=(12, 6))
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. Please run the backtest first.")
        return

    backtest_mini, backtest_maxi = df['Model Sharpe'].min(), df['Model Sharpe'].max()
    buy_hold_mini, buy_hold_maxi = df['Buy/Hold Sharpe'].min(), df['Buy/Hold Sharpe'].max()
    backtest_mean = df['Model Sharpe'].mean()
    buy_hold_mean = df['Buy/Hold Sharpe'].mean()
    backtest_std = df['Model Sharpe'].std()
    buy_hold_std = df['Buy/Hold Sharpe'].std()
    backtest_overlay = np.linspace(backtest_mini, backtest_maxi, 100)
    buy_hold_overlay = np.linspace(buy_hold_mini, buy_hold_maxi, 100)
    backtest_p = norm.pdf(backtest_overlay, backtest_mean, backtest_std)
    buy_hold_p = norm.pdf(buy_hold_overlay, buy_hold_mean, buy_hold_std)
    print("-" * 50)
    print(f"n = {len(df)}")
    print("-" * 50)

    if both:
        backtest = True
        buy_hold = True

    if backtest:
        plt.hist(df['Model Sharpe'], bins=50, density=True, color='blue', alpha=0.5, label='Model Sharpe')
        plt.xlim(backtest_mini, backtest_maxi)
        plt.plot(backtest_overlay, backtest_p, 'k', label='Model PDF')
        plt.axvline(backtest_mean, color='blue', linestyle='dashed', label='Model Mean')
        plt.text(backtest_mean, plt.ylim()[1] * .9, f'{round(backtest_mean, 2)}', color='black', ha='center')
        plt.title('Normal Distribution of Model Sharpe')
        if buy_hold is False:
            #Standard Deviation Plots
            plt.axvline(backtest_mean + backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean - backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean + 2 * backtest_std, color='purple', linestyle='dashed')
            plt.axvline(backtest_mean - 2 * backtest_std, color='purple', linestyle='dashed')

            #labels
            plt.text(backtest_mean + backtest_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(backtest_mean + 2 * backtest_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(backtest_mean - 2 * backtest_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(backtest_mean - backtest_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("Model Sharpe:")
        print(f"Model Mean: {round(backtest_mean, 4)}")
        print(f"Model Std: {round(backtest_std, 4)}")
    
    if buy_hold:
        plt.hist(df['Buy/Hold Sharpe'], bins=50, density=True, color='orange', alpha=0.5, label='Buy/Hold Sharpe')
        plt.xlim(buy_hold_mini, buy_hold_maxi)
        plt.plot(buy_hold_overlay, buy_hold_p, 'r', label='Buy/Hold PDF')
        plt.axvline(buy_hold_mean, color='orange', linestyle='dashed', label='Buy/Hold Mean')
        plt.text(buy_hold_mean, plt.ylim()[1] * .9, f'{round(buy_hold_mean, 2)}', color='black', ha='center')
        if backtest is False:
            #Standard Deviation Plots
            plt.axvline(buy_hold_mean + buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean - buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean + 2 * buy_hold_std, color='purple', linestyle='dashed')
            plt.axvline(buy_hold_mean - 2 * buy_hold_std, color='purple', linestyle='dashed')

            #labels

            plt.text(buy_hold_mean + buy_hold_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(buy_hold_mean + 2 * buy_hold_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(buy_hold_mean - 2 * buy_hold_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(buy_hold_mean - buy_hold_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("-" * 50)
        print("Buy/Hold Sharpe:")
        print(f"Buy/Hold Mean: {round(buy_hold_mean, 4)}")
        print(f"Buy/Hold Std: {round(buy_hold_std, 4)}")

    if both:
        plt.title('Normal Distribution of Model and Buy/Hold Sharpe')
        plt.xlim(min(backtest_mini, buy_hold_mini), max(backtest_maxi, buy_hold_maxi))

    plt.xlabel('Sharpe')
    plt.ylabel('Density')
    plt.legend()

    print("-" * 50)

if __name__ == "__main__":
    backtest()