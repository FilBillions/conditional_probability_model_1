import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
        plt.title(f'Normal Distribution of Model Results')
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
        plt.title(f'Normal Distribution of Backtest and Buy/Hold Results')
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
