import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sb
from datetime import date, timedelta
from scipy import stats
from scipy.stats import norm
sb.set_theme()
np.set_printoptions(legacy='1.25')

class Stats():
    def __init__(self, ticker, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d"):
    # Basic Constructors
        df = yf.download(ticker, start, end, interval = interval, multi_level_index=False)
        self.df = round(df, 2)
        self.ticker = ticker
        self.close = self.df['Close']
        self.percent_change = round(np.log(self.close).diff() * 100, 2)
        self.percent_change.dropna(inplace = True)
        self.df['Return'] = self.percent_change
    
    # - - - Descriptives - - - 
        n , minmax, mean, var, skew, kurt = stats.describe(self.percent_change)
        mini, maxi = minmax
        std = var ** .5
        self.random_sample = norm.rvs(mean, std, n)
        self.n = n
        self.mean = mean
        self.var = var
        self.skew = skew
        self.kurt = kurt
        self.mini = mini
        self.maxi = maxi
        self.std = std

    # - - - NORMAL CALCS - - -
        # overlay is your X value
        self.overlay = np.linspace(self.mini, self.maxi, 100)
        # p is simply your p value for normal calcs
        self.p = norm.pdf(self.overlay,self.mean,self.std)
        
    def is_normal(self):
    # descriptive statistics
        print(stats.describe(self.percent_change))
        random_test = stats.kurtosistest(self.random_sample)
        stock_test = stats.kurtosistest(self.percent_change)
        print('Null: The Sample is Normally Distributed')
        print('If P-Value < .05: Reject H0; If P-Value >= .05: Cannot Reject H0')
        print(f'{"-"*60}')
        print(f"Random Test: Statistic: {round(random_test[0], 2)}, P-Value: {round(random_test[1], 2)}")
        if random_test[1] >= .05:
            print('We cannot reject H0')
        else:
            print('We can reject H0')
        print(f'{"-"*60}')
        print(f"{self.ticker} Test: Statistic: {round(stock_test[0], 2)}, P-Value: {round(stock_test[1], 2)}")
        if stock_test[1] >= .05:
            print('We cannot reject H0, this is probably Normally Distributed')
        else:
            print('We can reject H0, this is probably not Normally Distributed')

    def visual(self):
        fig1 = plt.figure(figsize=(12, 6))
        plt.hist(self.percent_change, bins = 50, density = True)
        self.mini, self.maxi = plt.xlim()
        plt.plot(self.overlay, self.p, 'k')
        plt.axvline(self.mean, color='r', linestyle='dashed')
        
        # Standard Deviation Plots
        plt.axvline(self.mean + self.std, color='g', linestyle='dashed')
        plt.axvline(self.mean + (2 * self.std), color='b', linestyle='dashed')
        plt.axvline(self.mean - (2 * self.std), color='b', linestyle='dashed')
        plt.axvline(self.mean - self.std, color='g', linestyle='dashed')
        
        # labels
        plt.text(self.mean, plt.ylim()[1] * .9, 'mean', color='r', ha='center')
        plt.text(self.mean + self.std, plt.ylim()[1] * .8, '+1std', color='g', ha='center')
        plt.text(self.mean + (2 * self.std), plt.ylim()[1] * .7, '+2std', color='b', ha='center')
        plt.text(self.mean - (2 * self.std), plt.ylim()[1] * .7, '-2std', color='b', ha='center')
        plt.text(self.mean - self.std, plt.ylim()[1] * .8, '-1std', color='g', ha='center')
        plt.title(f"Mean: {round(self.mean, 2)}, Std: {round(self.std, 2)}")
        plt.xlabel('Percent Change')
        plt.ylabel('Density')

    def linear_regression(self):
        self.df['Offset 1 Day'] = self.df['Close'].shift(1)
        self.df['Offset 2 Day'] = self.df['Close'].shift(2)
        self.df.dropna(inplace=True)
        linear_regression_model = np.linalg.lstsq(self.df[['Offset 1 Day', 'Offset 2 Day']], self.df['Close'], rcond = None) [0]
        print(f'Linear Regression Model: {linear_regression_model}')
        self.df['Model Prediction'] = np.dot(self.df[['Offset 1 Day', 'Offset 2 Day']], linear_regression_model)
        fig1 = plt.figure(figsize=(12, 6));
        self.df.iloc[-252:][['Close', 'Model Prediction']].plot()

    def probability(self, threshold):
        if threshold == None:
            raise ValueError("No Threshold")
        if threshold <= 0:
            probability = 1 - (norm.sf(threshold, loc=self.mean, scale=self.std))
            print(f"Probability of {self.ticker} losing {threshold}% in one day is {round(probability*100,2):.2f}%")
        else:
            probability = norm.sf(threshold, loc=self.mean, scale=self.std)
            print(f"Probability of {self.ticker} gaining {threshold}% in one day is {round(probability*100,2):.2f}%")