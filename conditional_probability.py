import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sb
from datetime import date, timedelta
from scipy import stats
from scipy.stats import norm
from conditional_probability_func import conditional_probability
from charts import normal
sb.set_theme()
np.set_printoptions(legacy='1.25')

class Conditional_Probability():
    def __init__(self, ticker, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d"):
    # Basic Constructors
        df = yf.download(ticker, start, end, interval = interval, multi_level_index=False)
        self.df = df
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count
        self.ticker = ticker
        self.close = self.df['Close']
        self.percent_change = np.log(self.close).diff() * 100
        self.percent_change.dropna(inplace = True)
        self.df['Return'] = self.percent_change
        self.interval = interval
    
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

        # Get the previous close
        self.df['Prev Close'] = self.df['Close'].shift(1)

        self.df["Previous Period Return"] = self.df["Return"].shift(1,fill_value=0)
        self.df.dropna(inplace=True)

    def normal(self):
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
        return normal(self)

    def probability(self, threshold):
        # simple normal distribution probability calculation
        if threshold == None:
            raise ValueError("No Threshold")
        if threshold <= 0:
            probability = 1 - (norm.sf(threshold, loc=self.mean, scale=self.std))
            print(f"Probability of {self.ticker} losing {threshold}% in {self.interval} is {round(probability*100,2):.2f}%")
        else:
            probability = norm.sf(threshold, loc=self.mean, scale=self.std)
            print(f"Probability of {self.ticker} gaining {threshold}% in {self.interval} is {round(probability*100,2):.2f}%")
    
    def run_algo(self, target_probability=.55, start_date=date.today().year- 1, end_date=date.today(), print_table=False):
        # - - - Run the Algorithm - - -
        # - - - Initialize post data and pre data sets - - -
        if isinstance(start_date, int):
            post_data = self.df[self.df.index.year >= start_date]
            data_cutoff = []
        else:
            post_data = self.df[self.df.index >= start_date]
            if end_date == date.today():
                end_date = str(date.today())
            data_cutoff = self.df[self.df.index >= end_date]
        # import get row and column label function
        def get_row_label(value):
            if value < -8:
                return "<-8%"
            elif -8 <= value < -7:
                return "-8% to -7%"
            elif -7 <= value < -6:
                return "-7% to -6%"
            elif -6 <= value < -5:
                return "-6% to -5%"
            elif -5 <= value < -4:
                return "-5% to -4%"
            elif -4 <= value < -3:
                return "-4% to -3%"
            elif -3 <= value < -2:
                return "-3% to -2%"
            elif -2 <= value < -1:
                return "-2% to -1%"
            elif -1 <= value < 0:
                return "-1% to 0%"
            elif 0 <= value < 1:
                return "0% to 1%"
            elif 1 <= value < 2:
                return "1% to 2%"
            elif 2 <= value < 3:
                return "2% to 3%"
            elif 3 <= value < 4:
                return "3% to 4%"
            elif 4 <= value < 5:
                return "4% to 5%"
            elif 5 <= value < 6:
                return "5% to 6%"
            elif 6 <= value < 7:
                return "6% to 7%"
            elif 7 <= value < 8:
                return "7% to 8%"
            elif value >= 8:
                return ">8%"

        # Prepare lists to collect actions
        actions = []
        dates = []
        probs = []

        # Use integer indices instead of slicing DataFrames
        post_idx = self.df.index.get_loc(post_data.index[0])

        prev_action = None
        next_action = None  # This will hold the action for the next day
        action = 'No Action'  # <-- Ensure action is always initialized
        conditional_prob_prev = None  # Also initialize this
        while post_idx < (len(self.df) - len(data_cutoff)):
            if (post_idx - self.df.index.get_loc(post_data.index[0])) % 5 == 0:
                current_pre_data = self.df.iloc[:post_idx]
                algo_df = conditional_probability(current_pre_data,print_statement=False)
            
            start_date = self.df.index[post_idx]
            last_return = self.df['Return'].iloc[post_idx - 1]
            row_label = get_row_label(last_return)
            conditional_prob = algo_df.loc[row_label, 'Positive']

            # Decide action for the NEXT day
            if conditional_prob > target_probability:
                if prev_action == 'Buy' or prev_action == 'Hold':
                    action = 'Hold'
                else:
                    action = 'Buy'
            elif conditional_prob < target_probability:
                if prev_action == 'Hold' or prev_action == 'Buy':
                    action = 'Sell'
                if prev_action == 'No Action' or prev_action == 'Sell':
                    action = 'No Action'
            else:
                action = 'No Action'

            # Only append the action for the previous day (to avoid lookahead bias)
            if next_action is not None:
                dates.append(start_date)
                actions.append(next_action)
                probs.append(conditional_prob_prev)  # Save the previous day's probability

            # Prepare for next iteration
            prev_action = action
            next_action = action
            conditional_prob_prev = conditional_prob

            post_idx += 1

        # Create the actions DataFrame once at the end
        df_actions = pd.DataFrame({'Date': dates, 'Action': actions, 'Probability > 0%': probs})

        
        print(f"Buys/Sells {df_actions['Action'].value_counts()}")    # print the action taken
        self.df = self.df.join(df_actions.set_index('Date'), how='left')
        self.df['Probability > 0%'] = self.df['Probability > 0%'].ffill()
        # Previous close at Buy signals
        self.df['Buy Signal'] = np.where(self.df['Action'] == 'Buy', self.df['Prev Close'], np.nan)
        # Previous close at Sell signals
        self.df['Sell Signal'] = np.where(self.df['Action'] == 'Sell', self.df['Prev Close'], np.nan)
        
       # remove rows with NaN in action
        self.df.dropna(subset=['Action'], inplace=True)

        if print_table:
            return round(self.df, 4)
    
    def backtest(self, print_table=False):
        initial_investment = 10000
        cash = initial_investment
        position = 0
        portfolio_value = []

        # Calculate Buy/Hold Value
        share_cost = self.df['Prev Close'].iloc[0]
        num_shares = initial_investment / share_cost
        self.df['Buy/Hold Value'] = num_shares * self.df['Close']

        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price = self.df['Prev Close'].iloc[i]

            if action == 'Buy' and cash > 0:
                position = cash/price
                cash = 0
            elif action == 'Sell' and position > 0:
                cash = position * price
                position = 0
            elif action == 'Hold':
                pass

            portfolio_value.append(cash + (position * price))
        self.df['Portfolio Value'] = portfolio_value
                #dropping unnecessary columns
        if 'Volume' in self.df.columns:
                self.df.drop(columns=['Volume'], inplace = True)
        if 'Previous Bin' in self.df.columns:
                self.df.drop(columns=['Previous Bin'], inplace = True)
        if 'Current Bin' in self.df.columns:
                self.df.drop(columns=['Current Bin'], inplace = True)
        
        print(f"{self.ticker} Buy/Hold Result: {round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)}%")
        print(f"{self.ticker} Model Result: {round(((self.df['Portfolio Value'].iloc[-1] - self.df['Portfolio Value'].iloc[0])/self.df['Portfolio Value'].iloc[0]) * 100, 2)}%")
        print(f" from {self.df.index[0]} to {self.df.index[-1]}")
        if print_table:
            return self.df

