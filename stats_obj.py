import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sb
from datetime import date, timedelta
from scipy import stats
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

        self.df["Previous Period Return"] = self.df["Return"].shift(1,fill_value=0)
        self.df.dropna(inplace=True)

    def normal_test(self):
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

    def normal_viz(self):
        # normal distribution plot
        
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

    def print_table(self):
        return self.df
    
    def linear_regression(self):
                # --- Graph setup ---
        x = self.df[["Previous Period Return"]]
        y = self.df["Return"]
        model = LinearRegression().fit(x,y)
        x_range = np.linspace(x.min(),x.max(),100)
        y_pred_line = model.predict(x_range)
        fig1 = plt.figure(figsize=(12, 6))
        sb.scatterplot(x="Previous Period Return", y="Return", data=self.df, color='Blue', label="Returns")
        plt.plot(x_range, y_pred_line, color='red', label="Regression Line")
        plt.xlabel("Previous Period Return (%)")
        plt.ylabel("Current Return (%)")
        plt.title("Linear Regression")
        plt.legend()
        plt.show()

    def conditional_probability(self, print_table=False, print_count=False):
        # - - - Conditional Probability Setup - - -

        # - - - Takes in a single df and returns the conditional probability of a stock moving a certain percentage given the previous period's return

        # -- initialize ranges and labels for binning
        ranges = [-np.inf, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, np.inf]
        labels = ["<-8%", "-8% to -7%", "-7% to -6%", "-6% to -5%", "-5% to -4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%", "0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", "4% to 5%", "5% to 6%", "6% to 7%", "7% to 8%", ">8%"]
        # -- create bins for previous and current returns
        self.df["Previous Bin"] = pd.cut(self.df['Previous Period Return'], bins=ranges, labels=labels)
        self.df["Current Bin"] = pd.cut(self.df['Return'], bins=ranges, labels=labels)

        # -- create probability and count dataframes, with labels as both index and columns
        prob_df = pd.DataFrame(index=labels, columns=labels)
        count_df = pd.DataFrame(index=labels, columns=labels)

        # -- calculate counts and probabilities
        # for each combination of previous and current bins, calculate the count and probability
        for previous_bin in labels:
            for current_bin in labels:
                # Count how many times the previous bin and current bin occur together
                count_both = len(self.df[(self.df["Previous Bin"] == previous_bin) & (self.df["Current Bin"] == current_bin)])
                # Count how many times only the previous bin occurs
                count_prev = len(self.df[self.df["Previous Bin"] == previous_bin])
                # Store the counts and probabilities in the respective dataframes
                # count is used to calculate the probability
                count_df.loc[previous_bin, current_bin] = count_both
                probability = count_both / count_prev if count_prev > 0 else 0
                prob_df.loc[previous_bin, current_bin] = probability
        count_df = count_df.astype(int)
        prob_df = prob_df.astype(float)

        #- - - Format Columns - - -
        negative_return = ["<-8%", "-8% to -7%", "-7% to -6%", "-6% to -5%", "-5% to -4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%"]
        positive_return = ["0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", "4% to 5%", "5% to 6%", "6% to 7%", "7% to 8%", ">8%"]
        prob_df["Negative"] = prob_df[negative_return].sum(axis=1)
        prob_df["Positive"] = prob_df[positive_return].sum(axis=1)
        count_df["Negative"] = count_df[negative_return].sum(axis=1)
        count_df["Positive"] = count_df[positive_return].sum(axis=1)

        prob_df[">1%"] = prob_df['1% to 2%'] + prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">2%"] = prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">3%"] = prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">4%"] = prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">5%"] = prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">6%"] = prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">7%"] = prob_df["7% to 8%"] + prob_df[">8%"]
        prob_df[">8%"] = prob_df[">8%"]
        # Calculate the Total Probability by summing coulmns
        #prob_df["Total"] = prob_df['Positive'] + prob_df['Negative']

        count_df[">1%"] = count_df['1% to 2%'] + count_df["2% to 3%"] + count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
        count_df[">2%"] = count_df["2% to 3%"] + count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
        count_df[">3%"] = count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
        count_df[">4%"] = count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
        count_df[">5%"] = count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
        count_df[">6%"] = count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
        count_df[">7%"] = count_df["7% to 8%"] + count_df[">8%"]
        count_df[">8%"] = count_df[">8%"]

        # Calculate the Total Count by summing coulmns
        count_df["Total"] = count_df['Positive'] + count_df['Negative']
        
        # Top 5 Probabilities
        print('-'*60)
        print('Relevant Probabilities having occurred more than 10 times')
        print('-'*60)
        print('Top 5 Probabilities')
        top_probs = prob_df.stack().nlargest(999999)
        best_count = 0
        for prob in top_probs.index:
            if best_count == 5:
                break
            # if the count has happened more than 10 times, print it
            if count_df.loc[prob[0], prob[1]] > 10:
                best_count += 1
                print(f'P({prob[1]} | {prob[0]}) = {round(prob_df.loc[prob[0], prob[1]] * 100, 4)}% --- occurred {count_df.loc[prob[0], prob[1]]} times')
        print('-'*60)

        # Top 5 Worst Probabilities excluding 0
        print('-'*60)
        print('Top 5 Worst Probabilities')
        worst_probs = prob_df.stack().nsmallest(999999)
        worst_count = 0
        for prob in worst_probs.index:
            if worst_count == 5:
                break
            if prob_df.loc[prob[0], prob[1]] > 0:
                # if the count has happened more than 10 times, print it
                if count_df.loc[prob[0], prob[1]] > 10:
                    worst_count += 1
                    print(f'P({prob[1]} | {prob[0]}) = {round(prob_df.loc[prob[0], prob[1]] * 100, 4)}% --- occurred {count_df.loc[prob[0], prob[1]]} times')
        
        # - - - State columns and rows - - -
        prob_df = pd.concat([prob_df], keys=[f'Current {self.interval} Return'], axis=1)
        prob_df = pd.concat([prob_df], keys=[f'Previous {self.interval} Return'], axis=0)

        # final rounding
        if print_table == True:
            if print_count == True:
                return count_df
            return round(prob_df * 100, 2)
    
    def run_algo(self, target_probability=.55, target_year=date.today().year - 1, print_table=False):
        # - - - Run the Algorithm - - -
        # - - - Initialize post data and pre data sets - - -
        post_data = self.df[self.df.index.year >= target_year]
        pre_data = self.df[self.df.index.year < target_year]

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
        def conditional_probability(df):
            # - - - Conditional Probability Setup - - -
            df = df.copy()
            # - - - Takes in a single df and returns the conditional probability of a stock moving a certain percentage given the previous period's return

            # -- initialize ranges and labels for binning
            ranges = [-np.inf, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, np.inf]
            labels = ["<-8%", "-8% to -7%", "-7% to -6%", "-6% to -5%", "-5% to -4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%", "0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", "4% to 5%", "5% to 6%", "6% to 7%", "7% to 8%", ">8%"]
            # -- create bins for previous and current returns
            df["Previous Bin"] = pd.cut(df['Previous Period Return'], bins=ranges, labels=labels)
            df["Current Bin"] = pd.cut(df['Return'], bins=ranges, labels=labels)

            # -- create probability and count dataframes, with labels as both index and columns
            prob_df = pd.DataFrame(index=labels, columns=labels)
            count_df = pd.DataFrame(index=labels, columns=labels)

            # -- calculate counts and probabilities
            # for each combination of previous and current bins, calculate the count and probability
            for previous_bin in labels:
                for current_bin in labels:
                    # Count how many times the previous bin and current bin occur together
                    count_both = len(df[(df["Previous Bin"] == previous_bin) & (df["Current Bin"] == current_bin)])
                    # Count how many times only the previous bin occurs
                    count_prev = len(df[df["Previous Bin"] == previous_bin])
                    # Store the counts and probabilities in the respective dataframes
                    # count is used to calculate the probability
                    count_df.loc[previous_bin, current_bin] = count_both
                    probability = count_both / count_prev if count_prev > 0 else 0
                    prob_df.loc[previous_bin, current_bin] = probability
            count_df = count_df.astype(int)
            prob_df = prob_df.astype(float)

            #- - - Format Columns - - -
            negative_return = ["<-8%", "-8% to -7%", "-7% to -6%", "-6% to -5%", "-5% to -4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%"]
            positive_return = ["0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", "4% to 5%", "5% to 6%", "6% to 7%", "7% to 8%", ">8%"]
            prob_df["Negative"] = prob_df[negative_return].sum(axis=1)
            prob_df["Positive"] = prob_df[positive_return].sum(axis=1)
            count_df["Negative"] = count_df[negative_return].sum(axis=1)
            count_df["Positive"] = count_df[positive_return].sum(axis=1)

            prob_df[">1%"] = prob_df['1% to 2%'] + prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">2%"] = prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">3%"] = prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">4%"] = prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">5%"] = prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">6%"] = prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">7%"] = prob_df["7% to 8%"] + prob_df[">8%"]
            prob_df[">8%"] = prob_df[">8%"]
            # Calculate the Total Probability by summing coulmns
            #prob_df["Total"] = prob_df['Positive'] + prob_df['Negative']

            count_df[">1%"] = count_df['1% to 2%'] + count_df["2% to 3%"] + count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
            count_df[">2%"] = count_df["2% to 3%"] + count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
            count_df[">3%"] = count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
            count_df[">4%"] = count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
            count_df[">5%"] = count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
            count_df[">6%"] = count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
            count_df[">7%"] = count_df["7% to 8%"] + count_df[">8%"]
            count_df[">8%"] = count_df[">8%"]

            # Calculate the Total Count by summing coulmns
            count_df["Total"] = count_df['Positive'] + count_df['Negative']

            return prob_df

        # Prepare lists to collect actions
        actions = []
        dates = []
        probs = []

        # Use integer indices instead of slicing DataFrames
        post_idx = self.df.index.get_loc(post_data.index[0])
        pre_idx = self.df.index.get_loc(pre_data.index[-1])

        prev_action = None
        while post_idx < len(self.df):
            # Use a rolling window or update less frequently for conditional_probability
            # Updating every step makes the code run longer, by making it update every 2 or 5 steps we improve performance
            if (post_idx - self.df.index.get_loc(post_data.index[0])) % 5 == 0: # <<<< Change this to 5 for less frequent updates
                current_pre_data = self.df.iloc[:post_idx]
                algo_df = conditional_probability(current_pre_data)
            
            start_date = self.df.index[post_idx]
            last_return = self.df['Return'].iloc[post_idx - 1]
            row_label = get_row_label(last_return)
            conditional_prob = algo_df.loc[row_label, 'Positive']

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
            
            dates.append(start_date)
            actions.append(action)
            probs.append(conditional_prob)
            prev_action = action

            post_idx += 1

        # Create the actions DataFrame once at the end
        df_actions = pd.DataFrame({'Date': dates, 'Action': actions, 'Probability > 0%': probs})

        
        print(f"Buys/Sells {df_actions['Action'].value_counts()}")    # print the action taken
        self.df = self.df.join(df_actions.set_index('Date'), how='left')
        self.df['Probability > 0%'] = self.df['Probability > 0%'].ffill()
        self.df['Buy Signal'] = np.where(self.df['Action'] == 'Buy', self.df['Close'], np.nan)
        self.df['Sell Signal'] = np.where(self.df['Action'] == 'Sell', self.df['Close'], np.nan)
        
       # remove rows with NaN in action
        self.df.dropna(subset=['Action'], inplace=True)

        if print_table:
            return round(self.df, 4)

    def visual(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.01, subplot_titles=("Candlesticks with Buy/Sell Signals", "Probability of Next Period Return"))
        # Candlestick
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], low=self.df['Low'], close=self.df['Close'], name='Candlestick'))
        # Buy and Sell Signals
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Buy Signal'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Sell Signal'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))

        # Add Conditional Probability
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Probability > 0%'], mode='lines', name='Probability > 0%', line=dict(color='purple', width=2)), row=2, col=1)

        #update layout
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')

        fig.add_hline(y=0.5, line=dict(color='red', dash='dash'), row=2, col=1)

        return fig
    
    def backtest(self, print_table=False):
        initial_investment = 10000
        cash = initial_investment
        position = 0
        portfolio_value = []

        # Calculate Buy/Hold Value
        share_cost = self.df['Close'].iloc[0]
        num_shares = initial_investment / share_cost
        self.df['Buy/Hold Value'] = num_shares * self.df['Close']

        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price = self.df['Close'].iloc[i]

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
        
        if print_table:
            return self.df
    
    def gen_comp(self):
        labels = pd.to_datetime(self.df.index).strftime('%Y-%m-%d')
        fig1= plt.figure(figsize=(12, 6))
        x_values = range(len(self.df))

        # add buy/hold to legend if it doesn't exist
        if f'{self.ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
            plt.plot(x_values, self.df['Buy/Hold Value'], label=f'{self.ticker} Buy/Hold')
        # model plot
        plt.plot(x_values, self.df['Portfolio Value'], label=f'{self.ticker} Model')

        # Set x-axis to date values and make it so they dont spawn too many labels
        plt.xticks(ticks=x_values, labels=labels, rotation=45)
        plt.locator_params(axis='x', nbins=10)

        # grid and legend
        plt.legend(loc=2)
        plt.grid(True, alpha=.5)
        # print cumulative return % if not already printed
        print(f"{self.ticker} Buy/Hold Result:", round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2))
        print(f"{self.ticker} Model Result:", round(((self.df['Portfolio Value'].iloc[-1] - self.df['Portfolio Value'].iloc[0])/self.df['Portfolio Value'].iloc[0]) * 100, 2))
        print(f" from {self.df.index[0]} to {self.df.index[-1]}")

