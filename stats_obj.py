import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sb
from datetime import date, timedelta
from scipy import stats
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LinearRegression
sb.set_theme()
np.set_printoptions(legacy='1.25')

class Stats():
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

    # - - -  Returns - - - 
        self.df['Cumulative Return %'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100

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

    def conditional_probability(self, print_count=False):
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
        if print_count == True:
            return count_df
        return round(prob_df * 100, 2)
    
    def run_algo(self):
        # - - - Run the Algorithm - - -
        # - - - Initialize post data and pre data sets - - -
        post_data = self.df[self.df.index.year >= date.today().year - 1]
        pre_data = self.df[self.df.index.year < date.today().year - 1]

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
        def get_col_label(value):
            if value > 8:
                return ">8%"
            elif value > 7:
                return ">7%"
            elif value > 6:
                return ">6%"
            elif value > 5:
                return ">5%"
            elif value > 4:
                return ">4%"
            elif value > 3:
                return ">3%"
            elif value > 2:
                return ">2%"
            elif value > 1:
                return ">1%"
            elif value > 0:
                return ">0%"
            else:
                return "0%"
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

        # store the trading actions from our while loop
        df_actions = pd.DataFrame({'Date':[],'Action':[],'Probability > 0%':[]})

        while len(post_data) > 0:
           # initialze the algo_df with the conditional probability of the pre_data
            algo_df = conditional_probability(pre_data)
            # grab first value and date from post_data
            start_value = post_data['Close'].iloc[0]
            start_date = post_data.index[0]
            # get the last return from pre_data
            last_return = pre_data['Return'].iloc[-1]
            # get the row label for the last return
            row_label = get_row_label(last_return)
            # return conditional probability the item will be positive given the action last period
            conditional_prob = algo_df.loc[row_label, 'Positive']
            # if conditional probability is greater than .50, buy, else sell
            if conditional_prob > .50:
                action = 'Buy'
            else:
                action = 'Sell'

            action_list = [start_date, action, conditional_prob]
            df_actions.loc[len(df_actions)] = action_list

            # extract the first row from post_data
            first_row = post_data.iloc[0]
            #append the first row to pre_data
            pre_data = pd.concat([pre_data, first_row.to_frame().T], ignore_index=True)
            # drop the first row from post_data
            post_data = post_data.iloc[1:]
        print(f"Buys/Sells {df_actions['Action'].value_counts()}")    # print the action taken
        result_df = self.df.join(df_actions.set_index('Date'), how='left')
        result_df['Probability > 0%'] = result_df['Probability > 0%'].ffill()
        result_df['Buy Signal'] = np.where(result_df['Action'] == 'Buy', result_df['Close'], np.nan)
        result_df['Sell Signal'] = np.where(result_df['Action'] == 'Sell', result_df['Close'], np.nan)
        return round(result_df, 4)
