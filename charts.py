import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sb.set_theme()



# make charts generic


def comparison(object):
    labels = pd.to_datetime(object.df.index).strftime('%Y-%m-%d')
    fig1= plt.figure(figsize=(12, 6))
    x_values = range(len(object.df))

    # add buy/hold to legend if it doesn't exist
    if f'{object.ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
        plt.plot(x_values, object.df['Buy/Hold Value'], label=f'{object.ticker} Buy/Hold')
    # model plot
    plt.plot(x_values, object.df['Model Value'], label=f'{object.ticker} Model')

    # Set x-axis to date values and make it so they dont spawn too many labels
    plt.xticks(ticks=x_values, labels=labels, rotation=45)
    plt.locator_params(axis='x', nbins=10)

    # grid and legend
    plt.legend(loc=2)
    plt.grid(True, alpha=.5)
    
def linear_regression(object):
            # --- Graph setup ---
    x = object.df[["Previous Period Return"]]
    y = object.df["Return"]
    model = LinearRegression().fit(x,y)
    x_range = np.linspace(x.min(),x.max(),100)
    y_pred_line = model.predict(x_range)
    fig1 = plt.figure(figsize=(12, 6))
    sb.scatterplot(x="Previous Period Return", y="Return", data=object.df, color='Blue', label="Returns")
    plt.plot(x_range, y_pred_line, color='red', label="Regression Line")
    plt.xlabel("Previous Period Return (%)")
    plt.ylabel("Current Return (%)")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

def visual(object):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.01, subplot_titles=("Candlesticks with Buy/Sell Signals", "Probability of Next Period Return"))
    # Candlestick
    fig.add_trace(go.Candlestick(x=object.df.index, open=object.df['Open'], high=object.df['High'], low=object.df['Low'], close=object.df['Close'], name='Candlestick'))
    # Buy and Sell Signals
    fig.add_trace(go.Scatter(x=object.df.index, y=object.df['Buy Signal'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=object.df.index, y=object.df['Sell Signal'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))

    # Add Conditional Probability
    fig.add_trace(go.Scatter(x=object.df.index, y=object.df['Probability > 0%'], mode='lines', name='Probability > 0%', line=dict(color='purple', width=2)), row=2, col=1)

    #update layout
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')

    fig.add_hline(y=0.5, line=dict(color='red', dash='dash'), row=2, col=1)

    return fig

def normal(object):
    # normal distribution plot
    fig1 = plt.figure(figsize=(12, 6))
    plt.hist(object.percent_change, bins = 50, density = True)
    object.mini, object.maxi = plt.xlim()
    plt.plot(object.overlay, object.p, 'k')
    plt.axvline(object.mean, color='r', linestyle='dashed')
    
    # Standard Deviation Plots
    plt.axvline(object.mean + object.std, color='g', linestyle='dashed')
    plt.axvline(object.mean + (2 * object.std), color='b', linestyle='dashed')
    plt.axvline(object.mean - (2 * object.std), color='b', linestyle='dashed')
    plt.axvline(object.mean - object.std, color='g', linestyle='dashed')
    
    # labels
    plt.text(object.mean, plt.ylim()[1] * .9, 'mean', color='r', ha='center')
    plt.text(object.mean + object.std, plt.ylim()[1] * .8, '+1std', color='g', ha='center')
    plt.text(object.mean + (2 * object.std), plt.ylim()[1] * .7, '+2std', color='b', ha='center')
    plt.text(object.mean - (2 * object.std), plt.ylim()[1] * .7, '-2std', color='b', ha='center')
    plt.text(object.mean - object.std, plt.ylim()[1] * .8, '-1std', color='g', ha='center')
    plt.title(f"Mean: {round(object.mean, 2)}, Std: {round(object.std, 2)}")
    plt.xlabel('Percent Change')
    plt.ylabel('Density')