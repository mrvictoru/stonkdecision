"""
This is a custom class use for animating stock trading data and agent action from an open ai gym environment.
There will be two subplots, one illustrates the net worth of the agent;
And the other illustrates the stock price along with the agent action.

It will utilize the new mplfinance library to plot the candlestick chart of the stock price.
Simply line chart will be used to plot the net worth of the agent.

matplotlib.animation will be used to animate the chart.

To use this class, create an instance of the class, pass the dataframe with the stock price data,
When animating, call the animate function with current_step, list of net_worth, list of trades action so far and windows_size
"""
MARKER_SIZE = 80

import mplfinance as mpf
import numpy as np
import pandas as pd


class StockTradingGraph:
    def __init__(self, df, dfvolume, action_history, net_worth_history, windows_size=20):
        # Save the dataframe with the stock price data (get only open, high, low, close from df)
        self.df = df[['Open', 'High', 'Low', 'Close']]
        # get dfvolume as a column of df
        self.df.loc[:,'Volume'] = dfvolume

        self.net_worth = net_worth_history
        # the first element of the action history is the buy or sell action
        self.action_history = action_history

        self.windows_size = windows_size


    def plot(self, current_step):
        # Get the data for the current window without the networth column
        
        start = max(current_step - self.windows_size, 0)
        end = current_step + 1

        data = self.df.iloc[start:end]
        networth = self.net_worth[start-1:end-1]        
        act_history = self.action_history[start-1:end-1]

        # buy (1) or sell(-1) is store in the first element of the action history
        buy = np.array([(x[0] == 1) for x in act_history])
        sell = np.array([(x[0] == -1) for x in act_history])
        amount = np.array([x[1] for x in act_history])

        colors = np.where(buy, 'g', np.where(sell, 'r', 'w'))
        amountap = mpf.make_addplot(amount, type='bar', panel=1, ylabel='Amount', color=colors)
        
        # check if buy and sell match the length of the data
        if len(buy) < len(data):
            # pad the buy and sell array with False to match the length of the data
            buy = np.pad(buy, pad_width=((0, len(data) - len(buy))), mode='constant', constant_values=False)
            sell = np.pad(sell, pad_width=((0, len(data) - len(sell))), mode='constant', constant_values=False)


        # create a new column for sell marker position (slightly above the high price when the action history indicates sell)
        buy_marker = data['Low'].where(buy)*0.995
        # create a new column for buy marker position (slightly below the low price when the action history indicates buy)
        sell_marker = data['High'].where(sell)*1.005

        aplist = []
        # add subplot for networth
        aplist.append(mpf.make_addplot(networth, type='line', ylabel='Net Worth ($)', panel=2, title='Net Worth'))
        aplist.append(amountap)
        # check if both buy_marker and sell_marker are not null
        if not(buy_marker.isnull().values.all()) and not(sell_marker.isnull().values.all()):

            # add buy marker to subplot
            buy_ap = mpf.make_addplot(buy_marker, type='scatter', marker='^', markersize=MARKER_SIZE, color='green', panel=0)
            # add sell marker to subplot
            sell_ap = mpf.make_addplot(sell_marker, type='scatter', marker='v', markersize=MARKER_SIZE, color='red', panel=0)

            aplist.append(buy_ap)
            aplist.append(sell_ap)
        
        # check if buy_marker is not null but sell_marker is null
        elif not(buy_marker.isnull().values.all()) and sell_marker.isnull().values.all():

            # add buy marker to subplot
            buy_ap = mpf.make_addplot(buy_marker, type='scatter', marker='^', markersize=MARKER_SIZE, color='green', panel=0)

            aplist.append(buy_ap)
        
        # check if sell_marker is not null but buy_marker is null
        elif not(sell_marker.isnull().values.all()) and buy_marker.isnull().values.all():

            # add sell marker to subplot
            sell_ap = mpf.make_addplot(sell_marker, type='scatter', marker='v', markersize=MARKER_SIZE, color='red', panel=0)

            aplist.append(sell_ap)

        
        fig, axlist = mpf.plot(data, type='candle', addplot=aplist, 
                    returnfig=True, style='yahoo', datetime_format='%y-%m-%d', panel_ratios = (3,1,1))

        # return the fig
        return fig

# write a function that will take data with stock price (Open, High, Low, Close) and action history (buy, sell, amount) then create a candlestick chart with buy and sell marker
def plot_stock_trading_data(state: np, col: list, action: np, windows_size=20):
    # get the index of open, high, low, close and volume from matching the column name in col
    open_idx = col.index('Open')
    high_idx = col.index('High')
    low_idx = col.index('Low')
    close_idx = col.index('Close')
    volume_idx = col.index('Volume')

    Open = state[:,open_idx]
    High = state[:,high_idx]
    Low = state[:,low_idx]
    Close = state[:,close_idx]
    Volume = state[:,volume_idx]

    # create a dataframe with the stock price data
    df = pd.DataFrame({'Open':Open, 'High':High, 'Low':Low, 'Close':Close})
    # create a dataframe with the volume data
    dfvolume = pd.DataFrame({'Volume':Volume})
    # create a dataframe with the action history
    action_history = pd.DataFrame(action)
    # create a dataframe with the net worth history
    net_worth_history = pd.DataFrame(net_worth)

    # create an instance of the StockTradingGraph class
    stock_graph = StockTradingGraph(df, dfvolume, action_history, net_worth_history, windows_size)

    # create a figure
    fig = plt.figure(figsize=(12,8))
    # create a grid spec
    gs = fig.add_gridspec(3, 1)
    # add a subplot for the candlestick chart
    ax1 = fig.add_subplot(gs[0:2, :])
    # add a subplot for the net worth
    ax2 = fig.add_subplot(gs[2, :])
    # plot the stock trading graph
    stock_graph.plot(len(df)-1)
    # show the plot
    plt.show()