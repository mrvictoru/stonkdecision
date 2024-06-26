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
MARKER_SIZE = 40

import mplfinance as mpf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StockTradingGraph:
    def __init__(self, df, dfvolume, action_history, net_worth_history, windows_size=20):
        # Save the dataframe with the stock price data (get only open, high, low, close from df)
        self.df = df[['Open', 'High', 'Low', 'Close']]
        if dfvolume is None:
            self.volume_plot = False
        else:
            self.volume_plot = True
            # get dfvolume as a column of df
            self.df.loc[:,'Volume'] = dfvolume
        # check if the net_worth_history is a numpy array
        if isinstance(net_worth_history, np.ndarray):
            self.net_worth = net_worth_history
        else:
            # if not, convert it to a numpy array
            self.net_worth = net_worth_history.to_numpy()
        # the first element of the action history is the buy or sell action
        if isinstance(action_history, np.ndarray):
            self.action_history = action_history
        else:
            # if not, convert it to a numpy array
            self.action_history = action_history.to_numpy()

        self.windows_size = windows_size


    def plot(self, current_step):
        # Get the data for the current window without the networth column
        
        start = max(current_step - self.windows_size, 0)
        end = current_step + 1

        data = self.df.iloc[start:end]
        if start != 0:
            networth = self.net_worth[start-1:end-1]        
            act_history = self.action_history[start-1:end-1]
        else:
            networth = self.net_worth[start:end]        
            act_history = self.action_history[start:end]
        print('check: ', type(act_history))
        # buy (1) or sell(-1) is store in the first element of the action history
        buy = np.array([(x[0] >= 2/3) for x in act_history])
        sell = np.array([(x[0] <= -2/3) for x in act_history])
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
        aplist.append(mpf.make_addplot(networth, type='line', ylabel='Net Worth ($)', panel=2))
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
                    returnfig=True, style='yahoo', datetime_format='%y-%m-%d', panel_ratios = (3,1,1), volume = self.volume_plot)

        # return the fig
        return fig

# write a function that will take data with stock price (Open, High, Low, Close) and action history (buy, sell, amount) then create a candlestick chart with buy and sell marker
def plot_stock_trading_data(state: np, col: list, action_history: np, date: list):
    # get the index of open, high, low, close and volume from matching the column name in col
    open_idx = col.index('Open')
    high_idx = col.index('High')
    low_idx = col.index('Low')
    close_idx = col.index('Close')

    net_worth_dix = col.index('Net_worth')

    Open = state[:,open_idx]
    High = state[:,high_idx]
    Low = state[:,low_idx]
    Close = state[:,close_idx]
  
    net_worth = state[:,net_worth_dix]
    windows_size = state.shape[0]

    dt_index = pd.to_datetime(date)

    # create a dataframe with the stock price data
    df = pd.DataFrame({'Open':Open, 'High':High, 'Low':Low, 'Close':Close}, index=dt_index)
    # create an instance of the StockTradingGraph class
    stock_graph = StockTradingGraph(df, None, action_history, net_worth, windows_size)

    fig = stock_graph.plot(windows_size)
    # show fig using matplotlib
    plt.show(fig)

    return fig