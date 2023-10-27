import yfinance as yf
import pandas as pd
import ta

# the following code is used to get stock data
def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

# helper function that get stock data from a start date to now
def get_stock_data_yf(stock_name, past_years, interval) -> pd.DataFrame:
    """
    Get stock data from yahoo finance

    Args:
        stock_name: str, the name of the stock to get data for
        past_years: int, the number of years of data to get
        interval:   str, the interval at which the data is collected. Valid values are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
                    see https://pypi.org/project/yfinance/ for more details
    
    Returns:
        pd.DataFrame, the stock data
    """
    data = yf.download(stock_name, period=f'{past_years}y', interval=interval)
    data = to_numeric_and_downcast_data(data)
    return data

# helper function that get stock data between two dates
def get_stock_data_yf_between(stock_name, start_date, end_date, interval) -> pd.DataFrame:
    """
    Get stock data from yahoo finance

    Args:
        stock_name: str, the name of the stock to get data for
        start_date: str, the start date of the data to get
        end_date:   str, the end date of the data to get
        interval:   str, the interval at which the data is collected. Valid values are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
                    see https://pypi.org/project/yfinance/ for more details
    
    Returns:
        pd.DataFrame, the stock data
    """
    data = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
    data = to_numeric_and_downcast_data(data)
    return data

# helper function that get stock data between two dates as well as the technical indicators
def get_stock_data_yf_between_with_indicators(stock_name, start_date, end_date, interval, indicators=['all']) -> pd.DataFrame:
    """
    Get stock data from yahoo finance

    Args:
        stock_name: str, the name of the stock to get data for
        start_date: str, the start date of the data to get
        end_date:   str, the end date of the data to get
        interval:   str, the interval at which the data is collected. Valid values are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
                    see https://pypi.org/project/yfinance/ for more details
        indicators: list, the list of technical indicators to add to the data
    
    Returns:
        pd.DataFrame, the stock data
    """
    data = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
    data = to_numeric_and_downcast_data(data)
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    # check if indicators is not all
    if indicators[0] != 'all':
        # remove columns that is not in the indicators
        for col in data.columns:
            if col not in indicators and col not in ['Open', 'High', 'Low', 'Close']:
                data = data.drop(col, axis=1)
    return data