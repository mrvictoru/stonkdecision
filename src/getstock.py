import yfinance as yf
import pandas as pd
import ta
import requests
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt


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
    if not(indicators[0] == 'all'):
        # remove columns that is not in the indicators
        for col in data.columns:
            if col not in indicators and col not in ['Open', 'High', 'Low', 'Close']:
                data = data.drop(col, axis=1)
    return data

def get_api_key():
    # read json file for api key
    with open('api_key.json') as f:
        data = json.load(f)
    return data['api_key'], data['secret_key'], data['base_url']

def has_dollar_symbol(lst:list):
    for element in lst:
        if "$" in element or "USD" in element:
            return True
    return False
  

def get_newsheadline_sentiment(stock_name:str, start_date:dt.datetime, end_date:dt.datetime, device, tokenizer, model):

    # get api key
    api_key, secret_key, base_url = get_api_key()

    # convert start_date and end_date to string YYYY-MM-DD
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    url = f"https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&sort=desc&symbols={stock_name}&exclude_contentless=true"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    
    # use requests to get news from alpaca api

    response = requests.get(url, headers=headers)
    # check if response is successful
    if response.status_code != 200:
        print("Error: ", response.status_code)
        return None
    else:
        newslist = response.json()['news']

    news = [lst for lst in newslist if not has_dollar_symbol(lst['symbols'])]
    
    news= [ev["summary"] for ev in news]
    tokens = tokenizer(news, padding = True, return_tensors="pt").to(device)
    result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
    result = torch.nn.functional.softmax(torch.sum(result, 0), dim = -1)
    # turn the result into a list
    result = result.tolist()
    return result


# helper function that get stock data between two dates as well as the technical indicators and news sentiment
def get_stock_data_yf_between_with_indicators_news(stock_name, start_date, end_date, interval, indicators=['all']) -> pd.DataFrame:
    print(f"Getting stock data of {stock_name}...")
    data = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
    data = to_numeric_and_downcast_data(data)
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    # check if indicators is not all
    if not(indicators[0] == 'all'):
        print("sorting indicators...")
        # remove columns that is not in the indicators
        for col in data.columns:
            if col not in indicators and col not in ['Open', 'High', 'Low', 'Close']:
                data = data.drop(col, axis=1)

    print("setting up sentiment analysis model...")
    # set up tokenizer and model for sentiment analysis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    
    # create 3 new columns (positive, negative ,neutral) for news sentiment and probability
    data['positive'] = 0.0
    data['negative'] = 0.0
    data['neutral'] = 0.0
    
    print(f"Getting news sentiment of {stock_name}...")
    # loop through each timestep and get the news sentiment between that day and previous 4 days
    for i in range(len(data)):
        # get the news sentiment
        try:
            result = get_newsheadline_sentiment(stock_name, data.index[i-4], data.index[i], device, tokenizer, model)
        except Exception as e:
            print("Error: ", e)
            result = None
        if result is None:
            result = [0.0,0.0,0.0]
        # update the news sentiment for that day
        data.loc[data.index[i], 'positive'] = float(result[0])
        data.loc[data.index[i], 'negative'] = float(result[1])
        data.loc[data.index[i], 'neutral'] = float(result[2])
    print(f"Getting stock data of {stock_name} completed.")
    return data
