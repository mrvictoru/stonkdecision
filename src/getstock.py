import yfinance as yf
import pandas as pd
import ta
from alpaca_trade_api.rest import REST
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


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

def test():
    api_key, secret_key, base_url = get_api_key()
    api = REST(
        api_key,
        secret_key,
        base_url
    )
    testdf = api.get_quotes("AAPL", "2021-06-08", "2021-06-08", limit=10).df
    print(testdf)
    

def get_newsheadline_sentiment(stock_name, start_date, end_date, device, tokenizer, model):

    api_key, secret_key, base_url = get_api_key()
    # get news using alpaca api
    api = REST(
        api_key,
        secret_key,
        base_url
    )
    news = api.get_news(symbol = stock_name, start = start_date, end = end_date)
    print("type: ", type(news))

    # get the headline of the news
    news= [ev.__dict__["_raw"]["headline"] for ev in news]
    print("length: ", len(news))
    tokens = tokenizer(news, padding = True, return_tensors="pt").to(device)
    result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
    result = torch.nn.functional.softmax(torch.sum(result, 0), dim = -1)
    #probability = result[torch.argmax(result)]
    #sentiment = torch.argmax(result)
    return result


# helper function that get stock data between two dates as well as the technical indicators and news sentiment
def get_stock_data_yf_between_with_indicators_news(stock_name, start_date, end_date, interval, indicators=['all']) -> pd.DataFrame:
    data = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
    data = to_numeric_and_downcast_data(data)
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    # check if indicators is not all
    if not(indicators[0] == 'all'):
        # remove columns that is not in the indicators
        for col in data.columns:
            if col not in indicators and col not in ['Open', 'High', 'Low', 'Close']:
                data = data.drop(col, axis=1)

    # set up tokenizer and model for sentiment analysis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

    # create new columns for news sentiment and probability