import datetime as dt
import polars as pl
import requests
from bs4 import BeautifulSoup
import os
import json
import re

def get_nasdaq_tickers():
    url = 'https://api.nasdaq.com/api/quote/list-type/nasdaq100'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    }
    res = requests.get(url, headers=headers)
    main_data = res.json()['data']['data']['rows']
    tickers = [data['symbol'] for data in main_data]
    return tickers

def get_dow_tickers():
  url = "https://www.cnbc.com/dow-components/"
  # send a GET request to the url
  response = requests.get(url)
  # check if the response is successful
  if response.status_code == 200:
    # parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    # find all the elements that contain the ticker symbols
    elements = soup.find_all("a", class_="quote")
    # create an empty list to store the tickers
    tickers = []
    # loop through the elements
    for element in elements:
      # get the text of the element
      text = element.get_text()
      # use a regular expression to extract the ticker symbol
      match = re.search(r"\((\w+)\)", text)
      # if a match is found, append the ticker to the list
      if match:
        ticker = match.group(1)
        tickers.append(ticker)
    # return the list of tickers
    return tickers
  else:
    # return an empty list if the response is not successful
    return []

def get_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)
    return tickers




"""
the following function will create json files in the format below for each stock in the list of tickers
{
    "stock_name": "TSLA",
    "start_date": "2018-01-01",
    "num_days": 500,
    "interval": "1d",
    "indicators": ["Volume", "volume_cmf", "trend_macd", "momentum_rsi", "momentum_stoch_rsi", "trend_sma_fast"],
    "init_balance": 20000,
    "output_path": "trained_stable_agents/"
}
"""

def create_json_files(tickers, start_date, num_days, interval, indicators, init_balance, output_path, json_path):
    for ticker in tickers:
        # change output path to ticker name with output path
        new_output_path = os.path.join(output_path, ticker)
        json_file = {
            "stock_name": ticker,
            "start_date": start_date,
            "num_days": num_days,
            "interval": interval,
            "indicators": indicators,
            "init_balance": init_balance,
            "output_path": new_output_path
        }
        # Check if the directory exists
        if not os.path.exists(json_path):
            # If not, create the directory
            os.makedirs(json_path)

        # Now you can safely write your file
        with open(os.path.join(json_path, f'training_config_{ticker}.json'), 'w') as f:
            json.dump(json_file, f)