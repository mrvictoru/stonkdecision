# This script is to get stock data, create a gym environment from data, create agent to trade in the envrionment and collect trading data and save as dataset

# first group of functions are to get stock data

# import helper functions for getting stock data
from getstock import get_stock_data_yf_between_with_indicators
# import datatime library
from datetime import datetime, timedelta

from TradingEnvClass import StockTradingEnv

import numpy as np

# start_date need to be in format of 'YYYY-MM-DD'
def makegymenv(stock_name, start_date, period, interval='1d', indicators=['Volume', 'volume_cmf', 'trend_macd', 'momentum_rsi'], init_balance = 20000, random = False):
    # work out the end_date from start_date and period
    try:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = start_date_obj + timedelta(days=period)
        end_date = end_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        
    stock_data = get_stock_data_yf_between_with_indicators(stock_name, start_date, end_date, interval, indicators)

    # loop through the data and check for any NaN values or inf values
    infnancheck = False
    for col in stock_data.columns:
        if stock_data[col].isnull().values.any() or np.isinf(stock_data[col]).values.any():
            print(f'NaN or inf value found in {col}')
            infnancheck = True
    if infnancheck:
        raise ValueError("NaN or inf value found in stock data")
    
    # create gym environment
    max_step = len(stock_data) - 1
    env = StockTradingEnv(stock_data, init_balance, max_step, random)

    return env

# second group of functions are to get agent, run it in the environment, collect trading data and save as json dataset
def run_env(agent, env, num_episodes, save_path, normalize = False):
    # data dictionary to store data
    data = {'data':[]}
    # loop through episodes
    for i in range (num_episodes):
        # dictionary to store state, ation, reward, timestep
        dict = {'state':[], 'action':[], 'reward':[], 'timestep':[]}
        # reset the environment
        state = env.reset()
        
        dict['state'].append(state.tolist())
        timestep = 0
        done = False
        # loop to sample action, next_state, reward, from the env
        while not done:
            # sample action
            if normalize:
                norm_state = env.norm_obs()
            action, _states = agent.predict(state, deterministic=False)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(e)
                print('time step:', timestep)
                break
            dict['action'].append(action.tolist())
            dict['reward'].append(reward)
            dict['timestep'].append(timestep)
            dict['state'].append(state.tolist())
            # update timestep
            timestep += 1

        

    
