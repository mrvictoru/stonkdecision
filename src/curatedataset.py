# This script is to get stock data, create a gym environment from data, create agent to trade in the envrionment and collect trading data and save as dataset

# first group of functions are to get stock data

# import helper functions for getting stock data
from getstock import get_stock_data_yf_between_with_indicators
# import datatime library
from datetime import datetime, timedelta

from TradingEnvClass import StockTradingEnv

import numpy as np

import json

# start_date need to be in format of 'YYYY-MM-DD'
def makegymenv(stock_name, start_date, period, interval='1d', indicators=['Volume', 'volume_cmf', 'trend_macd', 'momentum_rsi', 'momentum_stoch_rsi'], init_balance = 20000, random = False):
    # work out the end_date from start_date and period
    try:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = start_date_obj + timedelta(days=period)
        end_date = end_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        
    stock_data = get_stock_data_yf_between_with_indicators(stock_name, start_date, end_date, interval, indicators)
    # check if momentum_stoch_rsi is in the indicators
    if 'momentum_stoch_rsi' in indicators:
        # if so then change the first 10 rows of momentum_stoch_rsi to 0.5
        stock_data['momentum_stoch_rsi'].iloc[:10] = 0.5

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
    env.reset()

    return env, env.observation_space.shape[0], env.action_space.shape[0], env.columns, stock_data

# second group of functions are to get agent, run it in the environment, collect trading data and save as json dataset
def run_env(agent, env, num_episodes, normalize = False):
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
        # use normalized state if normalize is True
        if normalize:
            state = env.norm_obs()
        # loop to sample action, next_state, reward, from the env
        while not done:
            # sample action
            action, _states = agent.predict(state, deterministic=False)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print('error in step')
                print('action: ', action)
                print(e)
                print('time step:', timestep)
                break
            dict['action'].append(action.tolist())
            dict['reward'].append(reward)
            dict['timestep'].append(timestep)
            # update timestep
            timestep += 1
            if normalize:
                next_state = env.norm_obs()
            state = next_state

            # check if the episode is done
            if terminated or truncated:
                done = True
                print('Episode: ', i, 'Timestep:', timestep,  ' done')
            else:
                dict['state'].append(state.tolist())

        # store the data for the episode
        data['data'].append(dict)
    
    return data

def save_data(data, file_name):

    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
