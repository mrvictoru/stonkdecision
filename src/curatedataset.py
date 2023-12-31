# This script is to get stock data, create a gym environment from data, create agent to trade in the envrionment and collect trading data and save as dataset

# first group of functions are to get stock data

# import helper functions for getting stock data
from getstock import get_stock_data_yf_between_with_indicators
# import datatime library
from datetime import datetime, timedelta

from TradingEnvClass import StockTradingEnv

from get_agent import Agent, TradingAlgorithm

from StockTradingGraph import plot_stock_trading_data

import numpy as np
import json
import os
import re

# start_date need to be in format of 'YYYY-MM-DD'
def makegymenv(stock_name, start_date, period, interval='1d', indicators=['Volume', 'volume_cmf', 'trend_macd', 'momentum_rsi'], init_balance = 20000, render = 'None', random = False, normalize = False):
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
    env = StockTradingEnv(stock_data, init_balance, max_step, render_mode=render, random=random, normalize=normalize)
    env.reset()

    return env, env.observation_space.shape[0], env.action_space.shape[0], env.columns, stock_data

# second group of functions are to get agent, run it in the environment, collect trading data and save as json dataset
def run_env(agent, stock_name, env, num_episodes, date, normalize = False, deterministic=False):
    # data dictionary to store data
    data = {'data':[], 'stock': stock_name, 'num_episodes':num_episodes, 'normalize':normalize, 'env_state': env.columns, 'date':date}
    agent.reset()
    # loop through episodes
    for i in range (num_episodes):
        # dictionary to store state, ation, reward, timestep
        dict = {'state':[], 'action':[], 'reward':[], 'timestep':[]}
        # reset the environment
        state = env.reset()[0]
        dict['state'].append(state.tolist())
        timestep = 0
        reward = 0
        done = False
        # use normalized state if normalize is True
        if normalize:
            state = env.norm_obs()
        # loop to sample action, next_state, reward, from the env
        while not done:
            # sample action
            action, _states = agent.predict(state, timestep, reward, deterministic)
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

            # check if the episode is done
            if terminated or truncated:
                #print('Terminated: ', terminated, '; Truncated: ', truncated)
                #print('env current step ', env.current_step, ' env max step ', env.max_step)
                done = True
                print('Episode: ', i, 'Timestep:', timestep,  ' done')
            else:
                dict['state'].append(next_state.tolist())

            # update timestep and the state (use normalized state if normalize is True)
            timestep += 1
            if normalize:
                next_state = env.norm_obs()
            state = next_state

        # store the data for the episode
        data['data'].append(dict)
    
    return data

def save_data(data, file_name):

    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

def full_curate_run(json_file_path, agents_folder, num_episodes = 200, trade_range = [0.05, 0.3]):
    print("num_episodes: ", num_episodes)
    # read the JSON file
    with open(json_file_path, 'r') as f:
        config = json.load(f)
    
    # extract the configuration parameters
    stock_name = config['stock_name']
    start_date = config['start_date']
    num_days = config['num_days']
    interval = config['interval']
    indicators = config['indicators']
    init_balance = config['init_balance']
    output_path = config['output_path']

    # check if output_path folder exists
    if not os.path.exists(output_path):
        print("Creating output folder: ", output_path)
        os.makedirs(output_path)

    # create the trading environment
    print("Creating environment")
    env, obs_space_dim, act_space_dim, col, data = makegymenv(stock_name, start_date, num_days, interval, indicators=indicators, normalize=False, init_balance=init_balance)
    env_date = data.index.strftime('%Y-%m-%d').tolist()

    print("Getting stable agents")
    # get trained stable agents' path insider agents_folder_path
    path = os.path.join(os.getcwd(),agents_folder)
    # get agent type without .zip
    stable_agents_type = [re.sub('.zip', '', agent) for agent in os.listdir(path) if agent.endswith('.zip')]

    for agent_type in stable_agents_type:
        # make agent
        agent_path = os.path.join(path, agent_type+'.zip')
        agent = Agent(env, agent_type, agent_path)
        # run agent in env and collect data
        print("Running agent: ", agent_type)
        data = run_env(agent, stock_name, env, num_episodes=num_episodes, date=env_date, normalize=True, deterministic=False)
        # save data
        print("Saving data to ", output_path)
        filename = os.path.join(output_path, agent_type+'_'+stock_name+'_'+start_date+'.json')
        save_data(data, filename)

    # run random agent in env and collect data
    print("Running random agent")
    agent = Agent(env, 'random')
    data = run_env(agent, stock_name, env, num_episodes, env_date)
    # save data
    print("Saving data to ", output_path)
    filename = os.path.join(output_path, 'random_'+stock_name+'_'+start_date+'.json')
    save_data(data, filename)

    # run algo agent in env and collect data
    print("Running momentum algo agents")
    momentum_algo = 'momentum_stoch_rsi'
    # find the momentum_stoch_rsi column
    momentum_stoch_rsi_col = col.index('momentum_stoch_rsi')
    print("momentum_stoch_rsi_col: ", momentum_stoch_rsi_col)
    momentum_trade_algo = TradingAlgorithm(algo_type = momentum_algo, indicator_column = momentum_stoch_rsi_col, amount_range = trade_range)
    momentum_algo_agent = Agent(env, 'algo', algo = momentum_trade_algo)
    data = run_env(momentum_algo_agent, stock_name, env, num_episodes, env_date, normalize = False)
    # save data
    print("Saving data to ", output_path)
    filename = os.path.join(output_path, momentum_algo+'_'+stock_name+'_'+start_date+'.json')
    save_data(data, filename)

    print("Running trend sma fast algo agents")
    trend_sma_fast_algo = 'trend_sma_fast'
    # find the trend_sma_fast column
    trend_sma_fast_col = col.index('trend_sma_fast')
    print("trend_sma_fast_col: ", trend_sma_fast_col)
    trend_sma_fast_trade_algo = TradingAlgorithm(algo_type = trend_sma_fast_algo, indicator_column = trend_sma_fast_col, amount_range = trade_range)
    trend_sma_fast_algo_agent = Agent(env, 'algo', algo = trend_sma_fast_trade_algo)
    data = run_env(trend_sma_fast_algo_agent, stock_name, env, num_episodes, env_date, normalize = False)
    # save data
    print("Saving data to ", output_path)
    filename = os.path.join(output_path, trend_sma_fast_algo+'_'+stock_name+'_'+start_date+'.json')
    save_data(data, filename)

def evaluate_data(data_path):
    # read the json file from data_path
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # get the number of episodes
    num_episodes = data['num_episodes']
    col = data['env_state']

    # select 10 random episodes
    random_episodes = np.random.randint(0, num_episodes-1, 10)

    # calculate the mean and std of reward for the 10 random episodes
    rewards = []
    for episode in random_episodes:
        rewards.append(np.sum(data['data'][episode]['reward']))
    mean_sum_reward = np.mean(rewards)
    std_sum_reward = np.std(rewards)
    
    print("Mean simple sum reward: ", mean_sum_reward)
    print("Std simple sum reward: ", std_sum_reward)

    # calculate the mean and std of networth growth for the 10 random episodes
    end_networth = []
    net_worth_dix = col.index('Net_worth')
    for episode in random_episodes:
        end_networth.append(data['data'][episode]['state'][-1][net_worth_dix]-data['data'][episode]['state'][0][net_worth_dix])
    mean_end_networth = np.mean(end_networth)
    std_end_networth = np.std(end_networth)

    print("Mean net worth growth: ", mean_end_networth)
    print("Std net worth growth: +/-", std_end_networth)

    # plot one of the random episodes
    episode = random_episodes[2]
    print("Plotting episode: ", episode)
    state = np.array(data['data'][episode]['state'])
    action = np.array(data['data'][episode]['action'])
    date = data['date'][:state.shape[0]]
    plot_stock_trading_data(state, col, action, date)

