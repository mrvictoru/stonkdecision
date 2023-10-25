"""
this trading environment class is able to take in a time series of stock data with abritrary number of features and create state space
the step function will take in an action (which is the number of shares to buy or sell) and output the next state, reward and done
the reset function will reset the environment to the initial state

see https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e for more details
"""

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'
VOLUME_CHART_HEIGHT = 0.33

# constant for reward function
# ALPHA is the reward for networth going up
ALPHA = 1
# BETA is the penalty for buying stock with high cost basis
BETA = 1
# GAMMA is the penalty for selling stock when there is no stock held
GAMMA = 100


LOOKBACK_WINDOW_SIZE = 30

# number of additonal features of the environment
ADD_FEATURES_NUM = 6


# import the necessary packages
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces, error, utils

import numpy as np
import pandas as pd

# helper function to get stock data
from getstock import *

# Candlestick graph class
from StockTradingGraph import StockTradingGraph
import matplotlib.backends.backend_agg as agg

# define the trading environment class
# This class defines a gym environment for simulating stock trading. The environment takes a pandas DataFrame of stock prices as input, 
# along with an initial account balance, a maximum number of steps, and a flag indicating whether to start at a random point in the data frame. 
# The environment provides an action space for buying, selling, or holding shares, and an observation space consisting of the current stock prices and additional features such as the current account balance and net worth. 
# The environment also provides a reward function based on the account balance and a delay modifier, which biases the reward towards having a higher balance towards the end of the episode. 
# The environment can be reset to an initial state, and can step forward in time by executing an action. 
# The environment provides a render function for displaying the current state of the environment, and a metadata attribute for specifying the available render modes.

# Example usage:
# import gym
# import pandas as pd
# from TradingEnvClass import StockTradingEnv

# load stock price data
# df = pd.read_csv('stock_prices.csv')

# create trading environment
# env = StockTradingEnv(df, init_balance=10000, max_step=1000, random=True)

# reset environment to initial state
# obs = env.reset()

# loop over steps
# for i in range(1000):
#     # choose random action
#     action = env.action_space.sample()
#     # step forward in time
#     obs, reward, done, info = env.step(action)
#     # render environment
#     env.render()
#     # check if episode is done
#     if done:
#         break

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'None']}
    visualization = None

    # add another parameter normalize to determine whether to normalize the observation when return
    def __init__(self, df, init_balance, max_step, render_mode = None, random=False, normalize=False):
        super(StockTradingEnv, self).__init__()
        self.render_mode = render_mode
        self.normalize = normalize
        # data
        # get all the features from df except for the column 'Volume'
        self.df = df.drop(columns=['Volume'])
        self.dfvolume = df['Volume']
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.init_balance = init_balance
        self.max_step = max_step
        self.random = random
        # turn the columns into a list
        self.columns = self.df.columns.tolist()
        print("init env with max step: ", self.max_step)

        self.net_worths = []

        # normalize the data
        self.price_mean = self.df['Close'].mean()
        self.price_std = self.df['Close'].std()
        self.df_standard = (self.df - self.df.mean()) / self.df.std()

        # trade action history
        self.action_history = []

        # action space (buy x%, sell x%, holdclass StockTradingEnv(gym.Env):

        self.action_space = spaces.Box(low=np.array([-1, 0.01]), high=np.array([1, 0.99]), dtype=np.float32)

        # observation space (prices and technical indicators)
        # shape should be (n_features + 6) where 6 is the number of additional dynamic features of the environment
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.df.columns) + ADD_FEATURES_NUM,), dtype=np.float32)

    # reset the state of the environment to an initial state
    def reset(self, seed = None):
        self.balance = self.init_balance
        self.net_worth = self.init_balance
        self.max_net_worth = self.init_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.action_history = []
        self.net_worths = []
        self.balance_history = []
        
        if self.random:
            # set the current step to a random point within the data frame
            if seed is not None:
                np.random.seed(seed)
            self.current_step = np.random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        else:
            self.current_step = 0
        
        if self.normalize:
            return self._next_observation_norm(), {}
        else:
            return self._next_observation(), {}

    def _next_observation(self):
        # get the features from the data frame for current time step
        frame = self.df.iloc[self.current_step].values

        # append additional features
        obs = np.append(frame, [
            self.balance,
            self.net_worth,
            self.shares_held,
            self.cost_basis,
            self.total_shares_sold,
            self.total_sales_value,
        ], axis=0)

        # update self.columns to include the additional features if it is not already included
        # check if the additional features are already included
        if len(self.columns) != len(obs):
            self.columns.extend(['Balance', 'Net_worth', 'Shares_held', 'Cost_basis', 'Total_shares_sold', 'Total_sales_value'])

        return obs.astype(np.float32)
    
    def _next_observation_norm(self):
        # get the features from the data frame for current time step
        frame = self.df_standard.iloc[self.current_step].values

        # normalize the additional data to avoid gradient issues.
        # # append additional features
        obs = np.append(frame, [
            self.balance/MAX_ACCOUNT_BALANCE,
            self.net_worth/MAX_ACCOUNT_BALANCE,
            self.shares_held/MAX_NUM_SHARES,
            (self.cost_basis - self.price_mean)/self.price_std,
            self.total_shares_sold/MAX_NUM_SHARES,
            self.total_sales_value/(MAX_NUM_SHARES *MAX_SHARE_PRICE),
        ], axis=0)

        # update self.columns to include the additional features if it is not already included
        # check if the additional features are already included
        if len(self.columns) != len(obs):
            self.columns.extend(['Balance', 'Net_worth', 'Shares_held', 'Cost_basis', 'Total_shares_sold', 'Total_sales_value'])

        return obs.astype(np.float32)

    def step(self,action):
        
        # Set the execute_price to the closing price of the time step
        execute_price = self.df.iloc[self.current_step]["Close"]
        # Execute one time step within the environment
        # an extra check is added to not execute inappropriate action such as buying when balance is not enough or selling when there is no stock held
        action_taken = self._take_action(action,execute_price)
        self.current_step += 1
        self.net_worths.append(self.net_worth)
        self.balance_history.append(self.balance)
        self.action_history.append(action_taken)

        # calculate reward based on the net worth/balance with a delay modifier. which bias towards having a higher balance towards the end of the episode
        # the modifier should be between 0.5 and 1, where toward the start of the episode it is closer to 0.5 and towards the end it is closer to 1
        delay_modifier = 0.5 + 0.5 * (self.current_step / self.max_step)
        # reward function reward networth going up and penalize buying stock with high cost basis as well as inappropriate action
        if len(self.net_worths) < 2:
            reward_costbasis = - self.cost_basis * BETA
            reward_inappropriate = - action_taken[2] * GAMMA
            reward = reward_costbasis + reward_inappropriate
        else:
            reward_networth = (self.net_worth - self.net_worths[-2])  * delay_modifier * ALPHA
            reward_balance = (self.balance - self.balance_history[-2]) * delay_modifier * ALPHA
            reward_costbasis = - self.cost_basis * BETA
            reward_inappropriate = - action_taken[2] * GAMMA
            reward = reward_networth + reward_balance + reward_costbasis + reward_inappropriate
        
        # if net_worth is below 0, or current_step is greater than max_step, then environment terminates
        truncated = (self.current_step >= self.max_step)
        terminated = bool(self.net_worth <= 0 or self.balance <= 0)

        if self.normalize:
            obs = self._next_observation_norm()
        else: 
            obs = self._next_observation()

        return obs, reward, terminated, truncated, {}
    
    def norm_obs(self):
        return self._next_observation_norm()
    
    def _take_action(self,action, execute_price):
        # Set the current price to a random price within the time step

        action_type = action[0]
        amount = action[1]
        # action taken has three elements, the first element is the action type, the second element is the amount of shares bought or sold, the third element being inappropriate action
        # inappropriate action is 1 if the action is to buy but the balance is not enough, or the action is to sell but the shares held is not enough
        action_taken = [0,0,0]

        # check if action_type between 2/3 and 1 then it is to buy
        if 2/3 <= action_type <= 1:
        
            # buy amount % of balance in shares
            total_possible = self.balance / execute_price
            # shares bought rounded to integer
            shares_bought = int(total_possible * amount)
            # if shares bought is 0 then make it one
            if shares_bought < 1:
                shares_bought = 1  

            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * execute_price
            
            if self.balance < additional_cost:
                shares_bought = 0
                additional_cost = 0

            self.balance -= additional_cost
            # calculate the new cost basis, check if it is divide by zero, if it is then set it to the execute price
            if self.shares_held + shares_bought == 0:
                self.cost_basis = execute_price
            else:
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            
            self.shares_held += shares_bought

            if shares_bought > 0:
                # change action taken to 1 to indicate buy and the amount of shares bought
                action_taken = [1, shares_bought, 0]
            else :
                action_taken = [0, shares_bought, 1]


        elif -1 <= action_type <= -2/3:
            # sell amount % of shares held (rounded to interger)
            shares_sold = int(self.shares_held * amount)
            # if shares sold is 0 then make it one unless we have no shares
            if shares_sold < 1 and self.shares_held > 0:
                shares_sold = 1
            else :
                shares_sold = 0
            self.balance += shares_sold * execute_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * execute_price
            # change action taken to -1 to indicate sell and the amount of shares sold
            if shares_sold > 0:
                action_taken = [-1, shares_sold, 0]
            else:
                action_taken = [0, shares_sold, 1]
            

        self.net_worth = self.balance + self.shares_held * execute_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # reset cost basis to 0 if no more shares held
        if self.shares_held == 0:
            self.cost_basis = 0
        
        return action_taken
          
    # see https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4        
    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - self.init_balance

        file = open(filename, 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n')
        file.write(f'Action: {self.action_history[-1]}\n')
        file.close()
    
    def _render_to_print(self):
        profit = self.net_worth - self.init_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        # print out the current stock price
        print(self.df.iloc[self.current_step])
    

    def render(self, **kwargs):
        mode = kwargs.get('mode', self.render_mode)
        if self.visualization == None:
            self.visualization = StockTradingGraph(self.df, self.dfvolume, self.action_history, self.net_worths, windows_size=LOOKBACK_WINDOW_SIZE)
        # Render the environment to the screen
        if mode == 'human':
            self._render_to_print()
        elif mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))
        elif mode == 'rgb_array':
            if self.current_step > LOOKBACK_WINDOW_SIZE:
                fig = self.visualization.plot(self.current_step)
                canvas = agg.FigureCanvasAgg(fig)
                canvas.draw()
                buf = canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4), self.current_step

        return None, None


    