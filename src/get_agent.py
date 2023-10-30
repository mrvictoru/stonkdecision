# this script is used to create or load agent.

from stable_baselines3 import PPO, A2C, DDPG
import numpy as np


# create a class for the agent, which is used to store either the stable-baselines agent, random sampling action space agent, or else
class Agent:
    def __init__(self, env, agent_type, model_path = None, algo = None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.agent = None
        if agent_type.startswith('stable-baselines'):
            model_type = agent_type.split('-')[1]
            if model_type == 'ppo':
                self.agent = PPO.load(model_path, env=env)
            elif model_type == 'a2c':
                self.agent = A2C.load(model_path, env=env)
            elif model_type == 'ddpg':
                self.agent = DDPG.load(model_path, env=env)
        elif agent_type == 'algo':
            self.agent = algo
        elif agent_type == 'random':
            self.agent = None

    def reset(self):
        if isinstance(self.agent, TradingAlgorithm):
            self.agent.reset()
    
    def predict(self, state, deterministic=False):
        # if the agent is None, then return a random action
        if self.agent is None:
            return self.action_space.sample(), 0
        # if the agent is a TradingAlgorith, then return the action from the algorithm
        elif isinstance(self.agent, TradingAlgorithm):
            return self.agent.trade(state), 0
        # else, return the action from the stable-baselines agent
        else:
            return self.agent.predict(state, deterministic=deterministic)


class TradingAlgorithm:
    def __init__(self, algo_type = 'momentum_stoch_rsi', indicator_column = -1, amount_range = [0.05, 0.3],window_size = 20):
        self.indicator_column = indicator_column
        self.bought = False
        self.memory1 = []
        self.type = algo_type
        self.amount_range = amount_range
        self.window_size = window_size
    
    def reset(self):
        self.bought = False
        self.memory1 = []
    
    def trade(self, state):
        # check if the algorithm is momentum_stoch_rsi
        # if so implement trading using momentum_stoch_rsi indicator
        if self.type == 'momentum_stoch_rsi':
            # load the momentum_stoch_rsi indicator from the observation
            momentum_stoch_rsi = state[self.indicator_column]
            
            # Determine the current position based on the momentum_stoch_rsi indicator
            if momentum_stoch_rsi < 0.2 and not self.bought:
                # check if we can afford to buy by checking if close price is above the balance,
                if state[3] > state[-6]:
                    # hold the current position
                    self.bought = False
                    # return a random number between 0.2 and -0.2 as the confidence and action
                    return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
                else:
                    # Buy the stock if it is oversold 
                    self.bought = True
                    # calculate return confidence and action
                    confidence = oversold_confidence(momentum_stoch_rsi)
                    return np.array([confidence, buy_action(confidence, self.amount_range[0], self.amount_range[1])])
                
            elif momentum_stoch_rsi > 0.8 and self.bought:
                # Sell the stock if it is overbought
                self.bought = False
                # calculate return confidence and action
                confidence = overbought_confidence(momentum_stoch_rsi)
                return np.array([confidence, sell_action(confidence, self.amount_range[0], self.amount_range[1])])

            else:
                # Hold the current position
                self.bought = False
                # return a random number between 0.2 and -0.2 as the confidence and action
                return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
            
        elif self.type == 'trend_sma_fast':
            #print("trend_sma_fast check")
            # load the trend_sma_fast indicator from the observation
            trend_sma_fast = state[self.indicator_column]
            # calculate the ratio of trend_sma_fast to the current close price (state[3])
            ratio = state[3]/trend_sma_fast
            # add the current ratio to the memory
            self.memory1.append(ratio)
            #print("trend_sma_fast check trend_sma_fast: ", trend_sma_fast)
            #print("trend_sma_fast check ratio: ", ratio)
            #print("trend_sma_fast check memory len: ", len(self.memory1))

            # check if we have enough data points in memory
            if len(self.memory1) >= self.window_size:
                # calculate the mean and std of ratio over the window_size
                mean = np.mean(self.memory1)
                std = np.std(self.memory1)
                # debug print
                #print("trend_sma check mean: ", mean, "std: ", std)
                # Determine the current position based on the mean and std of ratio
                if ratio < mean - std and not self.bought:
                    # check if we can afford to buy by checking if close price is below the balance,
                    if state[3] > state[-6]:
                        # hold the current position
                        self.bought = False
                        # return a random number between 0.2 and -0.2 as the confidence and action
                        return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
                    else:
                        # Buy the stock if the ratio is below the mean - std
                        self.bought = True
                        #print("trend_sma check buy")
                        # calculate return confidence and action
                        confidence = buy_trend_sma_fast_confidence(ratio, mean, std)
                        #print("trend_sma check confidence: ", confidence)
                        return np.array([confidence, buy_action(confidence, self.amount_range[0], self.amount_range[1])])
                    
                elif ratio > mean + std and self.bought:
                    # Sell the stock if the ratio is above the mean + std
                    self.bought = False
                    #print("trend_sma check sell")
                    # calculate return confidence and action
                    confidence = sell_trend_sma_fast_confidence(ratio, mean, std)
                    #print("trend_sma check confidence: ", confidence)
                    return np.array([confidence, sell_action(confidence, self.amount_range[0], self.amount_range[1])])

            # Hold the current position
            #print("trend_sma check hold")
            self.bought = False
            # return a random number between 0.2 and -0.2 as the confidence and action
            return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])



# the following helper functions are used to calculate the confidence of the action based on the momentum_stoch_rsi indicator        
def oversold_confidence(momentum_stoch_rsi):
    # map momentum_stoch_rsi [0 : 0.2] to [1 : 0.7]
    confidence = (momentum_stoch_rsi - 0) * (0.7 - 1) / (0.2 - 0) + 1
    # add a random value between -0.05 and 0.05 to the confidence
    confidence += np.random.uniform(-0.05, 0.05)
    return confidence

def overbought_confidence(momentum_stoch_rsi):
    # map momentum_stoch_rsi [0.8 : 1] to [-0.7 : -1]
    confidence = (momentum_stoch_rsi - 0.8) * (-1 - (-0.7)) / (1 - 0.8) + (-0.7)
    # add a random value between -0.05 and 0.05 to the confidence
    confidence += np.random.uniform(-0.05, 0.05)
    return confidence

def buy_trend_sma_fast_confidence(ratio, mean, std):
    # map ratio below the mean - 2*std to 1
    if ratio <= mean - 2*std:
        confidence = 1
    # map ratio between mean - 2*std and mean - std to [1:0.7]
    elif ratio <= mean - std:
        confidence = (ratio - (mean - std))*(1-0.7) / ((mean - 2*std) - (mean - std)) + 0.7
    # map ratio above the mean to 0.7
    else:
        confidence = 0.7
    # add a random value between -0.05 and 0.05 to the confidence
    confidence += np.random.uniform(-0.05, 0.05)
    return confidence

def sell_trend_sma_fast_confidence(ratio, mean, std):
    # map ratio below the mean + 2*std to -1
    if ratio >= mean + 2*std:
        confidence = -1
    # map ratio between mean + 2*std and mean + std to [-1:-0.7]
    elif ratio >= mean + std:
        confidence = (ratio - (mean + std)) * (-1 + 0.7) / ((mean + 2*std) - (mean + std)) - 0.7
    # map ratio above the mean to 0.7
    else:
        confidence = -0.7
    # add a random value between -0.05 and 0.05 to the confidence
    confidence += np.random.uniform(-0.05, 0.05)
    return confidence

def buy_action(confidence, lower_bound, higher_bound):
    # map confidence [1 : 0.7] to [higher_bound : lower_bound]
    action = (confidence - 1) * (lower_bound - higher_bound) / (0.7 - 1) + higher_bound
    # generate an array of values with a step size of 0.1
    values = np.arange(-0.02, 0.02, 0.005)
    # randomly select a value from the array
    action += np.random.choice(values)
    return action

def sell_action(confidence, lower_bound, higher_bound):
    # map momentum_stoch_rsi [-1 : -0.7] to [higher_bound : lower_bound]
    action = (confidence - (-1)) * (lower_bound - higher_bound) / ((-0.7) - (-1)) + higher_bound
    # generate an array of values with a step size of 0.1
    values = np.arange(-0.02, 0.02, 0.005)
    # randomly select a value from the array
    action += np.random.choice(values)
    return action
