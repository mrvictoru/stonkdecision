# this script is used to create or load agent.

from stable_baselines3 import PPO, A2C, DDPG
import numpy as np


# create a class for the agent, which is used to store either the stable-baselines agent, random sampling action space agent, or else
class Agent:
    def __init__(self, env, agent_type, model_path = None, algo = None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        if agent_type.startswith('stable-baselines'):
            model_type = agent_type.split('-')[1]
            if model_type == 'ppo':
                self.agent = PPO.load(model_path)
            elif model_type == 'a2c':
                self.agent = A2C.load(model_path)
            elif model_type == 'ddpg':
                self.agent = DDPG.load(model_path)
        elif agent_type == 'algo':
            self.agent = algo
        elif agent_type == 'random':
            self.agent = None
    
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
    def __init__(self, algo_type = 'momentum_stoch_rsi', indicator_column = -1, amount_range = [0.2, 0.5]):
        self.indicator_column = indicator_column
        self.bought = False
        self.type = algo_type
        self.amount_range = amount_range


    def trade(self, state):
        # check if the algorithm is momentum_stoch_rsi

        if self.type == 'momentum_stoch_rsi':
            # load the momentum_stoch_rsi indicator from the observation
            momentum_stoch_rsi = state[self.indicator_column]

            # Determine the current position based on the momentum_stoch_rsi indicator
            if momentum_stoch_rsi < 0.2 and not self.bought:
                # Buy the stock if it is oversold
                self.bought = True
                # calculate return confidence and action
                return np.array([oversold_confidence(momentum_stoch_rsi), oversold_action(momentum_stoch_rsi, self.amount_range[0], self.amount_range[1])])
                
            elif momentum_stoch_rsi > 0.8 and self.bought:
                # Sell the stock if it is overbought
                self.bought = False
                # calculate return confidence and action
                return np.array([overbought_confidence(momentum_stoch_rsi), overbought_action(momentum_stoch_rsi, self.amount_range[0], self.amount_range[1])])

            else:
                # Hold the current position
                self.bought = False
                # return a random number between 0.2 and -0.2 as the confidence and action
                return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])

# the following helper functions are used to calculate the confidence of the action based on the momentum_stoch_rsi indicator        
def oversold_confidence(momentum_stoch_rsi):
    # map momentum_stoch_rsi [0 : 0.2] to [1 : 0.7]
    confidence = (momentum_stoch_rsi - 0) * (0.7 - 1) / (0.2 - 0) + 1
    return confidence

def oversold_action(momentum_stoch_rsi, lower_bound, higher_bound):
    # map momentum_stoch_rsi [0 : 0.2] to [higher_bound : lower_bound]
    action = (momentum_stoch_rsi - 0) * (lower_bound - higher_bound) / (0.2 - 0) + higher_bound
    return action

def overbought_confidence(momentum_stoch_rsi):
    # map momentum_stoch_rsi [0.8 : 1] to [-0.7 : -1]
    confidence = (momentum_stoch_rsi - 0.8) * (-1 - (-0.7)) / (1 - 0.8) + (-0.7)
    return confidence

def overbought_action(momentum_stoch_rsi, lower_bound, higher_bound):
    # map momentum_stoch_rsi [1 : 0.8] to [higher_bound : lower_bound]
    action = (momentum_stoch_rsi - 1) * (lower_bound - higher_bound) / (0.8 - 1) + higher_bound
    return action