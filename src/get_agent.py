# this script is used to create or load agent.

from stable_baselines3 import PPO, A2C, DDPG
from cust_transf import DecisionTransformer
import numpy as np
import json
import torch

# create a class for the agent, which is used to store either the stable-baselines agent, random sampling action space agent, or else
class Agent:
    def __init__(self, env, agent_type, rtg_target=100, rtg_scale=0.75, model_path = None, algo = None, device = 'cpu', max_test_ep_len = 1000):
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
        elif agent_type == 'transformer':
            # load model parameters from the model_path json file
            with open(model_path, 'r') as f:
                params = json.load(f)
            
            state_dim = params['state_dim']
            act_dim = params['act_dim']
            n_block = params['n_blocks']
            h_dim = params['h_dim']
            self.context_len = params['context_len']
            n_heads = params['n_heads']
            drop_p = params['drop_p']
            model_dir = params['model_dir']

            self.agent = DecisionTransformer(state_dim, act_dim, n_block, h_dim, self.context_len, n_heads, drop_p).to(device)
            self.agent.load_state_dict(torch.load(model_dir))
            eval_batch_size = 1

            # zeros place holders
            self.actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim), dtype=torch.float32, device=device)
            self.states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim), dtype=torch.float32, device=device)
            self.rtg = torch.zeros((eval_batch_size, max_test_ep_len,1), dtype=torch.float32, device=device)
            
            # same as timesteps used for training the transformer
            self.timestep = torch.arange(start = 0, end = max_test_ep_len, step = 1)
            self.timestep = self.timestep.repeat(eval_batch_size, 1).to(device)

            self.device = device
            self.rtg_target = rtg_target
            self.rtg_scale = rtg_scale
            self.running_rtg = rtg_target/rtg_scale

        elif agent_type == 'algo':
            self.agent = algo
        elif agent_type == 'random':
            self.agent = None

    def reset(self):
        if isinstance(self.agent, TradingAlgorithm):
            self.agent.reset()
        elif isinstance(self.agent, DecisionTransformer):
            self.actions = torch.zeros_like(self.actions)
            self.states = torch.zeros_like(self.states)
            self.rtg = torch.zeros_like(self.rtg)
            self.running_rtg = self.rtg_target/self.rtg_scale
    
    def predict(self, state, t, running_award=0, deterministic=False):
        # if the agent is None, then return a random action
        if self.agent is None:
            return self.action_space.sample(), 0
        
        elif isinstance(self.agent, DecisionTransformer):
            self.agent.eval()
            device = self.device
            # add state in placeholder and normalize
            self.states[0,t] = torch.tensor(state).to(device)
            # calculate running rtg and add to placeholder
            self.running_rtg = self.running_rtg - (running_award/self.rtg_scale)
            self.rtg[0,t] = self.running_rtg

            if t < self.context_len:
                # run forward pass to get action
                _return_preds, state_preds, act_preds = self.agent.forward(self.states[:,:t+1], self.rtg[:,:t+1], self.timestep[:,:t+1], self.actions[:,:t+1])
                action_pred = act_preds[0,t].detach()
            else:
                # run forward pass to get action
                _return_preds, state_preds, act_preds = self.agent.forward(self.states[:,t-self.context_len+1:t+1], self.rtg[:,t-self.context_len+1:t+1], self.timestep[:,t-self.context_len+1:t+1],self.actions[:,t-self.context_len+1:t+1])
                action_pred = act_preds[0,-1].detach()
            # return the action as cpu numpy array
            self.actions[0,t] = action_pred
            return action_pred.cpu().numpy(), state_preds
        
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
        self.sold = False
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
                    self.sold = False
                    # return a random number between 0.2 and -0.2 as the confidence and action
                    return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
                else:
                    # Buy the stock if it is oversold 
                    self.bought = True
                    self.sold = False
                    # calculate return confidence and action
                    confidence = oversold_confidence(momentum_stoch_rsi)
                    return np.array([confidence, buy_action(confidence, self.amount_range[0], self.amount_range[1])])
                
            elif momentum_stoch_rsi > 0.8 and not self.sold:
                # Sell the stock if it is overbought
                self.sold = True
                self.bought = False
                # calculate return confidence and action
                confidence = overbought_confidence(momentum_stoch_rsi)
                return np.array([confidence, sell_action(confidence, self.amount_range[0], self.amount_range[1])])

            else:
                # Hold the current position
                self.bought = False
                self.sold = False
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
                        self.sold = False
                        # return a random number between 0.2 and -0.2 as the confidence and action
                        return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
                    else:
                        # Buy the stock if the ratio is below the mean - std
                        self.bought = True
                        self.sold = False
                        #print("trend_sma check buy")
                        # calculate return confidence and action
                        confidence = buy_trend_sma_fast_confidence(ratio, mean, std)
                        #print("trend_sma check confidence: ", confidence)
                        return np.array([confidence, buy_action(confidence, self.amount_range[0], self.amount_range[1])])
                    
                elif ratio > mean + std and not self.sold:
                    # Sell the stock if the ratio is above the mean + std
                    self.bought = False
                    self.sold = True
                    #print("trend_sma check sell")
                    # calculate return confidence and action
                    confidence = sell_trend_sma_fast_confidence(ratio, mean, std)
                    #print("trend_sma check confidence: ", confidence)
                    return np.array([confidence, sell_action(confidence, self.amount_range[0], self.amount_range[1])])

            # Hold the current position
            #print("trend_sma check hold")
            self.bought = False
            self.sold = False
            # return a random number between 0.2 and -0.2 as the confidence and action
            return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
        
        elif self.type == 'sentiment_react':
            # get the column index of the sentiment indicators (positive, negative, neutral)
            sentiment = [state[-3], state[-2], state[-1]]
            # come up with a strategy to trade based on the sentiment indicators
            # if the neutral sentiment is above 0.9, then hold the current position
            if sentiment[2] > 0.9:
                self.bought = False
                self.sold = False
                return np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(self.amount_range[0], self.amount_range[1])])
            # if the positive sentiment is above 0.8, then buy the stock
            elif sentiment[0] > 0.8:
                self.bought = True
                self.sold = False
                return np.array([np.random.uniform(0.7, 1), np.random.uniform(self.amount_range[0], self.amount_range[1])])
            # if the negative sentiment is above 0.8, then sell the stock
            elif sentiment[1] > 0.8:
                self.bought = False
                self.sold = True
                return np.array([np.random.uniform(-1, -0.7), np.random.uniform(self.amount_range[0], self.amount_range[1])])
            # else, hold the current position
            else:
                self.bought = False
                self.sold = False
                return np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(self.amount_range[0], self.amount_range[1])])


# the following helper functions are used to calculate the confidence of the action based on the momentum_stoch_rsi indicator        
def oversold_confidence(momentum_stoch_rsi):
    # map momentum_stoch_rsi [0 : 0.2] to [1 : 0.7]
    confidence = (momentum_stoch_rsi - 0) * (0.7 - 1) / (0.2 - 0) + 1
    # add a random value between -0.05 and 0.05 to the confidence
    confidence += np.random.uniform(-0.05, 0.05)
    # if the confidence is above 1, set it to 1
    if confidence > 1:
        confidence = 1
    return confidence

def overbought_confidence(momentum_stoch_rsi):
    # map momentum_stoch_rsi [0.8 : 1] to [-0.7 : -1]
    confidence = (momentum_stoch_rsi - 0.8) * (-1 - (-0.7)) / (1 - 0.8) + (-0.7)
    # add a random value between -0.05 and 0.05 to the confidence
    confidence += np.random.uniform(-0.05, 0.05)
    # if the confidence is below -1, set it to -1
    if confidence < -1:
        confidence = -1
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
