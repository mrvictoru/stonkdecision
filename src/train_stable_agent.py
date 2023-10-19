# train an agent using stable baselines

import torch
# set detect anomaly to true
torch.autograd.set_detect_anomaly(True)
import numpy

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# import custom functions and classes
from curatedataset import makegymenv
from TradingEnvClass import StockTradingEnv

def create_stable_agents(env, num_cpu = 6):
    print("Vectorizing environment")
    env_vec = SubprocVecEnv([lambda: env for i in range(num_cpu)])
    print("Creating PPO agent")
    modelPPO = PPO("MlpPolicy", env_vec, verbose=1)
    print("Creating A2C agent")
    modelA2C = A2C("MlpPolicy", env_vec, verbose=1)
    print("Creating DDPG agent")
    modelDDPG = DDPG("MlpPolicy", env, verbose=1)
    # store the models' name in a list
    return [modelPPO, modelA2C, modelDDPG]

def evaluate_stable_agent(model, env, n_eval_ep = 10, deterministic=False):
    # evaluate the agent
    print("Evaluating model: ", model)
    mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=n_eval_ep, deterministic=deterministic)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def train_stable_agent(model, num_timesteps = 10000):
    # train the agent
    print("Training model: ", model)
    model.learn(total_timesteps=num_timesteps)
    print("Training complete")
    return model

def output_stable_agent(model, path):
    # save the model
    model.save(path)
    print("Model saved")



