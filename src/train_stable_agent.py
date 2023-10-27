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

from stable_baselines3.common.env_checker import check_env

# import custom functions and classes
from curatedataset import makegymenv
from TradingEnvClass import StockTradingEnv

import argparse
import json
import re
import os

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

# write a main function that will take a json file with stock_name, start_date, num_days, interval, init_balance as arguments
# it will train the agent and save it in a folder with the stock_name
def full_run(json_file_path):
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
    stable_env, obs_space, act_space, col, data = makegymenv(stock_name, start_date, num_days, interval, indicators=indicators, normalize=True, init_balance=init_balance)

    try:
        check_env(stable_env)
    except Exception as e:
        print("Failed stable_baselines3 checking")
        print(e)
        return None
    
    model_list = create_stable_agents(stable_env)

    print("Evaluate pre training model")
    for model in model_list:
        _,_ = evaluate_stable_agent(model, stable_env, 5)
    
    for model in model_list:
        trained_model = train_stable_agent(model, len(data)*200)
        _,_ = evaluate_stable_agent(trained_model, stable_env, 5)
        match = re.search(r"\.(\w+)\.", str(model))
        if match:
            name = match.group(1)
        # join path and name
        path = os.path.join(output_path, name)
        output_stable_agent(trained_model,path)


def main():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train and save a stable agent for stock trading')

    # add an argument for the file path
    parser.add_argument('file_path', type=str, help='path to the JSON file with stock_name, start_date, num_days, interval, init_balance, output_path')

    # parse the command-line arguments
    args = parser.parse_args()

    # run the full run
    full_run(args.file_path)
    

if __name__ == "__main__":
    main()



