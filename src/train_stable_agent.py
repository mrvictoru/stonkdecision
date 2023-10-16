# train an agent using stable baselines

import torch
# set detect anomaly to true
torch.autograd.set_detect_anomaly(True)

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# import custom functions and classes
from curatedataset import makegymenv
from TradingEnvClass import StockTradingEnv

# create a custom SubprocVecEnv class to allow rendering of the environment according to the custom render function
class CustomSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.current_env = 0 # index of the current environment to render

    def render(self, mode=None):
        # create an empty numpy array to store the rendered observations
        obs_list = []
        # loop through all the remote objects
        for remote in self.remotes:
            # send a render command with the print argument to the remote object
            remote.send(('render', mode))
            # receive the rendered observation
            obs = remote.recv()

            # append the observation to the list
            obs_list.append(obs)
        # return the observation
        return obs_list

def make_dummy_env(stock_name, start_date, num_days, interval, num_cpu):
    env, obs_space_shape, act_space_shape, obs_features, data = makegymenv(stock_name, start_date, num_days, interval)
    env_stable = CustomSubprocVecEnv([lambda: env for i in range(num_cpu)])
    env_stable_dum = DummyVecEnv([lambda: env])
    return env_stable, env_stable_dum

def create_stable_agents(env_stable, env_stable_dum):
    modelPPO = PPO("MlpPolicy", env_stable, verbose=1)
    modelA2C = A2C("MlpPolicy", env_stable, verbose=1)
    # there seems to be a problem with this model (DDPG)
    modelDDPG = DDPG("MlpPolicy", env_stable_dum, verbose=1)
    # store the models' name in a list
    return [modelPPO, modelA2C, modelDDPG]

def evaluate_stable_agent(model, env, n_eval_ep = 10):
    # evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_ep)
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

