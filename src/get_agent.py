# this script is used to create or load agent.

from stable_baselines3 import PPO, PPO2, A2C, DDPG

# create a class for the agent, which is used to store either the stable-baselines agent, random sampling action space agent, or else
class Agent:
    def __init__(self, env, agent_type, model_path = None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        if agent_type.startswith('stable-baselines'):
            model_type = agent_type.split('-')[1]
            if model_type == 'ppo2':
                self.agent = PPO2.load(model_path)
            elif model_type == 'ppo':
                self.agent = PPO.load(model_path)
            elif model_type == 'a2c':
                self.agent = A2C.load(model_path)
            elif model_type == 'ddpg':
                self.agent = DDPG.load(model_path)
        elif agent_type == 'algo':
            self.agent = None
        elif agent_type == 'random':
            self.agent = None
    
    def predict(self, state, deterministic=False):
        if self.agent is None:
            return self.action_space.sample()
        else:
            return self.agent.predict(state, deterministic=deterministic)