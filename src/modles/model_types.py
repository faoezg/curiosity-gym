from enum import Enum
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3

class Models(Enum):
    PPO = PPO
    A2C = A2C
    TD3 = TD3
