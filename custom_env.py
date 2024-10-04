import math
from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle
from parameter import custom_parameter
from datetime import datetime, timedelta

class CustomCarRacing(gym.Env, EzPickle):
    def __init__(self):
        self.box_matrix = np.zeros((custom_parameter.state_width, custom_parameter.state_height))

    def step(self, action):
        pass

    def reset(self):
        random.seed(custom_parameter.random_seed)
        self.next_create_time = np.random.normal(custom_parameter.create_time, custom_parameter.sigma)
        self.last_create_time = datetime.now()-timedelta(seconds=self.next_create_time) #create a box at the beginning
    
    def creat_box(self):
        if(datetime.now()-self.last_create_time>timedelta(seconds=self.next_create_time)):
            self.last_create_time = datetime.now()
            self.next_create_time = np.random.normal(custom_parameter.create_time, custom_parameter.sigma)
            x = np.random.uniform(0, self.width)
    def render(self, mode='human', close=False):
        pass

    def _render(self, mode='human', close=False):
        pass