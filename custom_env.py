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
import Box2D
from Box2D.b2 import contactListener, fixtureDef, polygonShape
import pygame
from pygame import gfxdraw

class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        # todo: implement collision between car and box and reward the car
        pass

'''
To Do:
init function
reset function
step function
create box
render environment
calculate reward
'''
class CustomCarRacing(gym.Env, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(self,render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            True, # continuous
        )
        self.box_matrix = np.zeros((custom_parameter.width, custom_parameter.height))
        self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(custom_parameter.height, custom_parameter.width, 3), dtype=np.uint8
        ) # observation space is a height*width*3 numpy array, representing the RGB image of the map
        self.reward = 0.0
        self.prev_reward = 0.0
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref) #（0，0） gravity vector        
        self.FPS = custom_parameter.FPS

    def step(self, action):
        pass

    def reset(self):
        random.seed(custom_parameter.random_seed)
        self.done = False 
        self.next_create_time = 0 #time to create next box, create a box at time 0
        self.car = Car(self.world, 0, custom_parameter.width/2, custom_parameter.height/2) #create a car in the middle of the map
        self.box_list = []
        self.t = 0

    def creat_box(self):
        if(self.t>self.next_create_time):
            self.next_create_time = np.random.normal(custom_parameter.create_time, custom_parameter.sigma) + self.t
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            self.box_matrix[x][y] = 1
            box = self.world.CreateStaticBody(
                position=(x, y),
                shapes=polygonShape(box=(custom_parameter.box_width, custom_parameter.box_height)),
                userData='box'
            )
            self.box_list.append(box)

    def render(self, mode='human', close=False):
        pass

    def _render(self, mode='human', close=False):
        pass

if __name__ == '__main__':
    env = CustomCarRacing()
    env.reset()
    env.render()
    env.close()