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
import random
from scipy.spatial.distance import euclidean

class box:
    def __init__(self, env,x, y, width, height,expire_time):
        self.env = env
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.expire_time = expire_time
        self.expired = False
        self.body= self.env.world.CreateStaticBody(
            position=(x, y),
            shapes=polygonShape(box=(width/2, height/2)),
        )
    def reward(self,car):
        reward = 0  
        distance = euclidean((car.hull.position), (self.x, self.y))
        if distance < custom_parameter.crash_distance: # close enough to pick up the box
            if car.hull.velocity.norm() > 0.5 or car.carry is not None: # if the car is moving or already carrying a box
                reward += custom_parameter.crash_reward
            else:   # pick up the box
                car.carry = self
                reward += custom_parameter.pickup_reward
                self.env.box_matrix[self.x][self.y] = 0
                self.destroy() # destroy the box on the map

        if(self.env.t>self.expire_time):
            if self.expired:
                reward += custom_parameter.expire_reward_continuous
            else:
                self.expired = True
                reward += custom_parameter.expire_reward
        return reward

    def destroy(self):
        self.world.DestroyBody(self.body)

'''
To Do:
init function
reset function
step function
create box
calculate reward
render environment for human & imitation learning
'''
class CustomCarRacing(gym.Env):
    def __init__(self):
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
        self.world = Box2D.b2World((0, 0)) #（0，0） gravity vector        
        self.FPS = custom_parameter.FPS
        self.width = custom_parameter.width
        self.height = custom_parameter.height

    def step(self, action:Union[np.ndarray, int]):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
        self.car.step(1.0/self.FPS)
        self.t += 1.0/self.FPS
        self.world.Step(1.0/self.FPS, 6*30, 2*30)
        self.create_box()
        self.reward = self.calculate_reward()
        return self.get_obs(), self.reward, self.done

    def calculate_reward(self):
        step_reward = custom_parameter.step_reward #small neg reward for each step
        for b in self.box_list:
            step_reward += b.reward(self.car)
        if self.car.carry is not None: # add reward for reaching destination
            distance = euclidean((self.car.hull.position), (self.destionation))
            if distance < custom_parameter.crash_distance and self.car.hull.velocity.norm() < 0.5:
                self.car.carry = None
                step_reward += custom_parameter.reach_reward   
        return step_reward  
       
    def get_obs(self):
        #todo: get observation
        pass

    def reset(self):
        random.seed(custom_parameter.random_seed)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.done = False 
        self.next_create_time = 0 #time to create next box, create a box at time 0
        self.car = Car(self.world, 0, custom_parameter.width/2, custom_parameter.height/2) #create a car in the middle of the map
        self.car.carry = None
        self.box_list = []
        self.t = 0.0
        self.destionation = (random.randint(0, self.width), random.randint(0, self.height))

    def create_box(self):
        if(self.t>self.next_create_time):
            self.next_create_time = np.random.normal(custom_parameter.create_time, custom_parameter.sigma) + self.t
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            while self.box_matrix[x][y] == 1 or (x,y) == self.destionation: 
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
            self.box_matrix[x][y] = 1
            self.box_list.append(box(self, x, y, custom_parameter.box_width, custom_parameter.box_height, 
                                     np.random.normal(custom_parameter.expire_time, custom_parameter.sigma)))


    def render(self, mode='human', close=False):
        #todo: render the environment in different mode: human & RGB
        pass

    def _render(self, mode='human', close=False):
        pass
    
    def close(self):
        pass

if __name__ == '__main__':
    env = CustomCarRacing()
    env.reset()
    env.step([0,0,0])
    env.close()