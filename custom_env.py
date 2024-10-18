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
import time
import cv2

class logger:
    def __init__(self):
        self.log = custom_parameter.logger
    def log(message):
        if logger.log:
            print(message)

class box:
    def __init__(self, env,x, y, width, height,expire_time):
        self.env = env
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.expire_time = expire_time
        self.expired = False
        self.carry = False
        # self.body= self.env.world.CreateStaticBody(
        #     position=(x, y),
        #     shapes=polygonShape(box=(width/2, height/2)),
        # )
        self.env.box_matrix[x][y] = expire_time
        self.collision_enter = False
    def reward(self,car):
        reward = 0  
        distance = euclidean((car.hull.position), (self.x, self.y))
        if distance < custom_parameter.crash_distance: # close enough to pick up the box
            if (math.sqrt(car.hull.linearVelocity.lengthSquared) < custom_parameter.crash_speed \
                and car.carry is None):   # pick up the box
                car.carry = self
                self.carry = True
                reward += custom_parameter.pickup_reward
                self.env.box_matrix[self.x][self.y] = 0
                self.destroy_on_map() # destroy the box on the map
                logger.log("Pick up")

            elif self.collision_enter == False: # if crash for the first time
                reward += custom_parameter.crash_reward
                self.collision_enter = True
                logger.log("Crash")
        else:
            self.collision_enter = False

        if self.env.t > self.expire_time: # if the box is expired
            if self.expired:
                reward += custom_parameter.expire_reward_continuous
            else:
                self.expired = True
                reward += custom_parameter.expire_reward
                logger.log("expire box at ({},{}),expire time: {}".format(self.x, self.y, self.expire_time))

        if (self.env.t>self.expire_time + custom_parameter.destory_time): #destroy the box after it expired for 20 seconds
            self.destroy_on_map()
            self.destroy()
            if car.carry == self:
                car.carry = None
            logger.log("Destroy box at ({},{},destroy time: {})".format(self.x, self.y,self.env.t))
        return reward

    def destroy_on_map(self):
        self.env.box_matrix[self.x][self.y] = 0
        # self.env.world.DestroyBody(self.body)
        
    def destroy(self):
        self.env.box_list.remove(self)

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
    def __init__(self,render_mode='human'):
        self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            ) 
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(custom_parameter.height, custom_parameter.width, 3), dtype=np.uint8
        ) # observation space is a height*width*3 numpy array, representing the RGB image of the map             
        self.FPS = custom_parameter.FPS
        self.width = custom_parameter.width
        self.height = custom_parameter.height
        self.render_mode = render_mode

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
        self.state = self.render('rgb_array')
        if self.render_mode == 'human':
            self.render()
        return self.get_obs(), self.reward, self.done

    def calculate_reward(self):
        step_reward = custom_parameter.step_reward #small neg reward for each step
        for b in self.box_list:
            step_reward += b.reward(self.car) # add reward for each box
        if self.car.carry is not None: # add reward for reaching destination
            distance = euclidean((self.car.hull.position), (self.destionation))
            if distance < custom_parameter.crash_distance and math.sqrt(self.car.hull.linearVelocity.lengthSquared) < custom_parameter.crash_speed:
                self.car.carry.destroy()
                self.car.carry = None
                step_reward += custom_parameter.reach_reward
                logger.log("Reach")
        
        return step_reward  
       
    def get_obs(self):
        return {"state":self.state,"box": self.box_matrix,"position": self.car.hull.position, "velocity": self.car.hull.linearVelocity,"angle": self.car.hull.angle,
                "carry": self.car.carry.expire_time if self.car.carry is not None else -1,"destination": self.destionation, "time": self.t}

    def reset(self):
        random.seed(custom_parameter.random_seed)
        self.world = Box2D.b2World((0, 0)) #（0，0） gravity vector 
        self.box_matrix = np.zeros((custom_parameter.width, custom_parameter.height))
        self.reward = 0.0
        self.prev_reward = 0.0
        self.done = False
        self.next_create_time = 0 #time to create next box, create a box at time 0
        self.car = Car(self.world, 0, custom_parameter.width/2, custom_parameter.height/2) #create a car in the middle of the map
        self.car.carry = None
        self.destionation = (random.randint(0, self.width), random.randint(0, self.height))
        self.t=0
        self.box_list = []
        self.screen = None
        self.clock = None
        if self.render_mode == 'human':
            self.render()
        

    def create_box(self):
        if(self.t>self.next_create_time):
            self.next_create_time = np.random.normal(custom_parameter.create_time, custom_parameter.sigma) + self.t
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            while self.box_matrix[x][y] != 0  or (x,y) == self.destionation: 
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
            self.box_list.append(box(self, x, y, custom_parameter.box_width, custom_parameter.box_height, 
                                     self.t + np.random.normal(custom_parameter.expire_time, custom_parameter.sigma)))

    # copy from gymnasium.envs.box2d.car_racing
    def render(self, mode='human', close=False):
        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((custom_parameter.video_width, custom_parameter.video_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((custom_parameter.video_width, custom_parameter.video_height))
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * 2.7 * max(1 - self.t, 0) + 2.7 * 6.0 * min(self.t, 1) #6.0 scale,2.7 zoom
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = ( custom_parameter.video_width/ 2 + trans[0], custom_parameter.video_height / 4 + trans[1])
        self.render_objects(zoom,trans,angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        self.surf = pygame.transform.flip(self.surf, False, True)
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(custom_parameter.FPS)
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
            
        if mode == "rgb_array":
            return self._create_image_array(self.surf, (custom_parameter.width, custom_parameter.height))

    def render_objects(self,zoom, translation, angle):
        # draw background
        bounds = 96
        field = [
            (bounds, bounds),
            (bounds, 0),
            (0, 0),
            (0, bounds),
        ]
        self._draw_colored_polygon(
            self.surf, field, custom_parameter.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass
        grass_dim = bounds //20
        grass = []
        for x in range(0, 20, 2):
            for y in range(0, 20, 2):
                grass.append(
                    [
                        (grass_dim * x + grass_dim, grass_dim * y + 0),
                        (grass_dim * x + 0, grass_dim * y + 0),
                        (grass_dim * x + 0, grass_dim * y + grass_dim),
                        (grass_dim * x + grass_dim, grass_dim * y + grass_dim),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, custom_parameter.grass_color, zoom, translation, angle
            )
        # draw box
        for b in self.box_list:
            if b.carry:
                continue
            box = [
                (b.x + custom_parameter.box_width / 2, b.y + custom_parameter.box_height / 2),
                (b.x - custom_parameter.box_width / 2, b.y + custom_parameter.box_height / 2),
                (b.x - custom_parameter.box_width / 2, b.y - custom_parameter.box_height / 2),
                (b.x + custom_parameter.box_width / 2, b.y - custom_parameter.box_height / 2),
            ]
            self._draw_colored_polygon(
                self.surf, box, custom_parameter.box_color, zoom, translation, angle
            )
        # draw destination
        dest = [self.destionation, (self.destionation[0] + custom_parameter.box_width, self.destionation[1]),\
                (self.destionation[0] + custom_parameter.box_width, self.destionation[1]+custom_parameter.box_height),(self.destionation[0], self.destionation[1]+custom_parameter.box_height)]
        self._draw_colored_polygon(
            self.surf, dest, custom_parameter.dest_color, zoom, translation, angle
        )

    # copy from gymnasium.envs.box2d.car_racing
    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        if not clip or any(
            (-custom_parameter.max_shape_dim <= coord[0] <= custom_parameter.video_width + custom_parameter.max_shape_dim)
            and (-custom_parameter.max_shape_dim <= coord[1] <= custom_parameter.video_height  + custom_parameter.max_shape_dim)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)
   
    # copy from gymnasium.envs.box2d.car_racing
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])
    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CustomCarRacing(render_mode="human")
    env.reset()
    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        s, r, terminated= env.step(a)
        while True:
            register_input()
            s, r, terminated= env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated :
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f} at time {env.t}")
                positions = [(i, j) for i, row in enumerate(s["box"]) for j, value in enumerate(row) if value != 0]
                print(f"boxes: {positions}")
            steps += 1
            if terminated or restart or quit:
                break
    env.close()
