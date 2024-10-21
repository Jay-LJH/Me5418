'''
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
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
import pygame

from typing import Optional, Union, Any
import random
from datetime import datetime

from constants import constants
from truck import truck
from cargo import cargo
from destination import destination
from utils import utils

'''
To avoid misleading wording and allow readability, some wordings in this project are changed.
For example, the 'box' object is replaced by 'cargo', avoiding the repeated name with OpenAI
Box and Box2D objects.
'''

# Our custom class is modified from the car racing environment from OpenAI gymnasium.
# The interfaces specified by OpenAI are implemented first, then we do helper functions.
class cargo_moving_truck_env(gym.Env):

    '''
    From OpenAI docs
    Like all environments, our custom environment will inherit from gymnasium.Env that defines 
    the structure of environment. One of the requirements for an environment is defining the 
    observation and action space, which declare the general set of possible inputs (actions) and 
    outputs (observations) of the environment.
    '''
    def __init__(self, render_mode = 'rgb_array'):

        # Set constants
        self.metadata["render_modes"] = ['human', 'rgb_array']
        self.metadata["FPS"] = constants.FPS
        self.metadata["width"] = constants.width
        self.metadata["height"] = constants.height

        # The action space is 3D, namely steer, gas and brake.
        # Truck has to stop in order to carry load
        self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )

        # The observation space is a height * width * 3 numpy array, representing the RGB image of the map.
        self.observation_space = spaces.Box(
            low = 0, high = 255, shape = (constants.height, constants.width, 3), dtype = np.uint8
        )            

        # Set display size and rate as specified in constants.constants.
        # Not syncronized with environment registration standard.
        # self.FPS = constants.FPS
        # self.width = constants.width
        # self.height = constants.height

        # Set render_mode, seems not possible for headless machines to do 'human' render mode.
        # We do 'rgb_array' render mode instead, on headless machines.
        self.render_mode = render_mode

    '''
    From OpenAI docs
    Resets the environment to an initial state, required before calling step. Returns the
    first agent observation for an episode and information, i.e. metrics, debug info.
    ...
    Parameters:
    seed (optional int) – The seed that is used to initialize the environment’s PRNG (np_random)
    and the read-only attribute np_random_seed. If the environment does not already have a PRNG 
    and seed = None (the default option) is passed, a seed will be chosen from some source of entropy
    (e.g. timestamp or /dev/urandom). However, if the environment already has a PRNG and seed = None
    is passed, the PRNG will not be reset and the env’s np_random_seed will not be altered. If you
    pass an integer, the PRNG will be reset even if it already exists. Usually, you want to pass an
    integer right after the environment has been initialized and then never again. Please refer to 
    the minimal example above to see this paradigm in action.
    
    options (optional dict) – Additional information to specify how the environment is reset (optional,
    depending on the specific environment)
    
    Returns:
    observation (ObsType) – Observation of the initial state. This will be an element of observation_space
    (typically a numpy array) and is analogous to the observation returned by step().
    
    info (dictionary) – This dictionary contains auxiliary information complementing observation. It
    should be analogous to the info returned by step().
    '''
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, Any]]:

        # If no world seed specified, we obtain the entropy through timestamp.
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(datetime.now().microsecond)

        # Create a 2D world with no gravity, for the purpose of the project.
        self.world = Box2D.b2World((0, 0)) # (0, 0) gravity vector

        # Store cargo locations 
        # We are doing continuous, but this looks grid-like. Are there better representations?
        # self.cargo_matrix = np.zeros((constants.width, constants.height))

        # Initialize reward values
        self.reward = 0.0
        self.prev_reward = 0.0

        # Initialize termination/truncation indicators
        self.terminated = False
        self.truncated = False

        # Create every object we need at beginning
        # TODO: we need the 'truck' object to be a child object of 'car' and in a separate file
        # Won't it be better if the 'truck' has a random position at the beginning?
        self.next_create_time = 0 # Time to create next box, create a box at time 0
        '''
        self.truck = Car(self.world, 0, constants.width / 2, constants.height / 2) # Create a car in the middle of the map
        self.truck.carry = None
        '''
        self.truck = truck(self.world, 0, constants.width / 2, constants.height / 2)
        '''
        self.destionation = (random.randint(0, self.width), random.randint(0, self.height))
        
        self.destination = destination(random.randint(0, self.width), random.randint(0, self.height))
        '''
        self.t = 0
        '''
        self.cargo_list = []
        '''

        # Initialize pygame objects for rendering
        self.screen = None
        self.clock = None

        # Nothing to put into 'info' yet
        if self.render_mode == 'human':
            self.render()

        if self.render_mode == 'rgb_array':
            return (self.render, {})
            
        return (None, {})

    '''
    From OpenAI docs
    Renders the environments to help visualise what the agent see, examples modes are “human”, 
    “rgb_array”, “ansi” for text.
    ...
    By convention, if the render_mode is:

    None (default): no render is computed.
    
    “human”: The environment is continuously rendered in the current display or terminal, usually for
    human consumption. This rendering should occur during step() and render() doesn’t need to be called.
    Returns None.
    
    “rgb_array”: Return a single frame representing the current state of the environment. A frame is a 
    np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
    
    “ansi”: Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for 
    each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
    
    “rgb_array_list” and “ansi_list”: List based version of render modes are possible (except Human) through 
    the wrapper, gymnasium.wrappers.RenderCollection that is automatically applied during gymnasium.make(..., 
    render_mode="rgb_array_list"). The frames collected are popped after render() is called or reset().
    '''

    # For the purpose of this project, we only implement 'human' and 'rgb_array' modes. Treat other 'render_modes' as None.
    # From gymnasium.envs.box2d.car_racing
    def render(self, mode = 'rgb_array', close = False) -> Union[np.array, list[np.array], None]:

        # No need to set font
        # pygame.font.init()

        # Render the display window if the environment is human controlled.
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((constants.video_width, constants.video_height))

        # Set time clock for pygame
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Recreate the frame surface
        self.surface = pygame.Surface((constants.video_width, constants.video_height))

        # Computing transformations
        angle = -self.truck.hull.angle

        '''
        # Animating first second zoom.
        zoom = 0.1 * 2.7 * max(1 - self.t, 0) + 2.7 * 6.0 * min(self.t, 1) # 6.0 scale, 2.7 zoom
        scroll_x = -(self.truck.hull.position[0]) * zoom
        scroll_y = -(self.truck.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (constants.video_width / 2 + trans[0], constants.video_height / 4 + trans[1])
        self.render_objects(zoom, trans, angle)
        
        self.truck.draw(
            self.surface,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        '''

        zoom = 2.7 * 6.0 # 6.0 scale, 2.7 zoom
        scroll_x = -(self.truck.hull.position[0])
        scroll_y = -(self.truck.hull.position[1])
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (constants.video_width / 2 + trans[0], constants.video_height / 4 + trans[1])
        self.render_objects(zoom, trans, angle)
        
        self.truck.draw(
            self.surface,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        
        self.surface = pygame.transform.flip(self.surface, False, True)
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(constants.FPS)
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surface, (constants.width, constants.height))

        return None
        

    '''
    From OpenAI docs
    Updates an environment with actions returning the next agent observation, 
    the reward for taking that actions, if the environment has terminated or 
    truncated due to the latest action and information from the environment 
    about the step, i.e. metrics, debug info.
    ...
    Parameters:
    action (ActType) – an action provided by the agent to update the environment state.
    
    Returns:
    observation (ObsType) – An element of the environment’s observation_space as the 
    next observation due to the agent actions. An example is a numpy array containing 
    the positions and velocities of the pole in CartPole.
    
    reward (SupportsFloat) – The reward as a result of taking the action.
    
    terminated (bool) – Whether the agent reaches the terminal state (as defined under 
    the MDP of the task) which can be positive or negative. An example is reaching the 
    goal state or moving into the lava from the Sutton and Barto Gridworld. If true, 
    the user needs to call reset().
    
    truncated (bool) – Whether the truncation condition outside the scope of the MDP is
    satisfied. Typically, this is a timelimit, but could also be used to indicate an agent
    physically going out of bounds. Can be used to end the episode prematurely before a
    terminal state is reached. If true, the user needs to call reset().
    
    info (dict) – Contains auxiliary diagnostic information (helpful for debugging, learning,
    and logging). This might, for instance, contain: metrics that describe the agent’s performance
    state, variables that are hidden from observations, or individual reward terms that are 
    combined to produce the total reward. In OpenAI Gym <v26, it contains “TimeLimit.truncated” to 
    distinguish truncation and termination, however this is deprecated in favour of returning 
    terminated and truncated variables.
    '''
    def step(self, action:Union[np.ndarray, int]) -> tuple[np.array, float, bool, bool, dict[str, Any]]:

        # Step everything in the environment by time.
        time_delta = 1.0 / self.FPS
        self.t += time_delta
        self.world.Step(time_delta, 6 * 30, 2 * 30)
        self.truck.step(action, time_delta)
        '''
        for cargo in self.cargo_list:
            cargo.step(time_delta)

        self.destination.step(self.truck)
        '''
        
        

        # TODO: enable box generation at random here.

        # TODO: enable rendering in 'rgb_array mode' (here?).
        self.reward = self.calculate_reward()
        if self.render_mode == 'human':
            self.render()

        # TODO: Not sure yet what to put into 'info' yet.
        return self.render(), self.reward, self.terminated, self.truncated, {}

    

    '''
    From OpenAI docs
    Closes the environment, important when external software is used, i.e. pygame for rendering, databases
    ...
    After the user has finished using the environment, close contains the code necessary to “clean up” the 
    environment.

    This is critical for closing rendering windows, database or HTTP connections. Calling close on an already 
    closed environment has no effect and won’t raise an error.
    '''
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    # Some of the followings are helper functions.
    # I might think them private, does not make sense for others to call them.

    # copy from gymnasium.envs.box2d.car_racing
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def render_objects(self, zoom, translation, angle):
        # draw background
        bounds = 96
        field = [
            (bounds, bounds),
            (bounds, 0),
            (0, 0),
            (0, bounds),
        ]
        utils.draw_colored_polygon(
            self.surface, field, constants.bg_color, zoom, translation, angle, clip=False
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
            utils.draw_colored_polygon(
                self.surface, poly, constants.grass_color, zoom, translation, angle
            )
        '''
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
        dest = [self.destionation, (self.destionation[0] + 1, self.destionation[1]),\
                (self.destionation[0] + 1, self.destionation[1]+1),(self.destionation[0], self.destionation[1]+1)]
        self._draw_colored_polygon(
            self.surf, dest, custom_parameter.dest_color, zoom, translation, angle
        )
        '''

    '''
    def create_box(self):
        if(self.t > self.next_create_time):
            self.next_create_time = np.random.normal(custom_parameter.create_time, custom_parameter.sigma) + self.t
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            while self.box_matrix[x][y] != 0  or (x,y) == self.destionation: 
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
            self.box_list.append(box(self, x, y, custom_parameter.box_width, custom_parameter.box_height, 
                                     np.random.normal(custom_parameter.expire_time, custom_parameter.sigma)))
    
    
    def calculate_reward(self):
        step_reward = custom_parameter.step_reward # small neg reward for each step
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
        return {"box": self.box_matrix,"position": self.car.hull.position, "velocity": self.car.hull.linearVelocity, 
                "carry": self.car.carry.expire_time if self.car.carry is not None else -1,"destination": self.destionation, "time": self.t}

    
        '''

    


    
