import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
import pygame
from scipy.spatial.distance import euclidean
from typing import Optional, Union, Any
import random
from datetime import datetime

from constants import constants
from truck import truck
from cargo import cargo
from destination import destination
from utils import utils
from copy import copy
from Logger import logger

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

        # Set constants.
        self.metadata["render_modes"] = ['human', 'rgb_array']
        self.metadata["FPS"] = constants.FPS
        self.metadata["width"] = constants.WIDTH
        self.metadata["height"] = constants.HEIGHT

        # The action space is 3D, namely steer, gas and brake.
        # Carry/Unload is implemented such that truck has to stop in order to carry load
        self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )

        # Different definitions of observation space for partially observable and the fully observable ones.
        if constants.FULLY_OBSERVABLE:
            # The full observation space is the map size, the position and orientation of the truck itself,
            # the position and orientation, appear time and expiration of boxes, and the position and orientation of destination.
            # Map size is defined in 'constants.WIDTH' and 'constants.HEIGHT'
            # The first index is for the truck, the second is for the destination, and the rest are for cargoes.
            # The forth data of the truck array is for not loaded/loaded state, the fifth one keep track of cargo expiration time.
            # Therefore, the data structure is defined by the following:
            self.observation_space = spaces.Box(
                # lower bound 0 for positions because we need to represent the cargo carried on truck
                # and only for carried because we have bleed.
                # lower bound 0 for create time and expire time because we have truck and
                # destination with no create time and expire time.
                low = np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 
                high = np.array([constants.WIDTH - constants.BLEED, constants.HEIGHT - constants.BLEED, 
                                 2 * constants.PI, constants.MAX_CREATE_TIME, constants.MAX_EXPIRE_TIME]), 
                shape = (5, ), dtype = np.float32
            )
        else:
            # The partial observation space is the same as human operation.
            # i.e. It is a height * width * 3 numpy array, representing the rendered surface.
            self.observation_space = spaces.Box(
                low = 0, high = 255, shape = (constants.HEIGHT, constants.WIDTH, 3), dtype = np.uint8
            )

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

        # Randomly generate objects in question.
        # Specificly, they are 'self.truck', 'self.cargo', 'self.destination' and determine
        # 'self.world_gen', which is the observation space if the envionment is fully observable.
        self._generate_world()

        # Initialize reward.
        self.reward = 0.0

        # Initialize termination/truncation indicators.
        self.terminated = False
        self.truncated = False

        # Initialize timestep.
        self.t = 0
        
        # Set time clock for pygame.
        self.clock = pygame.time.Clock()

        # Initialize screen for human manipuation.
        self.screen = None

        # Nothing to put into 'info' yet
        if self.render_mode == 'human':
            # Initialize the display window if the environment is human controlled.
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((constants.VIDEO_WIDTH, constants.VIDEO_HEIGHT))
            return (self.render(), {"reward": self.reward})

        # Put world gen into additional information if fully observable space specified.
        # Also, put reward as additional information.
        if self.render_mode == 'rgb_array':
            if constants.FULLY_OBSERVABLE:
                return (self.full_observation, {"world_gen": self.world_gen})
            else:
                return (self.render(), {})
            
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
    # From gymnasium.envs.box2d.car_racing.
    def render(self, mode = 'rgb_array', close = False) -> Union[np.array, list[np.array], None]:

        # Recreate the frame surface for rendering.
        self.surface = pygame.Surface((constants.VIDEO_WIDTH, constants.VIDEO_HEIGHT))

        # Computing transformations.
        angle = -self.truck.hull.angle

        # Compute rendering parameters for 3rd person, with our agent on the center of vision.
        scroll_x = -(self.truck.hull.position[0]) * constants.ZOOM
        scroll_y = -(self.truck.hull.position[1]) * constants.ZOOM
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (constants.VIDEO_WIDTH / 2 + trans[0], constants.VIDEO_HEIGHT / 4 + trans[1])

        # Render entities, the highest layer priority last.
        # Render the 'habor' in question.
        self._render_field(trans, angle)

        # Render the destination
        self.destination.draw(self.surface, constants.ZOOM, trans, angle)

        # Render cargoes
        for cargo in self.cargoes:
            if cargo.is_on_ground():
                cargo.draw(self.surface, constants.ZOOM, trans, angle)

        # Render our agent, the truck.
        self.truck.draw(self.surface, constants.ZOOM, trans, angle,
                        mode not in ["state_pixels_list", "state_pixels"])
        
        self.surface = pygame.transform.flip(self.surface, False, True)
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(constants.FPS)
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surface, (constants.VIDEO_WIDTH, constants.VIDEO_HEIGHT))

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
    def step(self, action:np.ndarray) -> tuple[np.array, float, bool, bool, dict[str, Any]]:

        # Step everything in the environment by time.
        time_delta = 1.0 / self.metadata['FPS']
        self.t += time_delta

        # According to documentation, setting both precision to 10 is sufficient for simulation.
        self.world.Step(time_delta, constants.VELOCITY_PRECISION, constants.POSITION_PRECISION)

        # Step the truck with given action
        self.truck.step(action, time_delta)

        # Step all the cargoes, emerge, vanish or become loaded at right times
        for cargo in self.cargoes:
            cargo.step(time_delta)

        # Step the destination: no action required
        self.destination.step(time_delta)

        # Update full observation
        self.full_observation = [self.truck.collect_observation(), self.destination.collect_observation()] + \
            [cargo.collect_observation() for cargo in self.cargoes if cargo.is_on_ground()]

        # Compute interactions of the entities in the state
        # e.g. Is there a crash? Is the truck carrying cargo?
        status, expire_count = self._status_step()

        # Based on status, calculate reward for this step.
        self._calculate_reward_step(status, expire_count)

        # Terminate episode if global time limit is reached
        if self.t >= constants.MAX_TERMINATION:
            self.terminated = True

        # Refresh the last observation
        del self.last_observation
        self.last_observation = copy(self.full_observation)

        # Render to window directly if human manipulation.
        if self.render_mode == 'human':
            self.render()

        done = self.terminated or self.truncated

        # Put world gen into additional information if fully observable space specified.
        if constants.FULLY_OBSERVABLE:
            return self.full_observation, self.reward, self.terminated, self.truncated, {"world_gen": self.world_gen}
        #return includes: 
        else:
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
        # Close pygame window
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

        del self.truck
        del self.destination
        del self.last_observation
        for cargo in self.cargoes:
            del cargo

        # TODO: Delete all the rest objects, arraylists here.

    # Some of the followings are helper functions.
    # I might think them private, does not make sense for others to call them.

    # We generate all the information we need to form the problem at the beginning.
    # So that we only generate world once and cache it to enhance performance and make the 
    # code more trackable.
    # To make the environment stochastic, reset the environment after a small batch of training.
    def _generate_world(self):
        
        # Create world gen.
        self.world_gen = []

        # Generate truck metadata, with x, y positions and random angle, no emerge/vanish time.
        self.world_gen.append([
            random.uniform(constants.BLEED, constants.WIDTH - constants.BLEED),
            random.uniform(constants.BLEED, constants.HEIGHT - constants.BLEED),
            random.uniform(0, 2 * constants.PI), 0, 0
        ])

        # Generate destination metadata, with x, y positions and random angle, no emerge/vanish time.
        self.world_gen.append([
            random.uniform(constants.BLEED, constants.WIDTH - constants.BLEED),
            random.uniform(constants.BLEED, constants.HEIGHT - constants.BLEED),
            random.uniform(0, 2 * constants.PI), 0, 0
        ])

        # Pre-create some cargo, also no physics for cargo, not interesting for our problem.
        self.num_of_cargoes = random.randint(constants.MIN_NUM_CARGOES, constants.MAX_NUM_CARGOES)

        # Generate cargo metadata, with x, y positions and random angle, and emerge/vanish time.
        for _ in range(0, self.num_of_cargoes):
            self.world_gen.append([
                random.uniform(constants.BLEED, constants.WIDTH - constants.BLEED),
                random.uniform(constants.BLEED, constants.HEIGHT - constants.BLEED),
                random.uniform(0, 2 * constants.PI),
                random.uniform(constants.MIN_CREATE_TIME, constants.MAX_CREATE_TIME),
                random.uniform(constants.MIN_EXPIRE_TIME, constants.MAX_EXPIRE_TIME)
            ])

        # Create the truck at random positions.
        self.truck = truck(self.world, self.world_gen[0])
        
        # Create the destination, no need "self.world" since it has no physics.
        self.destination = destination(self.world_gen[1])

        # Pre-create the cargoes, this is somewhat anti-intuitive but convenient for parallizing entity workflow.
        self.cargoes = [cargo(metadata) for metadata in self.world_gen[2:]]

        # Create full observation data by collecting all the object observations
        self.full_observation = [self.truck.collect_observation(), self.destination.collect_observation()] + \
            [cargo.collect_observation() for cargo in self.cargoes if cargo.is_on_ground()]

        # Create last observation buffer, since we need to give reward based on state transfer.
        self.last_observation = copy(self.full_observation)

    # Enable rgb_array rendering.
    # Copy from gymnasium.envs.box2d.car_racing.
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    # Render the 'habor' in question.
    def _render_field(self, translation, angle):
        
        # Draw background as a dark-green surface.
        bounds = 96
        field = [
            (bounds, bounds),
            (bounds, 0),
            (0, 0),
            (0, bounds),
        ]
        
        utils.draw_colored_polygon(
            self.surface, field, constants.BG_COLOR, constants.ZOOM, translation, angle, clip = False
        )

        # Draw grass as light-green grids.
        grass_dim = bounds // 20
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
                self.surface, poly, constants.GRASS_COLOR, constants.ZOOM, translation, angle
            )

    # Check status of everything for calculating reward.
    # 0 for accident truncation, 1 for empty truck, 2 for loading cargo, 3 for loaded truck, 
    # 4 for unloading at destination
    # The second interger counts number of boxes expired
    def _status_step(self) -> (int, int):

        vanish = 0

        # Get truck metadata through full observation as updated by truck above
        truck_position = (self.full_observation[0][0], self.full_observation[0][1])
        truck_load_logit = self.full_observation[0][3]
        
        # Check whether accident occurs at boarder
        # The boarder is not hard. it is a habor, truck get into water if it go across boarder.
        x_out_of_bound = truck_position[0] < constants.REACH_DISTANCE or \
            truck_position[0] > constants.WIDTH - constants.REACH_DISTANCE
        y_out_of_bound = truck_position[1] < constants.REACH_DISTANCE or \
            truck_position[1] > constants.HEIGHT - constants.REACH_DISTANCE
        
        # Check every cargo on the ground
        for cargo in self.cargoes:
            if cargo.is_on_ground:

                # If the cargo is expired, vanish it and continue loop
                if cargo.get_expiration_time() < constants.EQUIVALANCE_THRESHOLD:
                    self.cargoes.remove(cargo)
                    vanish += 1

                # Compute distance between cargo and truck
                distance_to_cargo = euclidean(truck_position, 
                                     (cargo.collect_observation()[0], cargo.collect_observation()[1]))

                # If truck and cargo are close enough
                if distance_to_cargo < constants.REACH_DISTANCE: 
    
                    # If truck is empty and close to cargo at low velocity
                    if (math.sqrt(self.truck.hull.linearVelocity.lengthSquared) < constants.STOP_THRESHOLD \
                        and truck_load_logit == 0):
    
                        # Load cargo
                        self.truck.carry(cargo)
                        logger.log("Load")
                        return (2, vanish)
                        
                    # If cargo is reached at large velocity, or if the truck is already loaded, we consider these as crashes.
                    else:
                        return (0, vanish)

        # Report accident if truck gets out of bound after cargo vanish calculation
        if x_out_of_bound or y_out_of_bound:
            return (0, vanish)

        # Destination unload check
        destination_position = (self.full_observation[1][0], self.full_observation[1][1])

        # Check for loaded truck reach destination
        if truck_load_logit == 1:

            # Compute distance between destination and truck
            distance_to_destination = euclidean(truck_position, destination_position)

            # If truck and cargo are close enough
            if distance_to_destination < constants.REACH_DISTANCE:

                # No collision for destination, just unload at low velocities
                if math.sqrt(self.truck.hull.linearVelocity.lengthSquared) < constants.STOP_THRESHOLD:
                    self.truck.unload()
                    return (4, vanish)

            else:
                return (3, vanish)

        return (1, vanish)

    # Calculate reward after each step.
    # Give reward to status 4, penalty for status 0.
    # Also, give reward for saving energy, etc.
    def _calculate_reward_step(self, status, expire_count):

        # Reward the truck if generalized velocity is below low threshold.
        if abs(self.last_observation[0][0] - self.full_observation[0][0]) < constants.STOP_THRESHOLD \
        and abs(self.last_observation[0][1] - self.full_observation[0][1]) < constants.STOP_THRESHOLD \
        and abs(self.last_observation[0][2] - self.full_observation[0][2]) < constants.STOP_THRESHOLD:
            self.reward += constants.STOP_REWARD / self.metadata['FPS']

        # Penalty for expired cargo
        self.reward += constants.EXPIRE_REWARD * expire_count

        # Reward for sending cargo to destination
        # Penalize for sending expired cargo though
        if status == 4:
            self.reward += constants.CARGO_REACH_DEST_REWARD
            if self.full_observation[0][4] < constants.EQUIVALANCE_THRESHOLD:
                self.reward += constants.EXPIRE_REWARD

        # Continuous penalty for expired cargo on truck
        if self.full_observation[0][3] == 1 and \
        self.full_observation[0][4] < constants.EQUIVALANCE_THRESHOLD:
            self.reward += constants.EXPIRE_REWARD_CONTINUOUS / self.metadata['FPS']
        
        # If there is an accident, truncate this episode directly.
        if status == 0:
            self.reward += constants.CRASH_REWARD
            self.truncated |= True
            logger.log("Accident")
