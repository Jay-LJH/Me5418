from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
from typing import Union
from copy import copy

from utils import utils
from constants import constants
from cargo import cargo
'''
The truck entity, inherit physics from OpenAI racing car
with passive loading capability
'''
class truck(Car):

    def __init__(self, world, metadata):
        
        # Create pointer, modify global observation directly.
        self.metadata = copy(metadata)
        
        # Its drive function is the same as parent.
        super().__init__(world, metadata[2], metadata[0], metadata[1])
        
        # With capacity of one cargo, intitially empty.
        self.carried = []

        self.rest_carry_time = 0.0
        self.rest_unload_time = 0.0

    # For the environment to update observation
    def collect_observation(self):
        return self.metadata

    # Allow environment check status of truck, whether it is carrying load
    def is_carried(self):
        return len(self.carried) == 1

    # Allow environment to specify cargo carrying
    def carry(self, cargo):

        # Set the cargo to be carried
        cargo.set_carried()

        # Add the cargo to capacity
        self.carried.append(cargo)

        # Set carry timer
        self.rest_carry_time = constants.CARGO_LOAD_TIME

    # Allow environment to specify cargo unloading
    def unload(self):

        # Remove cargo
        self.carried.remove(self.carried[0])

        # Set unload timer
        self.rest_unload_time = constants.CARGO_UNLOAD_TIME

    # The action space is 3D, namely steer, gas and brake.
    # Carry/Unload is implemented such that truck has to stop in order to carry load
    def step(self, action:np.ndarray, timedelta):

        # Calculate rest carry/unload time
        if self.rest_carry_time > constants.EQUIVALANCE_THRESHOLD:
            self.rest_carry_time -= timedelta

        if self.rest_unload_time > constants.EQUIVALANCE_THRESHOLD:
            self.rest_unload_time -= timedelta
            
        # Do the specified actions if it is not carrying/unloading
        # Force stop truck otherwise
        if self.rest_carry_time < constants.EQUIVALANCE_THRESHOLD and \
            self.rest_unload_time < constants.EQUIVALANCE_THRESHOLD:
            
            if action is not None:
                self.steer(-action[0])
                self.gas(action[1])
                self.brake(action[2])
                
        else:
            
            self.steer(0)
            self.gas(0)
            self.brake(1)

        # Update truck state
        super().step(timedelta)

        # Update metadata, the forth variable is the logit whether it is carrying load, 
        # the fifth one is cargo expiration time.
        self.metadata = [self.hull.position[0], self.hull.position[1], self.hull.angle, 
                         len(self.carried), 0 if len(self.carried) == 0 else self.carried[0].get_expiration_time()]

    def draw(self, surface, zoom, translation, angle, draw_particles=True):

        # Draw the truck itself
        super().draw(surface, zoom, translation, angle, draw_particles)

        # Draw the cargo on the truck if it is loaded
        if len(self.carried) == 1:
            coordinates = [(self.hull.position[0] - constants.CARRIED_CARGO_SIZE / 2, self.hull.position[1] - constants.CARRIED_CARGO_SIZE / 2), 
                           (self.hull.position[0] + constants.CARRIED_CARGO_SIZE / 2, self.hull.position[1] - constants.CARRIED_CARGO_SIZE / 2), 
                           (self.hull.position[0] + constants.CARRIED_CARGO_SIZE / 2, self.hull.position[1] + constants.CARRIED_CARGO_SIZE / 2), 
                           (self.hull.position[0] - constants.CARRIED_CARGO_SIZE / 2, self.hull.position[1] + constants.CARRIED_CARGO_SIZE / 2)]

            utils.draw_colored_polygon(surface, coordinates, constants.CARGO_COLOR, zoom, trans, angle)

    def __del__(self):
        del self.metadata


        