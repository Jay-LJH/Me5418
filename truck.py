from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
from typing import Union

from cargo import cargo

class truck(Car):

    def __init__(self, world, init_angle, init_x, init_y):
        # Its drive function is the same as parent
        super().__init__(world, init_angle, init_x, init_y)
        
        # With capacity of one cargo, intitially empty.
        self.carried = []

    def step(self, action:np.ndarray, timedelta):

        # Do the specified actions
        if action is not None:
            self.steer(-action[0])
            self.gas(action[1])
            self.brake(action[2])

        # Update truck state
        super().step(timedelta)

    def draw(self, surface, zoom, translation, angle, draw_particles=True):

        # Draw the truck itself
        super().draw(surface, zoom, translation, angle, draw_particles)

        # TODO: Draw the cargo if it is loaded

    def carry(self, cargo):

        