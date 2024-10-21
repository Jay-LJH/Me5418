from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
from typing import Union

class truck(Car):

    def __init__(self, world, init_angle, init_x, init_y):
        # Its drive function is the same as parent
        super().__init__(world, init_angle, init_x, init_y)
        
        # With capacity of one cargo, intitially empty.
        self.carried = []

    """
    def step(self, action:Union[np.ndarray, int]):
        pass
        # Take the action if the pass in is in action space, or do nothing if no action specified.
        '''
        if action is not None:
            self.car.steer(-action[0][0])
            self.car.gas(action[0][1])
            self.car.brake(action[0][2])
        '''
        

        # TODO: pick up cargo when action[1][0] == 1

    def render(self):
        pass

    def carry(self):
        pass
    """