from copy import copy

from constants import constants
from utils import utils

class cargo:
    
    def __init__(self, metadata):

        self.metadata = copy(metadata)

        # Unpack metadata
        self.x = metadata[0]
        self.y = metadata[1]
        self.angle = metadata[2]
        self.create_time = metadata[3]
        self.vanish_time = metadata[4]

        # Rendering
        self.size = constants.CARGO_SIZE
        self.color = constants.CARGO_COLOR

        # State track
        self._is_on_ground = True if self.create_time < constants.EQUIVALANCE_THRESHOLD else False                                            
        
    # Calculate so that cargo emerge/vanish/become carried at right timestamps.
    def step(self, timedelta):

        if self.vanish_time >= constants.EQUIVALANCE_THRESHOLD:
            
            # Calculate timestamps time left
            if self.create_time >= constants.EQUIVALANCE_THRESHOLD:
                self.create_time -= timedelta
                
            else:
                self.vanish_time -= timedelta
                self._is_on_ground = True
                self.metadata[4] = self.vanish_time 
                
        else:

            # Vanish itself (no longer on ground) if it is still on ground
            self._is_on_ground = False
            
    # Let truck carry
    def set_carried(self):
        # Set it not on ground
        self._is_on_ground = False

    # Let truck observe expiration time
    def get_expiration_time(self):
        return self.vanish_time

    # Determine whether cargo is both created and yet to be carried by truck for rendering
    def is_on_ground(self):
        return self._is_on_ground

    # The environment only gets the observation if the cargo is on the harbor.
    def collect_observation(self):
        return self.metadata
        
    # The cargo is rendered only if it is on the harbor
    def draw(self, surface, zoom, translation, angle):
        
        coordinates = [(self.x - self.size / 2, self.y - self.size / 2), 
                       (self.x + self.size / 2, self.y - self.size / 2), 
                       (self.x + self.size / 2, self.y + self.size / 2), 
                       (self.x - self.size / 2, self.y + self.size / 2)]

        coordinates = utils.rotate_polygon([self.x, self.y], coordinates, self.angle)
        
        utils.draw_colored_polygon(surface, coordinates, self.color, zoom, translation, angle)

     
    def __del__(self):
        del self.metadata
