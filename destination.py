from constants import constants
from utils import utils
from copy import copy

# The destination entity
# Check loaded trucks, when loaded truck reach at low speed, set stop status for truck for 3s 
# 
class destination:

    # Create the destination at specified location.
    def __init__(self, metadata):

        self.metadata = copy(metadata)
        self.x = metadata[0]
        self.y = metadata[1]
        self.angle = metadata[2]
        self.size = constants.DEST_SIZE
        self.color = constants.DEST_COLOR

    # For the environment to update observation. Never gets updated for our problem.
    def collect_observation(self):
        return self.metadata

    # Draw the destination on a given surface.
    def draw(self, surface, zoom, trans, angle):
        
        coordinates = [(self.x - self.size / 2, self.y - self.size / 2), 
                       (self.x + self.size / 2, self.y - self.size / 2), 
                       (self.x + self.size / 2, self.y + self.size / 2), 
                       (self.x - self.size / 2, self.y + self.size / 2)]

        coordinates = utils.rotate_polygon([self.x, self.y], coordinates, self.angle)
        
        utils.draw_colored_polygon(surface, coordinates, self.color, zoom, trans, angle)

    # Destination entity has nothing to do.
    def step(self, timedelta):
        pass

    def __del__(self):
        del self.metadata
        