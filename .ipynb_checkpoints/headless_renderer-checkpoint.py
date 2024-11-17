import matplotlib.pyplot as plt
import gymnasium
from IPython import display
from constants import constants
from logger import logger
from copy import copy

import matplotlib.animation as animation

'''
The save video module is from
https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
However, the module is not recommended since it is very poorly optimized.
Takes a long time even recording a single simulation.
'''

class headless_renderer:

    # Creation of a renderer relative to a given environment 
    # The environment has already set to rgb_array mode in the new version of gymnasium
    def __init__(self, env, create_video = False):
        
        # Need to save environment to call renderer
        self.env = env

        # Whether to create video
        self._create_video = create_video

        # Create frame buffer
        self._frame_buffer = []

        # Create plot figure
        self._fig = plt.figure()
        
        # Creation of plot
        self._plot = plt.imshow(env.render())

        # Turn off axis display
        plt.axis('off')

        # Animation object, used in creating video
        self._ani = None

    # Update display from the environment
    def update_display(self, logger = None):

        # Get observation plot data
        render = self.env.render()

        # No need to set rgb_array mode here
        self._plot.set_data(render)

        # Save frame
        if self._create_video:
            self._frame_buffer.append(render)

        # Display the plot
        display.display(plt.gcf())

        # Output all the information needed for debugging purposes
        if constants.LOGGER and logger is not None:
            logger.output()

        # Clear to play the next frame
        display.clear_output(wait=True)
        
    # Save video up to now, nothing if not cached.
    def save_video(self, filename):
        
        if self._create_video:
            self._ani = animation.ArtistAnimation(self._fig, 
                [[plt.imshow(frame, animated = True)] for frame in self._frame_buffer], 
                interval = 1000 / (self.env.metadata["FPS"] if "FPS" in self.env.metadata else constants.FPS))
            self._ani.save(filename)

    # Release frame buffer memory
    def __del__(self):

        del self._frame_buffer
