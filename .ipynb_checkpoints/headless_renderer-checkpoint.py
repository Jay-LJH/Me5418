import matplotlib.pyplot as plt
import gymnasium
from IPython import display

'''
It seems not possible to save as gif locally since the solution online is not for headless machines
Adopted from github: https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server
Render the environment into gif so we can do headless.

import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

env = gym.make('Breakout-v0')
env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
for _ in range(100):
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = env.action_space.sample()
    env.step(action)
'''

class headless_renderer:

    # Creation of a renderer relative to a given environment 
    # The environment has already set to rgb_array mode in the new version of gymnasium
    def __init__(self, env):
        
        # Need to save environment to call renderer
        self.env = env
        
        # Creation of plot
        self.plot = plt.imshow(env.render())

        # Turn off axis display
        plt.axis('off')

    # Update display from the environment
    def update_display(self):

        # No need to set rgb_array mode here
        self.plot.set_data(self.env.render())

        # Display the plot
        display.display(plt.gcf())

        # Clear to play the next frame
        display.clear_output(wait=True)

        

