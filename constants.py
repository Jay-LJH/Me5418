import numpy as np

class constants:
    
    logger = True # verbose switch
    width = 96 # width of map
    height = 96 # height of map
    video_width = 1000 # video width in pygame
    video_height = 800 # video height in pygame
    random_seed = 42
    create_time = 10 #create a new box average each 10 seconds 
    expire_time = 30 #expire a box after average 30 seconds
    sigma = 1 #sigma for normal distribution
    box_width = 2   #box width
    box_height = 2  #box height
    FPS = 50 #frame per second
    destory_time = 20 #destory a box after expired 20 seconds
    crash_distance = 2 #distance to crash
    crash_speed = 1 #speed to crash

    bg_color = np.array([102, 204, 102])
    grass_color = np.array([102, 230, 102])
    box_color = np.array([102, 102, 102])
    dest_color = np.array([255, 255, 255])
    scale = 6.0  # Track scale
    playfield = 2000 / scale  # Game over boundary
    zoom = 2.7
    max_shape_dim = 381

    step_reward = -0.1 #reward for each step
    crash_reward = -10 #reward for crash
    pickup_reward = 100 #reward for pickup each box
    reach_reward = 100 #reward for reach destination
    expire_reward = -10 #reward for expire box
    expire_reward_continuous = -0.01 #reward for expire box each frame
