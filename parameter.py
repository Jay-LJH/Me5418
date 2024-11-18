import numpy as np

class custom_parameter:
    logger = True
    width =  96 #width of map
    height = 96 #height of map
    video_width = 1000 #video width in pygame
    video_height = 800 #video height in pygame
    random_seed = 42
    create_time =  5 #create a new box average each 10 seconds 
    expire_time = 10000 #expire a box after average 30 seconds
    destory_time = 0 #destory a box after expired 20 seconds
    sigma = 1 #sigma for normal distribution
    box_width = 2   #box width
    box_height = 2  #box height
    FPS = 50 #frame per second
    crash_distance = 5 #distance to crash
    crash_speed = 35 #speed to crash

    bg_color = np.array([102, 204, 102])
    grass_color = np.array([102, 230, 102])
    box_color = np.array([102, 102, 102])
    dest_color = np.array([255, 255, 255])
    scale = 6.0  # Track scale
    playfield = 2000 / scale  # Game over boundary
    zoom = 2.7
    max_shape_dim = 381

    closer_reward = 2 #reward for closer to box, this reward become higher when closer to box, negative if further
    step_reward = -0.1  #reward for each step
    crash_reward = -10  #reward for crash
    pickup_reward = 100 #reward for pickup each box
    reach_reward = 100  #reward for reach destination
    expire_reward = -10 #reward for expire box
    expire_reward_continuous = -0.01 #reward for expire box each frame
    action_reward = -0.5 #reward for each action

class training_parameter:
    wandb = True
    channel = 3
    net_size = 96
    matrix_size = custom_parameter.width * custom_parameter.height
    lr = 1e-5
    opti_eps=1e-8
    weight_decay=0
    max_step = 100000
    sync_windows = 32
    gamma = 0.95
    lam = 0.95
    clip_range = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 20
    num_envs = 1
    num_epochs = 8
    batch_size = 16
    save_interval = 5120
    num_CPU = 1
    num_GPU = 1