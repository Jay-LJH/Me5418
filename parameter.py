import numpy as np

class custom_parameter:
    logger = True
    width =  96 #width of map
    height = 96 #height of map
    render_height = 96 # height in rgb_array render mode
    render_width = 96 # width in rgb_array render mode
    video_width = 1000 #video width in pygame
    video_height = 800 #video height in pygame
    random_seed = 42 # random seed
    create_time =  5 #create a new box average each 10 seconds 
    expire_time = 10000 #expire a box after average 30 seconds
    destory_time = 0 #destory a box after expired 20 seconds
    sigma = 1 #sigma for normal distribution
    box_width = 2   #box width
    box_height = 2  #box height
    FPS = 50 #frame per second
    pickup_distance = 5 #distance to crash
    pickup_speed = 30 #speed to crash
    max_time = 60 #max time for each episode

    bg_color = np.array([102, 204, 102]) # background color
    grass_color = np.array([102, 230, 102]) # grass color
    box_color = np.array([102, 102, 102]) # box color
    dest_color = np.array([255, 255, 255]) # destination color
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
    stay_reward = -1 #reward for stay in the same place
    action_reward = -0.5 #reward for each action
    out_reward = -10 #reward for out of boundary
    explore_reward = 10 #reward for explore

class training_parameter:
    retrain = False # retrain the model. if set retrain, need to set the retrain path at driver.py
    wandb = True # use wandb
    channel = 3 # channel of input
    vector_len = 8 #length of the input vector
    net_size = 96 # size of network
    lr = 1e-5 # learning rate
    opti_eps=1e-8 # optimizer epsilon
    weight_decay=0 # weight decay
    max_step = 1e7 # max step
    sync_windows = 32 # sync windows for async training
    gamma = 0.95 # gamma for reward
    lam = 0.95 # lambda for gae
    clip_range = 0.2 # clip range for ppo
    entropy_coef = 0.01 # entropy coefficient
    value_coef = 0.5 # value coefficient
    max_grad_norm = 20 # max grad norm
    num_envs = 1 # number of environments
    num_epochs = 8 # number of epochs
    batch_size = 16  # batch size
    save_interval = 5120 # save model interval
    num_CPU = 1 # number of CPU
    num_GPU = 1 # number of GPU