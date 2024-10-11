class custom_parameter:
    width =  96 #width of map
    height = 96 #height of map
    video_width = 600 #video width in pygame
    video_height = 400 #video height in pygame
    random_seed = 42
    create_time = 10 #create a new box average each 10 seconds 
    expire_time = 30 #expire a box after average 30 seconds
    sigma = 1 #sigma for normal distribution
    box_width = 0.2   #box width
    box_height = 0.2  #box height
    box_color = (0, 255, 0) #box color
    FPS = 50 #frame per second
    destory_time = 20 #destory a box after expired 20 seconds
    crash_distance = 0.2 #distance to crash

    step_reward = -0.1 #reward for each step
    crash_reward = -10 #reward for crash
    pickup_reward = 100 #reward for pickup each box
    reach_reward = 100 #reward for reach destination
    expire_reward = -10 #reward for expire box
    expire_reward_continuous = -0.01 #reward for expire box each frame
