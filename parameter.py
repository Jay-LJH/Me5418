class custom_parameter:
    state_width = 96 #state width
    state_height = 96 #state height
    map_width = 600 #map width
    map_height = 400 #map height
    video_width = 600 #video width
    video_height = 400 #video height
    random_seed = 42
    create_time = 10 #create a new box average each 10 seconds 
    expire_time = 30 #expire a box after average 30 seconds
    sigma = 1 #sigma for normal distribution
    pickup_reward = 100 #reward for pickup each box
    reach_reward = 100 #reward for reach destination
    box_width = 10   #box width
    box_height = 10  #box height
    box_color = (0, 255, 0) #box color
    crash_reward = -10 #reward for crash

