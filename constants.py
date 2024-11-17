import numpy as np

class constants:

    # Verbose switch
    LOGGER = True

    # Switch between partially observable and fully observable
    FULLY_OBSERVABLE = True

    # Map size
    WIDTH = 96
    HEIGHT = 96

    # Rendered zoom magnification
    ZOOM = 5

    # Rendered image size
    VIDEO_WIDTH = 320 
    VIDEO_HEIGHT = 200

    # Precision of Box2D world simulation, the larger the better
    VELOCITY_PRECISION = 10
    POSITION_PRECISION = 10

    # Default objects gen seed
    RANDOM_SEED = 40

    # Bleed so that objects does not generate outside environment
    BLEED = 16

    # Radian of half-circle
    PI = 3.1415927

    # Floating point strict equivalance threshold
    EQUIVALANCE_THRESHOLD = 0.01

    # Linear truck stop threshold
    STOP_THRESHOLD_LINEAR = 0.25

    # Angular truck stop threshold
    STOP_THRESHOLD_ANGULAR = 0.1

    # Destination size
    DEST_SIZE = 4

    # Cargo size and random generation
    CARGO_SIZE = 2

    # Generate at least 2 cargoes and maximum of 5 cargoes, uniform distribution.
    MIN_NUM_CARGOES = 2
    MAX_NUM_CARGOES = 5

    # Cargo creation has uniform distribution of minimum 30s and maximum 60s.
    MIN_CREATE_TIME = 30
    MAX_CREATE_TIME = 60
    
    # Cargo expiration has uniform distribution of minimum 30s and maximum 60s.
    MIN_EXPIRE_TIME = 30 
    MAX_EXPIRE_TIME = 60

    # Cargo needs 3s to be carried and 3s to be unloaded
    CARGO_LOAD_TIME = 3
    CARGO_UNLOAD_TIME = 3

    # Carried cargo render size
    CARRIED_CARGO_SIZE = 2

    # Environment rendering metadata
    FPS = 60

    # Render colors
    BG_COLOR = np.array([102, 204, 102])
    GRASS_COLOR = np.array([102, 230, 102])
    CARGO_COLOR = np.array([102, 102, 102])
    DEST_COLOR = np.array([255, 255, 255])
    
    # Environment/physics definitions
    MAX_TERMINATION = 120 # Maximum time out in seconds
    REACH_DISTANCE = 2 # Radius to be considered reaching cargo/destintation for load/unload
    
    # Reward structure, time outs and accidents lead to truncation directly 
    CRASH_REWARD = -50 # Penalty for crash with a cargo or border, accidents are strictly not allowed.
    CARGO_REACH_DEST_REWARD = 20 # Reward for cargo reach destination
    EXPIRE_REWARD = -10 # Penalty for expire box reach destination
    EXPIRE_REWARD_CONTINUOUS = -0.1 # Penalty for expire box on truck each second
    STOP_REWARD = 0.02 # Reward for stop per second, energy saving

    # Rendering rectangles optimization utility
    MAX_SHAPE_DIM = 64

    # A3C/Neural network properties, avoid confusion of 'training parameters' and 'trained parameters'
    '''
    # The number of environments we are synchronously running
    NUM_ENVS = 1

    # Max number of episodes in which the agent is trained
    MAX_EPISODES = 100000
    '''
