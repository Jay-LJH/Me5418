import numpy as np

class constants:

    # Verbose switch
    LOGGER = True 

    # Switch between partially observable and fully observable
    FULLY_OBSERVABLE = False

    # Map size
    WIDTH = 96
    HEIGHT = 96

    # Rendered zoom magnification
    ZOOM = 5

    # Rendered image size
    VIDEO_WIDTH = 320 
    VIDEO_HEIGHT = 200

    # Default objects gen seed
    RANDOM_SEED = 40

    # Bleed so that objects does not generate outside environment
    BLEED = 16

    # Radian of half-circle
    PI = 3.1415927

    # Floating point equivalance threshold
    THRESHOLD = 0.01

    # Cargo size and random generation
    CARGO_WIDTH = 2
    CARGO_HEIGHT = 2

    # Generate at least 2 cargoes and maximum of 5 cargoes, uniform distribution.
    MIN_NUM_CARGOES = 2
    MAX_NUM_CARGOES = 5

    # Cargo creation has uniform distribution of minimum 30s and maximum 60s.
    CREATE_TIME_MIN = 30
    CREATE_TIME_MAX = 60
    
    # Cargo expiration has uniform distribution of minimum 30s and maximum 60s.
    EXPIRE_TIME_MIN = 30 
    EXPIRE_TIME_MAX = 60

    # Environment rendering metadata
    FPS = 60

    # Render colors
    BG_COLOR = np.array([102, 204, 102])
    GRASS_COLOR = np.array([102, 230, 102])
    BOX_COLOR = np.array([102, 102, 102])
    DEST_COLOR = np.array([255, 255, 255])
    
    # Environment/physics definitions
    MAX_TRUNCATION = 120 # Maximum time out in seconds
    REACH_DISTANCE = 1 # Radius to be considered reaching cargo/destintation for load/unload
    CRASH_SPEED = 0.5 # Speed to crash cargo
    
    # Reward structure, time outs and accidents lead to truncation directly 
    CRASH_REWARD = -50 # Penalty for crash with a cargo or border, accidents are strictly not allowed.
    CARGO_REACH_DEST_REWARD = 20 # Reward for cargo reach destination
    EXPIRE_REWARD = -10 # Penalty for expire box
    EXPIRE_REWARD_CONTINUOUS = -0.1 # Penalty for expire box each second
    STOP_REWARD = 0.02 # Reward for stop per second, energy saving

    # Rendering rectangles optimization utility
    MAX_SHAPE_DIM = 64
