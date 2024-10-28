import numpy as np
from constants import constants

class training_parameter:
    channel = 3
    net_size = 96
    matrix_size = constants.WIDTH * constants.HEIGHT
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