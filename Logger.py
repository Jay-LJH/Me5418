from constants import constants

class logger:
    
    def __init__(self):
        self.log = constants.LOGGER
        
    def log(message):
        if logger.log:
            print(message)