from copy import copy

from constants import constants

class logger:
    
    def __init__(self):
        
        self.buffer = []
        
    def log(self, message):
        
        self.buffer.append(message)

    def output(self):
        
        for message in self.buffer:
            print(message)
        
        del self.buffer
        self.buffer = []