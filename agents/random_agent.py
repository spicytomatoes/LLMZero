import numpy as np
import random

class RandomAgent:
    def __init__(self):
        self.valid_actions = [0, 1, 2, 3, 4, 5]
        
    def act(self, state):
        return random.choice(self.valid_actions)
    