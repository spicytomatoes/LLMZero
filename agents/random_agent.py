import numpy as np
import random

class RandomAgent:
    def __init__(self):
        self.valid_actions = [0, 1, 2, 3]   # 0: nothing, 1: move, 2: close door, 3: open door
        
    def act(self, state):
        return random.choice(self.valid_actions)
    