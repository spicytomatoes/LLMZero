import numpy as np
import random

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.valid_actions = env.get_valid_actions()
        
    def act(self, state):
        return random.choice(self.valid_actions)
    