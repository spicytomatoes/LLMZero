import numpy as np
import random
random.seed(42)

class RandomAgent:
    def __init__(self, env):
        self.env = env
        
    def act(self, state):
        valid_actions = self.env.get_valid_actions(state)
        return random.choice(valid_actions)
    