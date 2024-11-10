import numpy as np
from utils import elevator_estimate_value

class BFSAgent:
    def __init__(self, env, max_depth=10):
        self.env = env
        self.max_depth = max_depth
        
    def search(self, state):
        valid_actions = self.env.get_valid_actions(state)
        
        best_value = -np.inf
        best_action = None
        
        for action in valid_actions:
            checkpoint = self.env.checkpoint()
            value = self.evaluate(state, action, self.max_depth)
            self.env.restore_checkpoint(checkpoint)
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
    
    def evaluate(self, state, action, depth):
        next_state, reward, done, _, _ = self.env.step(action)
        
        if done or depth == 1:
            return reward + elevator_estimate_value(next_state)
        
        valid_actions = self.env.get_valid_actions(next_state)
        
        best_value = -np.inf
        
        for action in valid_actions:
            checkpoint = self.env.checkpoint()
            value = self.evaluate(next_state, action, depth - 1)
            self.env.restore_checkpoint(checkpoint)
            if value > best_value:
                best_value = value
                
        return reward + best_value
        
        
        
    