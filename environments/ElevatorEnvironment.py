import gym
from pyRDDLGym.Elevator import Elevator
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
import copy

class ElevatorEnvironment(gym.Wrapper):
    '''
    wrapper for Elevator environment
    '''
    def __init__(self):
        env = Elevator(instance=5)
        super(ElevatorEnvironment, self).__init__(env)
        
    def reset(self, seed=None):
        state = self.base_env.reset(seed)
        return state
    
    def checkpoint(self):
        '''
        return a checkpoint of the current environment state
        '''
        orig_subs = copy.deepcopy(self.base_env.sampler.subs)
        orig_H = copy.deepcopy(self.base_env.currentH)
        done = self.base_env.done
        
        checkpoint = (orig_subs, orig_H, done)
        
        return checkpoint
        
    def restore_checkpoint(self, checkpoint):
        '''
        restore the environment to a previous state
        '''
        orig_subs, orig_H, done = checkpoint
        
        self.base_env.sampler.subs = orig_subs
        self.base_env.currentH = orig_H
        self.base_env.done = done           
        
    def step(self, action):
        action = self.map_action(action)
        cont_action = self.disc2action(action)
        next_state, reward, done, info = self.base_env.step(cont_action)
        return next_state, reward, done, info
    
    def get_valid_actions(self, state=None):
        return range(4) # 0: nothing, 1: move, 2: close door, 3: open door
    
    def get_valid_actions_text(self, state=None):
        return ["nothing", "move", "close door", "open door"]
    
    def state_to_text(self, state_dict):
        state_text = ""
        
        num_person_waiting = [None for _ in range(5)]
        num_in_elavator = None
        door_state = None
        direction = None
        current_floor = None
        
        for feature, value in state_dict.items():
            if "num-person-waiting" in feature:
                num_person_waiting[int(feature[-1])] = value
            if "elevator-at-floor" in feature and value == True:
                current_floor = int(feature[-1]) + 1
            if feature == "elevator-dir-up___e0":
                direction = "up" if value == True else "down"
            if feature == "elevator-closed___e0":
                door_state = "closed" if value == True else "open"
            if feature == "num-person-in-elevator___e0":
                num_in_elavator = value
                
        for i in range(2, 6):
            num_waiting = num_person_waiting[i - 1]
            state_text += f"People waiting at floor {i}: {num_waiting}\n"                
            
        state_text += f"Elevator at floor {current_floor}.\n"
        state_text += f"There are {num_in_elavator} people in the elevator.\n"
        state_text += f"Elevator is moving {direction}.\n"
        state_text += f"Elevator door is {door_state}.\n"
        
        return state_text
    
    def action_to_text(self, action):
        if action == 0:
            return "nothing"
        elif action == 1:
            return "move"
        elif action == 2:
            return "close door"
        elif action == 3:
            return "open door"
        else:
            raise ValueError(f"Invalid action {action}")
    
    def action_txt_to_idx(self, action_txt):
        if action_txt == "nothing":
            return 0
        elif action_txt == "move":
            return 1
        elif action_txt == "close door":
            return 2
        elif action_txt == "open door":
            return 3
        else:
            raise ValueError(f"Invalid action text {action_txt}")
    
    def map_action(self, action):
        if action == 0:
            return 0
        elif action == 1:
            return 1
        elif action == 2:
            return 3
        elif action == 3:
            return 5
        else:
            raise ValueError(f"Invalid action {action}")