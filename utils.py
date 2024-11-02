import re
import gym
from pyRDDLGym.Elevator import Elevator
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
import copy
import numpy as np

class ElevatorEnvWrapper(gym.Wrapper):
    '''
    wrapper for Elevator environment to enable reseting to a specific state and other features
    '''
    def __init__(self, env):
        super(ElevatorEnvWrapper, self).__init__(env)
        
        
    def reset(self, seed=None):
        state = self.base_env.reset(seed)
        return state
    
    def checkpoint(self):
        orig_subs = copy.deepcopy(self.base_env.sampler.subs)
        orig_H = copy.deepcopy(self.base_env.currentH)
        done = self.base_env.done
        
        checkpoint = (orig_subs, orig_H, done)
        
        return checkpoint
        
    def restore_checkpoint(self, checkpoint):
        orig_subs, orig_H, done = checkpoint
        
        self.base_env.sampler.subs = orig_subs
        self.base_env.currentH = orig_H
        self.base_env.done = done           
        
    def step(self, action):
        action = self.map_action(action)
        cont_action = self.disc2action(action)
        next_state, reward, done, info = self.base_env.step(cont_action)
        return next_state, reward, done, info
    
    def get_valid_actions(self):
        return range(4) # 0: nothing, 1: move, 2: close door, 3: open door
    
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
    
def make_elevator_env(env_instance=5):
    env = Elevator(instance=env_instance)
    env = ElevatorEnvWrapper(env)
    return env

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


# TO DO: refactor relevant functions into ElevatorEnvWrapper
def state_to_text(state_dict):
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
            
    state_text += f"There are "
    flag = False
    for i in range(5):
        if num_person_waiting[i] > 0:
            state_text += f"{num_person_waiting[i]} people waiting at floor {i+1}. "
            flag = True
            
    if not flag:
        state_text += "no one waiting at any floor."
    state_text += "\n"
        
    state_text += f"Elevator at floor {current_floor}.\n"
    state_text += f"There are {num_in_elavator} people in the elevator.\n"
    state_text += f"Elevator is moving {direction}.\n"
    state_text += f"Elevator door is {door_state}.\n"
    
    return state_text

def action_txt_to_idx(action_txt):
        if action_txt == "move":
            return 1
        elif action_txt == "open":
            return 3
        elif action_txt == "close":
            return 2
        elif action_txt == "nothing":
            return 0
        else:
            raise ValueError(f"Invalid action text {action_txt}")
        
def action_to_text(action):
    if action == 0 or action == 2 or action == 4:
        return "nothing"
    elif action == 1:
        return "move"
    elif action == 3:
        return "close door"
    elif action == 5:
        return "open door"
    else:
        raise ValueError(f"Invalid action {action}")
    
def env_state_2_rddl_state(env_state):
    '''
    Convert the environment state to rddl state for restting to a specific state.
    Will probably bug out for other environments. (currently testing on Elevator)
    '''
    rddl_state_tmp = {}
    
    for features, value in env_state.items():
        feature_type = re.search(r"(.+?)___", features).group(1)
        feature_id = re.search(r"\d+$", features).group()
        feature_id = int(feature_id)
        
        if feature_type not in rddl_state_tmp:
            rddl_state_tmp[feature_type] = []
            
        rddl_state_tmp[feature_type].append((feature_id, value))
        
    #sort the features
    rddl_state = {}
    
    for feature_type, feature_list in rddl_state_tmp.items():
        feature_list.sort(key=lambda x: x[0])
        feature_list = [x[1] for x in feature_list]
        rddl_state[feature_type] = feature_list            
        
    return rddl_state

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y