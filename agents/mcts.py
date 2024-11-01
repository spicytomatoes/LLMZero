import numpy as np
from utils import DictToObject
import tqdm
import json

class StateNode:
    def __init__(self, state, valid_actions, reward = 0, done = False, parent=None):
        self.state = state
        self.valid_actions = valid_actions
        self.reward = reward
        self.done = done
        self.parent = parent   # link back to previous action node
        
        self.children = []
        for action in valid_actions:
            self.children.append(ActionNode(action, self))
        
        self.children_probs = np.ones(len(valid_actions)) / len(valid_actions)    
        
        self.N = 1 # number of times this state node has been visited, node created implies one visit
        
    def get_unique_id(self):
        #convert state to a unique string
        data = {key: (value.tolist() if isinstance(value, np.ndarray) 
                      else int(value) if isinstance(value, np.integer) 
                      else float(value) if isinstance(value, np.floating) 
                      else bool(value) if isinstance(value, np.bool_)
                        else value)
        for key, value in self.state.items()}
        return json.dumps(data)

class ActionNode:
    '''
    Action nodes are like chance nodes in MDPs, they represent the possible outcomes of taking an action
    '''
    def __init__(self, action, parent):
        self.action = action
        self.parent = parent # link back to previous state node
        self.N = 0 # number of times this action node has been visited
        self.Q = 0 # average reward of this action node
        # self.Rs = [] # list of rewards received from this action node
        self.children = {}  # dictionary of next state nodes id: StateNode
        
class MCTSAgent:
    def __init__(self,
                 env,
                 use_llm=False,
                 args=None,
                 debug=False):
        '''
        env: pyRDDLGym environment object, must be discrete action space
        use_llm: bool, whether to use LLM policy to guide the search
        args: dict, additional arguments to override default values
        '''
        
        self.env = env
        self.valid_actions = env.get_valid_actions()
        self.use_llm = use_llm
        
        self.args = {
            "num_simulations": 1000,
            "c_puct": 100,    #should be proportional to the scale of the rewards
            "gamma": 0.99,
            "max_depth": 32,
        }
        
        if args is not None:
            self.args.update(args)
            
        self.args = DictToObject(self.args)
            
        self.debug = debug
        
    def act(self, state):
        '''
        Run the MCTS algorithm to select the best action
        '''
        
        root = self.build_state(state)
        
        for _ in tqdm.tqdm(range(self.args.num_simulations)):
            self.env.begin_search()
            self.simulate(root)
            self.env.end_search()
            
        best_action = self.select_action_node_greedily(root).action
        
        return best_action
            
        
    def simulate(self, state_node):
        '''
        Simulate a trajectory from the current state node
        '''
        
        current_state_node = state_node
        depth = 0
        
        # Step 1: Selection, traverse down the tree until a leaf node is reached
        while not current_state_node.done and depth < self.args.max_depth:
            
            best_action_node = self.select_action_node(state_node)
            obs, reward, done, _ = self.env.step(best_action_node.action)
            
            reward = reward * self.args.gamma
            next_state_node = self.build_state(obs, reward, done, current_state_node)
            next_state_id = next_state_node.get_unique_id()
            
            if next_state_id not in best_action_node.children.keys():
                # Step 2: Expansion, add the new state node to the tree
                best_action_node.children[next_state_id] = next_state_node
                break
            else:
                current_state_node = best_action_node.children[next_state_id]
                current_state_node.reward = (current_state_node.reward * current_state_node.N + reward) / (current_state_node.N + 1)
                depth += 1
                
        # Step 3: Rollout, simulate the rest of the trajectory using a random policy
        rollout_reward = 0
        rollout_depth = 0
        while not done and depth < self.args.max_depth:
            action = np.random.choice(self.valid_actions)
            obs, reward, done, _ = self.env.step(action)
            depth += 1
            rollout_depth += 1
            rollout_reward += reward * self.args.gamma ** rollout_depth
            
        # Step 4: Backpropagation, update the Q values of the nodes in the trajectory
        current_action_node = best_action_node
        cumulative_reward = rollout_reward
        
        while current_action_node is not None:
            current_action_node.N += 1
            current_action_node.Q += (cumulative_reward - current_action_node.Q) / current_action_node.N
            current_state_node = current_action_node.parent
            current_state_node.N += 1
            cumulative_reward = current_state_node.reward + self.args.gamma * cumulative_reward
            current_action_node = current_state_node.parent
            
        return cumulative_reward    #return not actually needed, just for debugging
            
            
    def build_state(self, state, reward=0, done=False, parent=None):
        
        state_node = StateNode(state, self.valid_actions, reward, done, parent)
        if self.use_llm:
            #suppose to set the children probs based on the LLM policy
            raise NotImplementedError("LLM policy not implemented yet")  
        
        return state_node
        
        
    def select_action_node(self, state_node, debug=False):
        '''
        Select the action with the highest UCB value
        '''
        
        best_ucb = -np.inf
        best_children = []
        best_children_prob = []
        
        for i in range(len(state_node.children)):
            child = state_node.children[i]
            child_prob = state_node.children_probs[i]
            
            #PUCT formula
            c_puct = self.args.c_puct * len(state_node.children) # multiply to offset child_prob's effect
            ucb = child.Q + c_puct * child_prob * np.sqrt(np.log(state_node.N) / (1 + child.N))
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_children = [child]
                best_children_prob = [child_prob]
            elif ucb == best_ucb:
                best_children.append(child)
                best_children_prob.append(child_prob)
                
        if debug:
            for i, child in enumerate(state_node.children):
                if child.N > 0:
                    print(f"Action {child.action}: Q = {child.Q}, N = {child.N}, prob = {state_node.children_probs[i]}")
                    
        best_children_prob = np.array(best_children_prob) / np.sum(best_children_prob)
        best_action_idx = np.argmax(best_children_prob)
        
        return best_children[best_action_idx]
    
    def select_action_node_greedily(self, state_node):
        '''
        Select the action with the highest average reward
        '''
        
        best_reward = -np.inf
        best_action = None
        
        for child in state_node.children:
            if self.debug:
                print(f"Action {child.action}: Q = {child.Q}, N = {child.N}")
            
            if child.N > 0:
                reward = child.Q
                if reward > best_reward:
                    best_reward = reward
                    best_action = child
                    
        return best_action
        
        
        
        