import os
import numpy as np
import pickle
import datetime
from agents.llm_policy import LLMPolicyAgent
from agents.mcts import StateNode, ActionNode
from utils import DictToObject, softmax
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import os
import re
import tqdm
import time

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

if os.getenv("USE_OPENAI_CUSTOM"):
    client = OpenAI(
        base_url=os.getenv("CUSTOM_BASE_URL"),
        api_key=os.getenv("CUSTOM_API_KEY")
    )
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
class LLMTransitionModel:
    def __init__(self, 
                 env_params,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/elevator_transition",
                 save_buffer_interval=10,
                 llm_model='gpt-4o-mini', 
                 debug=False):
        
        self.env_params = env_params
        
            
        self.system_prompt = open(self.env_params["system_prompt_path"], "r").read()
        self.extract_state_regex = self.env_params["extract_state_regex"]
        self.extract_regex_fallback = self.env_params["extract_regex_fallback"]
        self.llm_model =  os.getenv("CUSTOM_MODEL_ID") if os.getenv("USE_OPENAI_CUSTOM") else llm_model
        self.debug = debug
        
        self.prompt_buffer = {}
        if load_prompt_buffer_path is not None:
            #check file exists
            if os.path.exists(load_prompt_buffer_path):
                with open(load_prompt_buffer_path, "rb") as f:
                    print(f"Loading prompt buffer from {load_prompt_buffer_path}")
                    self.prompt_buffer = pickle.load(f)        
            else:
                print(f"Warning: Prompt buffer file {load_prompt_buffer_path} not found. Creating new buffer.")        
        
        self.prompt_buffer_save_path = f"{prompt_buffer_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.save_buffer_interval = save_buffer_interval
        self.call_count = 0    
        
    def get_next_state(self, state: str, action: str):
        '''
        Get the next state given the current state and action
        '''
        
        # construct user prompt
        user_prompt = "**Current State:**\n"
        user_prompt += state
        user_prompt += "\n**Action:**"
        user_prompt += action + "\n"
        
        response = self.query_llm(user_prompt)
        
        next_state, status = self.extract_state(response)
        
        return next_state, status
    
    def query_llm(self, user_prompt):
        '''
        Query the LLM with the user prompt
        '''
        
        if user_prompt in self.prompt_buffer:
            return self.prompt_buffer[user_prompt]
    
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        
        while True:
            try:
                response = client.chat.completions.create(model=self.llm_model, messages=messages)
                break
            except Exception as e:
                print(f"Error calling API: {e}, retrying...")
                time.sleep(1)
        
        # grab the content of the first choice (only one choice is returned)
        response = response.choices[0].message.content
        
        self.prompt_buffer[user_prompt] = response
        
        self.call_count += 1
        
        if self.call_count % self.save_buffer_interval == 0:
            with open(self.prompt_buffer_save_path, "wb") as f:
                print(f"Saving prompt buffer to {self.prompt_buffer_save_path}")
                pickle.dump(self.prompt_buffer, f)
        
        if self.debug:
            print(f"response:\n {response}")
        
        return response
    
    def extract_state(self, response: str):
        '''
        Extract the next state from the LLM response
        '''
        
        match = re.search(self.extract_state_regex, response, re.DOTALL | re.IGNORECASE)
        if match is not None:
            next_state = match.group(1)
            return next_state, "success"
        else:
            if self.debug:
                print("Warning: No match found, trying fallback regex...")
            
            for regex in self.extract_regex_fallback:
                match = re.search(regex, response, re.DOTALL | re.IGNORECASE)
                if match is not None:
                    next_state = match.group(1)
                    return next_state, "success on fallback regex"
            else:
                print("Error: No match found with fallback regex, using full response as next state")
                return response, "error"
        

class LLMZeroAgent:
    def __init__(self, env, cfg=None, debug=False):
        
        self.env = env  # env object is only needed for the state_to_text, and get_valid_actions methods, no steps needed
        
        self.cfg = {
            "env": "elevator",  # TO DO: implement for alfworld
            "mcts": {
                "num_simulations": 40,
                "c_puct": 500,    #should be proportional to the scale of the rewards 
                "gamma": 0.95,
                "max_depth": 100,   # setting this higher would have less of an effect because there is no rollout
                "backprop_T": 50,
            },
            "llm_policy": {
                "env_params": {
                    "system_prompt_path": "prompts/prompt_elevator_policy.txt",
                    "extract_action_regex": r"optimal action: (.*)",
                },
                "load_prompt_buffer_path": "prompt_buffer/elevator_20241107_065530.pkl", # update this path to the path of the saved prompt buffer
                "prompt_buffer_prefix": "prompt_buffer/elevator_policy",
                "save_buffer_interval": 10,
            } ,
            "llm_transition": {
                "env_params": {
                    "system_prompt_path": "prompts/prompt_elevator_transition.txt",
                    "extract_state_regex": r"next state:(.*?)```",
                    "extract_regex_fallback": [r"next state:(.*)"],
                },
                "load_prompt_buffer_path": None, # update this path to the path of the saved prompt buffer   
            }
        }
        
        if cfg is not None:
            self.cfg.update(cfg)
            
        self.args = self.cfg["mcts"]
        self.args = DictToObject(self.args)
            
        self.debug = debug
        
        # initialize policy
        self.policy = LLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_policy"])
        
    def act(self, state):
        '''
        Run the MCTS algorithm to select the best action
        '''
        
        state = self.env.state_to_text(state)
        root = self.build_state(state)
        
        for _ in tqdm.tqdm(range(self.args.num_simulations)):
            self.simulate(root)
            
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
            obs, reward, done, _, _ = self.env.step(best_action_node.action)
            
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
        rollout_rewards = []
        
        for _ in range(self.args.num_rollouts):
            checkpoint = self.env.checkpoint()
            rollout_reward = 0
            rollout_depth = 0
            tmp_depth = depth
            
            obs = current_state_node.state
            while not done and tmp_depth < self.args.max_depth:
                valid_actions = self.env.get_valid_actions(obs)
                action = np.random.choice(valid_actions)
                obs, reward, done, _, _ = self.env.step(action)
                tmp_depth += 1
                rollout_depth += 1
                rollout_reward += reward * self.args.gamma ** rollout_depth
                
            rollout_rewards.append(rollout_reward)
            self.env.restore_checkpoint(checkpoint)
            
        rollout_reward = np.mean(rollout_rewards)
            
        # Step 4: Backpropagation, update the Q values of the nodes in the trajectory
        current_action_node = best_action_node
        cumulative_reward = rollout_reward
        
        while current_action_node is not None:
            current_action_node.N += 1
            # current_action_node.Q += (cumulative_reward - current_action_node.Q) / current_action_node.N
            current_action_node.Rs.append(cumulative_reward)
            # softmax to prioritize actions with higher rewards
            best_action_node.Q = np.sum(np.array(best_action_node.Rs) * softmax(best_action_node.Rs, T=self.args.backprop_T))
            current_state_node = current_action_node.parent
            current_state_node.N += 1
            cumulative_reward = current_state_node.reward + self.args.gamma * cumulative_reward
            current_action_node = current_state_node.parent
            
        return cumulative_reward    #return not actually needed, just for debugging
            
            
    def build_state(self, state, reward=0, done=False, parent=None):
        valid_actions = self.env.get_valid_actions(state)
        state_node = StateNode(state, valid_actions, reward, done, parent)
        if self.policy is not None:
            distribution = self.policy.get_action_distribution(state)
            if isinstance(distribution, dict):
                distribution = list(distribution.values())
            state_node.children_probs = distribution
            # if self.debug:
            #     # print(f"State: {state}")
            #     print(f"Action distribution: {distribution}")
        
        return state_node
        
        
    def select_action_node(self, state_node, debug=False):
        '''
        Select the action with the highest UCB value
        '''
        
        best_ucb = -np.inf
        best_children = []
        best_children_prob = []
        EPS = 1e-6
        
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
                    
        best_children_prob = np.array(best_children_prob) / (np.sum(best_children_prob)+EPS)
        best_action_idx = np.argmax(best_children_prob)
        
        return best_children[best_action_idx]
    
    def select_action_node_greedily(self, state_node):
        '''
        Select the action with the most visits
        '''
        
        best_children = []
        best_children_prob = []
        most_visits = 0
        
        for i, child in enumerate(state_node.children):
            if self.debug:
                print(f"Action {child.action}: Q = {child.Q}, N = {child.N}")
            
            if child.N == most_visits:
                most_visits = child.N
                best_children.append(child)
                best_children_prob.append(state_node.children_probs[i] + child.Q)
            if child.N > most_visits:
                most_visits = child.N
                best_children = [child]
                best_children_prob = [state_node.children_probs[i] + child.Q]
                
        # in case of ties, return highest Q value + child prob
        best_children_prob = np.array(best_children_prob)
        best_child = best_children[np.argmax(best_children_prob)]
        best_action = best_child
                    
        return best_action
    
    