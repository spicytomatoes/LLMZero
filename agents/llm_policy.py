import os
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import os
import re

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# TO DO: abstract out the LLM model and the environment specific code
class LLMPolicyAgent:
    def __init__(self, 
                 env,
                 device, 
                 llm_model='gpt-4o-mini', 
                 env_params=None,
                 api_params=None,
                 debug=False,
                 temp=1.0   #smoothing factor for action distribution
                ):
        
        self.env = env
        self.device = device
        self.llm_model = llm_model
        self.cos_sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)
        
        self.env_params = {
            "system_prompt_path": "prompts/prompt_elevator_policy.txt",
            "extract_action_regex": r"optimal action: (.*)",
        }
        
        self.api_params = {
            "n": 10,
            "logprobs": True,
        }
        
        #override default values
        if env_params is not None:
            self.env_params.update(env_params)
        if api_params is not None:
            self.api_params.update(api_params)
            
            
        self.system_prompt = open(self.env_params["system_prompt_path"], "r").read()
        self.extract_action_regex = self.env_params["extract_action_regex"]
        self.debug = debug
        self.temp = temp
        
        self.prompt_buffer = {}
        
        
    def act(self, state, greedy=True):
        action_dist = self.get_action_distribution(state)
        
        if greedy:
            action = np.argmax(action_dist)
        else:
            action = np.random.choice(len(action_dist), p=action_dist)                
        
        return action
    
    def get_action_distribution(self, state):
        state_text = self.env.state_to_text(state)
        valid_actions_text = self.env.get_valid_actions_text(state)
        
        user_prompt = "**State**\n" + state_text
        
        messages, probs = self.query_llm(user_prompt)
        
        dist = self._get_action_distribution(messages, probs, valid_actions_text)
            
        if self.debug:
            print(f"State: {state_text}")
            print(f"Action distribution: {dist}")
            
        return dist
    
    def _get_action_distribution(self, messages, probs, valid_actions_text):
        
        action_samples = [self.extract_action_from_message(message) for message in messages]
        
        cos_sim = self.compute_cos_sim(action_samples, valid_actions_text)
        
        COS_SIM_THRESHOLD = 0.7
    
        similar_actions = []
        
        for i, cs in enumerate(cos_sim):
            if np.max(cs) > COS_SIM_THRESHOLD:
                similar_actions.append(valid_actions_text[np.argmax(cs)])
            else:
                if self.debug:
                    print(f"Warning: No similar action found for action {action_samples[i]}")
                similar_actions.append(None)    # do not consider this action
                
        action_probs = np.zeros(len(valid_actions_text))
        
        for i, action in enumerate(similar_actions):
            if action is not None:
                action_idx = self.env.action_txt_to_idx(action)
                action_probs[action_idx] += probs[i]
                
        action_probs = action_probs ** (1/self.temp)
        action_probs /= np.sum(action_probs)
        
        return action_probs
    
    def query_llm(self, user_prompt):
        if user_prompt in self.prompt_buffer:
            outputs, choice_probs = self.prompt_buffer[user_prompt]
            return outputs, choice_probs
        
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        
        # generates n possible completions, n in api_params
        response = client.chat.completions.create(model=self.llm_model, messages=messages, **self.api_params)
        
        # n messages
        return_msgs = [choice.message.content for choice in response.choices]
        # list of logprobs objects
        choice_logprobs_list = [choice.logprobs.content for choice in response.choices]

        # log probabilities of each choice
        choice_logprobs = np.zeros(len(choice_logprobs_list))   # to be summed and exponentiated
        
        for i, logprobs in enumerate(choice_logprobs_list):
            token_logprobs = [obj.logprob for obj in logprobs]
            choice_logprobs[i] = np.sum(token_logprobs)
        
        # convert to probabilities
        # subtracting max for numerical stability
        choice_probs = choice_logprobs - np.max(choice_logprobs)
        # clip to avoid overflow
        choice_probs = np.clip(choice_probs, -100, 0)
        choice_probs = np.exp(choice_probs)
        choice_probs /= np.sum(choice_probs)
        
        self.prompt_buffer[user_prompt] = (return_msgs, choice_probs)
        
        return return_msgs, choice_probs
        
    def compute_cos_sim(self, txt_list_1, txt_list_2):
        '''
        args:
            txt_list_1: list of strings
            txt_list_2: list of strings
        returns:
            similarities: pytorch tensor of shape (len(txt_list_1), len(txt_list_2))
        '''
        text_embedding = self.cos_sim_model.encode(txt_list_1, convert_to_tensor=True)
        action_embeddings = self.cos_sim_model.encode(txt_list_2, convert_to_tensor=True)
        cos_sim = st_utils.pytorch_cos_sim(text_embedding, action_embeddings)
        cos_sim = cos_sim.cpu().numpy()
        
        return cos_sim
        
    # currently not used
    def find_most_similar_action(self, action, valid_actions_text):
        cos_sim = self.compute_cos_sim([action], valid_actions_text)
        action_idx = np.argmax(cos_sim)
        
        return action_idx
    
    def extract_action_from_message(self, message):
        # assuming the pattern "Optimal action: <action>" where action is text
        match = re.search(self.extract_action_regex, message, re.IGNORECASE)
        
        if match:
            return match.group(1)
        else:
            if self.debug:
                print(f"Warning: No action found in message {message}")
            return "invalid action"
        
    
    
        
    