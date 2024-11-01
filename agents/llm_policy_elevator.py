import os
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import os

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from utils import state_to_text, action_txt_to_idx

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# TO DO: abstract out the LLM model and the environment specific code
class ElvatorLLMPolicyAgent:
    def __init__(self, 
                 device, 
                 llm_model='gpt-4o-mini', 
                 prompt_template_path="prompts/prompt_elevator_policy.txt", 
                 debug=False,
                 temp=2.0):
        self.device = device
        self.llm_model = llm_model
        self.cos_sim_model = SentenceTransformer('paraphrase-MiniLM-L12-v2').to(self.device)
        self.valid_actions_text = ["move", "open", "close", "nothing"]
        
        self.prompt_template = open(prompt_template_path, "r").read()
        
        self.params = \
            {
                "max_tokens": 10,
                "n": 1,
                "stop": [',', '.', '\n'],
                "logprobs": True,
                "top_logprobs": 20,
            }
            
        self.debug = debug
        self.temp = temp
        
        self.prompt_buffer = {}
        
        
    def query_llm(self, prompt):
        if prompt in self.prompt_buffer:
            return self.prompt_buffer[prompt]
        
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(model=self.llm_model, messages=messages, **self.params)
        
        res = response.choices[0].logprobs.content
        
        self.prompt_buffer[prompt] = res
        
        return res
        
    def get_cos_sim(self, txt_list_1, txt_list_2):
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
        
    def find_most_similar_action(self, action):
        cos_sim = self.get_cos_sim([action], self.valid_actions_text)
        action_idx = np.argmax(cos_sim)
        
        return action_idx
    
    def get_action_distribution(self, res):
        logprobs_obj = res[0].top_logprobs # list of logprobs info for the first token, .token and .logprob is what we want
        tokens = [x.token for x in logprobs_obj]
        logprobs = [x.logprob for x in logprobs_obj]
        probs = [np.exp(logprob) for logprob in logprobs]
        
        cos_sim = self.get_cos_sim(tokens, self.valid_actions_text)
        
        COS_SIM_THRESHOLD = 0.7
    
        similar_actions = []
        
        for cs in cos_sim:
            if np.max(cs) > COS_SIM_THRESHOLD:
                similar_actions.append(self.valid_actions_text[np.argmax(cs)])
            else:
                similar_actions.append(None)
                
        dist = {action: 0.0 for action in self.valid_actions_text}
        
        #get action distribution
        for i, action in enumerate(tokens):
            if similar_actions[i] is not None:
                dist[similar_actions[i]] += probs[i]
                
        #smooth distribution with temperature
        dist = {action: prob ** (1/self.temp) for action, prob in dist.items()}
        total_prob = sum(dist.values())
        
        dist = {action: prob/total_prob for action, prob in dist.items()}
        
        return dist     
        
    def act(self, state, greedy=False):
        state_txt = state_to_text(state)
        prompt = self.prompt_template.replace("{current_observation}", state_txt)
        
        res = self.query_llm(prompt)
        
        dist = self.get_action_distribution(res)
        
        if greedy:
            action = max(dist, key=dist.get)
        else:
            action = random.choices(list(dist.keys()), list(dist.values()))[0]
        
        if self.debug:
            print(f"State: {state_txt}")
            print(f"Action distribution: {dist}")
            print(f"Action: {action}")
        
        return action_txt_to_idx(action)
    
        
    