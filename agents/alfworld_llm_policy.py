import numpy as np

from agents.llm_policy import LLMPolicyAgent

class ALFWorldLLMPolicyAgent(LLMPolicyAgent):
    def __init__(self, 
                 env,
                 device, 
                 llm_model='gpt-4o-mini', 
                 env_params=None,
                 api_params=None,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/alfworld",
                 save_buffer_interval=100,
                 debug=False,
                 temp=1.0   #smoothing factor for action distribution
                ):

        custom_env_params = {
            "system_prompt_path": "prompts/prompt_alfworld_policy.txt",
        }
        if env_params is not None:
            custom_env_params.update(env_params)

        super().__init__(env, device, llm_model, custom_env_params, api_params, load_prompt_buffer_path, prompt_buffer_prefix, save_buffer_interval, debug, temp)

    def act(self, state, greedy=True):
        int_action = super().act(state, greedy)
        text_action = state['valid_actions'][int_action]
        return text_action
    
    def get_action_distribution(self, state):
        state_text = self.env.state_to_text(state)
        valid_actions_text = self.env.get_valid_actions_text(state)
        goal_text = self.env.goal_to_text()

        user_prompt = f'''
        **Task**: {goal_text}
        **State**: {state_text}
        **Valid Actions**: {valid_actions_text}
        '''

        messages, probs = self.query_llm(user_prompt)
        
        dist = self._get_action_distribution(messages, probs, valid_actions_text)
            
        if self.debug:
            state_text = self.env.state_to_text(state)
            print(f"State: {state_text}")
            print(f"Action distribution: {dist}")
            
        return dist

    # TODO: Abstract most of this away, only leave change at bottom
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
                action_idx = self.env.action_txt_to_idx(action, valid_actions_text) # ONLY DIFFERENCE
                action_probs[action_idx] += probs[i]
                
        action_probs = action_probs ** (1/self.temp)
        action_probs /= np.sum(action_probs + 1e-10)
        
        return action_probs
    
    


