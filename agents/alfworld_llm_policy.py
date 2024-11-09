import numpy as np
import time
from agents.llm_policy import LLMPolicyAgent

class ALFWorldLLMPolicyAgent(LLMPolicyAgent):
    def __init__(self, 
                 env,
                 device, 
                 llm_model='gpt-4o-mini', 
                #  llm_model='qwen2.5:32b',
                 env_params=None,
                 api_params=None,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/alfworld",
                 save_buffer_interval=100,
                 debug=False,
                 temp=1.0   #smoothing factor for action distribution
                ):
        
        super().__init__(env, device, llm_model, env_params, api_params, load_prompt_buffer_path, prompt_buffer_prefix, save_buffer_interval, debug, temp)

    def act(self, state, greedy=True):
        int_action = super().act(state, greedy)
        text_action = state['valid_actions'][int_action]
        return text_action
    
    def get_action_distribution(self, state):
        state_text = self.env.state_to_text(state)
        valid_actions_text = self.env.get_valid_actions_text(state)
        goal_text = self.env.goal_to_text()
        action_history = self.env.action_history
        print("llm_model:",self.llm_model)
        user_prompt = f'''
        **Task**: {goal_text}
        **Previous action**: {action_history}
        **State**: {state_text}
        **Valid Actions**: {valid_actions_text}
        '''
        
        # user_prompt = f'''
        # **Task**: {goal_text}
        # **Previous action**: {prev_action}
        # **State**: {state_text}
        # **Valid Actions**: {valid_actions_text}
        # #NOTES:
        # - your optimal action must not be the same as the previous action from **Previous action**
        # '''
        print("prompt:",user_prompt)
        messages, probs = self.query_llm(user_prompt)
        print("response:",messages)
        print("response_prob:",probs)
        print("--------------------- this is after response ----------------------")
        print("valid_actions_space:",len(valid_actions_text))
        dist = self._get_action_distribution(messages, probs, valid_actions_text)
        time.sleep(1)
        if self.debug:
            state_text = self.env.state_to_text(state)
            print(f"State: {state_text}")
            print(f"Action distribution: {dist}")
        print("alfworld_output_distrib:",dist)
        return dist

    # TODO: Abstract most of this away, only leave change at bottom
    def _get_action_distribution(self, messages, probs, valid_actions_text):
        print("entering_probs:",probs)

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
        print("similar_actions:",similar_actions)
        for i, action in enumerate(similar_actions):
            if action is not None:
                action_idx = self.env.action_txt_to_idx(action, valid_actions_text) # ONLY DIFFERENCE
                action_probs[action_idx] += probs[i]
        print("action_prob_b4_calculated:",action_probs)
        action_probs = action_probs ** (1/self.temp)
        action_probs /= np.sum(action_probs + 1e-10)
        print('action_alfworld_probs:',action_probs)
        
        return action_probs
    
    


