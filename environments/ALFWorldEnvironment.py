import gym
import alfworld.agents.environment as environment
import yaml
import re

class ALFWorldEnvironment(gym.Wrapper):
    '''
    wrapper for ALFWorld environment
    '''
    current_goal = None

    def __init__(self):
        config = self._load_config()
        env_type = config['env']['type']

        env = getattr(environment, env_type)(config, train_eval='train')
        env = env.init_env(batch_size=1) # batch_size = how many environments to run in parallel
        super().__init__(env)

    def reset(self):
        match = False
        TOTAL_NUM_RETRIES = 10
        num_retries = 0
        while not match and num_retries < TOTAL_NUM_RETRIES:
            try:
                state, infos = self.env.reset() # Can also accept task_file if needed
                state = self._map_state(state, infos)

                find_goal_regex = r"Your task is to:\s+(.*)"
                match = re.search(find_goal_regex, state['text_state'])

                if match:
                    self.current_goal = match.group(1)
                else:
                    raise Exception("Could not find valid task with goal.")
            except:
                if num_retries >= TOTAL_NUM_RETRIES-1:
                    raise
                num_retries += 1

        return state, infos

    def checkpoint(self):
        '''
        return a checkpoint of the current environment state
        '''
        raise NotImplementedError()

    def restore_checkpoint(self, checkpoint):
        '''
        restore the environment to a previous state
        '''
        raise NotImplementedError()

    def step(self, action):
        next_state, reward, done, infos = self.env.step([action])
        next_state = self._map_state(next_state, infos)
        return next_state, reward[0], done[0], None, infos
    
    def get_valid_actions(self, state):
        return state['valid_actions']
    
    def get_valid_actions_text(self, state):
        return state['valid_actions']
    
    def state_to_text(self, state):
        return state['text_state']
    
    def action_to_text(self, action):
        return action # Action already in text
    
    def action_txt_to_idx(self, action_txt, valid_actions_txt):
        return valid_actions_txt.index(action_txt)
    
    def format_llm_prompt(self, prompt: str, state):
        state_text = self.state_to_text(state)
        valid_actions_text = ', '.join(self.get_valid_actions_text(state))
        prompt = prompt.replace('[STATE]', state_text).replace('[ACTIONS]', valid_actions_text).replace('[GOAL]', self.current_goal)
        return prompt
    
    def _load_config(self):
        CONFIG_FILE_PATH = 'configs/alfworld_env.yaml'
        with open(CONFIG_FILE_PATH) as reader:
            config = yaml.safe_load(reader)
        return config

    def _map_state(self, state, infos):
        return {
            'text_state': state[0],
            'valid_actions': infos['admissible_commands'][0],
        }
    
