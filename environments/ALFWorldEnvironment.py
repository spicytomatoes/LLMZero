import gym
import alfworld.agents.environment as environment
import yaml

class ALFWorldEnvironment(gym.Wrapper):
    '''
    wrapper for ALFWorld environment
    '''
    def __init__(self):
        config = self._load_config()
        env_type = config['env']['type']

        env = getattr(environment, env_type)(config, train_eval='train')
        env = env.init_env(batch_size=1) # batch_size = how many environments to run in parallel
        super().__init__(env)

    def reset(self):
        state, infos = self.env.reset() # Can also accept task_file if needed
        state = self._map_state(state, infos)
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
    
    def action_txt_to_idx(self, action_txt, valid_actions):
        return valid_actions.index(action_txt)
    
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
    
