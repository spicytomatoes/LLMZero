import gym
import alfworld.agents.environment as environment
import yaml
import re

from alfworld.agents.modules.generic import EpisodicCountingMemory, ObjCentricEpisodicMemory

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

        self.episodic_counting_memory = EpisodicCountingMemory() # Tracks newly explored states
        self.obj_centric_episodic_counting_memory = ObjCentricEpisodicMemory() # Tracks newly explored objects

        super().__init__(env)

    def reset(self):
        self.episodic_counting_memory.reset()
        self.obj_centric_episodic_counting_memory.reset()

        match = False
        TOTAL_NUM_RETRIES = 10
        num_retries = 0
        while not match and num_retries < TOTAL_NUM_RETRIES:
            try:
                state, infos = self.env.reset() # Can also accept task_file if needed

                find_goal_regex = r"Your task is to:\s+(.*)"
                match = re.search(find_goal_regex, state[0])

                if match:
                    self.current_goal = match.group(1)
                else:
                    raise Exception("Could not find valid task with goal.")
            except:
                if num_retries >= TOTAL_NUM_RETRIES-1:
                    raise
                num_retries += 1

        # Add initial state to memory
        self.episodic_counting_memory.push(state)
        self.obj_centric_episodic_counting_memory.push(state)

        state = self._map_state(state, infos)

        return state, infos

    def checkpoint(self):
        '''
        return a checkpoint of the current environment state
        '''
        return self.env.batch_env.envs[0].unwrapped.state.copy()

    def restore_checkpoint(self, checkpoint):
        '''
        restore the environment to a previous state
        '''
        self.env.batch_env.envs[0].unwrapped.state = checkpoint

    def step(self, action):
        next_state, reward, done, infos = self.env.step([action])
        curr_reward = self._get_current_reward(reward[0], next_state)
        next_state = self._map_state(next_state, infos)

        return next_state, curr_reward, done[0], None, infos

    def _get_current_reward(self, step_reward, next_state):
        MAX_NORM_VALUE = 0.2
        new_state_reward = self.episodic_counting_memory.is_a_new_state(next_state)[0] # Between 0 and 1
        new_state_reward = new_state_reward / MAX_NORM_VALUE # Normalize value to be between 0 and 0.2
        new_object_reward = self.obj_centric_episodic_counting_memory.get_object_novelty_reward(next_state)[0] # Between 0 and 1
        new_object_reward = new_object_reward / MAX_NORM_VALUE # Normalize value to be between 0 and 0.2
        step_reward = -0.03 if step_reward == 0 else step_reward # Haven't reached goal: 0 -> -3, Reaching goal: 1

        current_reward = step_reward + new_state_reward + new_object_reward

        self.episodic_counting_memory.push(next_state)
        self.obj_centric_episodic_counting_memory.push(next_state)

        return current_reward
    
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
    
    def goal_to_text(self):
        return self.current_goal

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
    
