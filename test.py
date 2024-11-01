from pyRDDLGym.Elevator import Elevator
import numpy as np
from utils import state_to_text, action_txt_to_idx
from llm_policy_elevator import ElvatorLLMPolicyAgent
from tqdm import tqdm

env = Elevator(instance=5)
agent = ElvatorLLMPolicyAgent(device="cuda", debug=False)

state = env.reset()
state = env.disc2state(state)

num_episodes_to_run = 10
rewards = []

pbar = tqdm(range(num_episodes_to_run))

for i in pbar:
    num_steps = 0
    total_reward = 0
    state = env.reset()
    state = env.disc2state(state)
    
    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = env.disc2state(next_state)
        
        total_reward += reward
        
        if done:
            rewards.append(total_reward)
            break
        
        state = next_state
        num_steps += 1
        
        pbar.set_description(f"Episode {i+1}, current steps: {num_steps}, total reward: {total_reward}")
        
print("Average reward: ", np.mean(rewards))
