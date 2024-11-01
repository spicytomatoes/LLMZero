from pyRDDLGym.Elevator import Elevator
import numpy as np
from utils import state_to_text, action_txt_to_idx
from llm_policy_elevator import ElvatorLLMPolicyAgent

env = Elevator(instance=5)
agent = ElvatorLLMPolicyAgent(device="cuda", debug=True)

state = env.reset()
state = env.disc2state(state)

num_steps = 0

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = agent.act(state)
    # print(f"action_idx: {action}")
    state, reward, done, info = env.step(action)
    state = env.disc2state(state)
    print("reward: ", reward)
    
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    
    wait = input("Press Enter to continue...")