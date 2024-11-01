import numpy as np
from utils import state_to_text, action_to_text, make_elevator_env
from agents.llm_policy_elevator import ElvatorLLMPolicyAgent
from agents.mcts import MCTSAgent
import copy

env = make_elevator_env(env_instance=5)
#agent = ElvatorLLMPolicyAgent(device="cuda", debug=True)
agent = MCTSAgent(env, use_llm=False, debug=True)

state = env.reset()

num_steps = 0

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = agent.act(state)
    
    state, reward, done, _ = env.step(action)
    
    print(f"action: {action_to_text(action)}")
    print(f"state:\n{state_to_text(state)}")
    print("reward: ", reward)
    
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    
    wait = input("Press Enter to continue...")