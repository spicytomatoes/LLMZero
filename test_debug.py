import numpy as np
from utils import state_to_text, action_to_text, make_elevator_env
from agents.llm_policy_elevator import ElvatorLLMPolicyAgent
from agents.mcts import MCTSAgent
from agents.expert_policy import ExpertPolicyAgent
import copy

env = make_elevator_env(env_instance=5)
llm_agent = ElvatorLLMPolicyAgent(device="cuda", debug=True, temp=10)
# agent = ExpertPolicyAgent()
mcts_args = {
            "num_simulations": 20,
            "c_puct": 1000,    #should be proportional to the scale of the rewards 
            "gamma": 0.95,
            "max_depth": 30,
            "num_rollouts": 1,
            "backprop_T": 10,
        }
agent = MCTSAgent(env, policy=llm_agent, debug=True, args=mcts_args)
# agent = MCTSAgent(env, policy=ExpertPolicyAgent(), debug=True, args=mcts_args)

state = env.reset()

num_steps = 0

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = agent.act(state)
    
    state, reward, done, _ = env.step(action)
    
    print(f"action: {action_to_text(env.map_action(action))}")
    print(f"state:\n{state_to_text(state)}")
    print("reward: ", reward)
    
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    
    wait = input("Press Enter to continue...")