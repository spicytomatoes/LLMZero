from pyRDDLGym.Elevator import Elevator
import numpy as np
from utils import state_to_text, action_txt_to_idx, make_elevator_env
from agents.llm_policy_elevator import ElvatorLLMPolicyAgent
from agents.random_agent import RandomAgent
from agents.mcts import MCTSAgent
from agents.expert_policy import ExpertPolicyAgent
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run elevator environment")
    parser.add_argument("--agent", type=str, default="llm", help="Agent to run (llm, random, mcts, expert, mcts-expert, mcts-llm)")
    return parser.parse_args()

args = parse_args()

env = make_elevator_env(env_instance=5)
agent = None

if args.agent == "llm":
    agent = ElvatorLLMPolicyAgent(device="cuda", debug=True)
elif args.agent == "random":
    agent = RandomAgent()
elif args.agent == "mcts":
    agent = MCTSAgent(env, policy=None, debug=False)
elif args.agent == "expert":
    agent = ExpertPolicyAgent()
elif args.agent == "mcts-expert":
    mcts_args = {
            "num_simulations": 100,
            "c_puct": 100,    #should be proportional to the scale of the rewards
            "gamma": 0.99,
            "max_depth": 24,
        }
    agent = MCTSAgent(env, policy=ExpertPolicyAgent(), debug=False, args=mcts_args)
elif args.agent == "mcts-llm":
    llm_agent = ElvatorLLMPolicyAgent(device="cuda", debug=True, temp=10)
    mcts_args = {
            "num_simulations": 100,
            "c_puct": 500,    #should be proportional to the scale of the rewards
            "gamma": 0.95,
            "max_depth": 24,
        }
    agent = MCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)

else:
    raise ValueError("Invalid agent")

state = env.reset()

num_episodes_to_run = 1
rewards = []

pbar = tqdm(range(num_episodes_to_run))

for i in pbar:
    num_steps = 0
    total_reward = 0
    state = env.reset()
    
    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        if done:
            rewards.append(total_reward)
            break
        
        state = next_state
        num_steps += 1
        
        pbar.set_description(f"Episode {i+1}, current steps: {num_steps}, total reward: {total_reward}")
        
print("Average reward: ", np.mean(rewards))
