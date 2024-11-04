from pyRDDLGym.Elevator import Elevator
import numpy as np
from agents.llm_policy import LLMPolicyAgent
from agents.random_agent import RandomAgent
from agents.mcts import MCTSAgent
from agents.elvator_expert import ElevatorExpertPolicyAgent
from environments.elevator.ElevatorEnvironment import ElevatorEnvironment
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run elevator environment")
    parser.add_argument("--agent", type=str, default="llm", help="Agent to run (llm, random, mcts, expert, mcts-expert, mcts-llm)")
    return parser.parse_args()

args = parse_args()

env = ElevatorEnvironment()
agent = None

if args.agent == "llm":
    agent = LLMPolicyAgent(env, device="cuda", debug=False)
elif args.agent == "random":
    agent = RandomAgent()
elif args.agent == "mcts":
    agent = MCTSAgent(env, policy=None, debug=False)
elif args.agent == "expert":
    agent = ElevatorExpertPolicyAgent()
elif args.agent == "mcts-expert":
    mcts_args = {
            "num_simulations": 100,
            "c_puct": 100,    #should be proportional to the scale of the rewards
            "gamma": 0.99,
            "max_depth": 24,
        }
    agent = MCTSAgent(env, policy=ElevatorExpertPolicyAgent(), debug=False, args=mcts_args)
elif args.agent == "mcts-llm":
    llm_agent = LLMPolicyAgent(env, device="cuda", debug=False, temp=10)
    mcts_args = {
            "num_simulations": 100,
            "c_puct": 500,    #should be proportional to the scale of the rewards
            "gamma": 0.95,
            "max_depth": 30,
        }
    agent = MCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)

else:
    raise ValueError("Invalid agent")

state = env.reset()

num_episodes_to_run = 1
rewards = []

# pbar = tqdm(range(num_episodes_to_run))

for i in range(num_episodes_to_run):
    num_steps = 0
    total_reward = 0
    state = env.reset()
    
    while True:
        action = agent.act(state)
        print(action)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        if done:
            rewards.append(total_reward)
            break
        
        state = next_state
        num_steps += 1
        
        print(f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")
        # pbar.set_description(f"Episode {i+1}, current steps: {num_steps}, total reward: {total_reward}")
        
print("Average reward: ", np.mean(rewards))
