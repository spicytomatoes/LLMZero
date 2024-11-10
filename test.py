from agents.alfworld_llm_policy import ALFWorldLLMPolicyAgent
from environments.ALFWorldEnvironment import ALFWorldEnvironment
from pyRDDLGym.Elevator import Elevator
import numpy as np
from agents.llm_policy import LLMPolicyAgent
from agents.random_agent import RandomAgent
from agents.mcts import MCTSAgent
from agents.elevator_expert import ElevatorExpertPolicyAgent
from agents.nn_agent import NNAgent
from environments.ElevatorEnvironment import ElevatorEnvironment
from environments.BlackjackEnvironment import BlackjackEnvironment
from tqdm import tqdm
import torch
import argparse
import yaml
import os


from dotenv import load_dotenv
load_dotenv()
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Run elevator environment")
    parser.add_argument("--agent", type=str, default="llm", help="Agent to run (llm, random, mcts, expert, mcts-expert, mcts-llm, nn)")
    parser.add_argument("--env", type=str, default="elevator", help="Environment to run (elevator, blackjack, alfworld)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
    return parser.parse_args()

args = parse_args()


if args.env == "elevator":
    env = ElevatorEnvironment()
    cfg = yaml.safe_load(open("configs/elevator.yaml"))
    
elif args.env == "blackjack":
    env = BlackjackEnvironment()
    cfg = yaml.safe_load(open("configs/blackjack.yaml"))

elif args.env == "alfworld":
    if os.getenv("ALFWORLD_DATA") is None:
        raise Exception("Must set path in ALFWORLD_DATA environment variable to use ALFWorld Environment")

    env = ALFWorldEnvironment()
    cfg = yaml.safe_load(open("configs/alfworld.yaml"))
else:
    raise Exception("Invalid environment selected.")


agent = None

if args.agent == "llm":
    llm_agents = {
        "elevator": LLMPolicyAgent,
        "blackjack": LLMPolicyAgent,
        "alfworld": ALFWorldLLMPolicyAgent
    }
    agent = llm_agents[args.env](env, device="cuda", debug=False, **cfg["llm_policy"])
elif args.agent == "random":
    agent = RandomAgent(env, seed = args.seed)
elif args.agent == "mcts":
    agent = MCTSAgent(env, policy=None, debug=True)
elif args.agent == "expert":
    agent = ElevatorExpertPolicyAgent() # TO DO: abstract out the expert policy
elif args.agent == "mcts-expert":
    mcts_args = cfg["mcts_expert"]["mcts_args"]
    agent = MCTSAgent(env, policy=ElevatorExpertPolicyAgent(), debug=False, args=mcts_args)
elif args.agent == "mcts-llm":
    llm_agents = {
        "elevator": LLMPolicyAgent,
        "blackjack": LLMPolicyAgent,
        "alfworld": ALFWorldLLMPolicyAgent
    }
    llm_agent = llm_agents[args.env](env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
    mcts_args = cfg["llm_mcts"]["mcts_args"]
    agent = MCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)
elif args.agent == "nn":
    agent = NNAgent(env, cfg=cfg["nn_agent"])
else:
    raise ValueError("Invalid agent")


if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

num_episodes_to_run = args.num_episodes
rewards = []

# pbar = tqdm(range(num_episodes_to_run))

for i in range(num_episodes_to_run):
    num_steps = 0
    total_reward = 0

    if args.seed is not None:
        seed = args.seed + i
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    
    while True:
        action = agent.act(state)
        print(action)
        next_state, reward, done, _, info = env.step(action)
        
        total_reward += reward
        
        if done:
            print("last state:",next_state)
            rewards.append(total_reward)
            break
        
        state = next_state
        num_steps += 1
        
        print(f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")
        # pbar.set_description(f"Episode {i+1}, current steps: {num_steps}, total reward: {total_reward}")
        
print("Average reward: ", np.mean(rewards))
