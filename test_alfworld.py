from agents.alfworld_llm_policy import ALFWorldLLMPolicyAgent
from environments.ALFWorldEnvironment import ALFWorldEnvironment
import numpy as np
from agents.llm_policy import LLMPolicyAgent
from agents.random_agent import RandomAgent
from agents.mcts import MCTSAgent
from agents.nn_agent import NNAgent
from tqdm import tqdm
import torch
import argparse
import yaml
import os

from dotenv import load_dotenv
load_dotenv()
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ALFworld environment")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()

def get_agent(agent_name, env, args):
    cfg = yaml.safe_load(open("configs/alfworld.yaml"))
    agent = None
    if agent_name == "llm":
        agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_policy"])
    elif agent_name == "random":
        agent = RandomAgent(env, seed = args.seed)
    elif agent_name == "mcts":
        agent = MCTSAgent(env, policy=None, debug=True)
    elif agent_name == "mcts-llm":
        llm_agent = LLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
        mcts_args = cfg["llm_mcts"]["mcts_args"]
        agent = MCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)
    else:
        raise ValueError("Invalid agent")

    return agent

def log(log_file, text):
    log_file.write('\n')
    log_file.write(text)
    log_file.flush()
    print(text)

def run_trial(env, agent, log_file):
    rewards = []
    NUM_ENVIRONMENTS = 35 # Number of ALFWorld evaluation environments
    num_failure = 0
    num_success = 0
    for i in range(NUM_ENVIRONMENTS):
        log(log_file, '--------------------------------------------------')
        num_steps = 0
        total_reward = 0
        max_num_steps = 40

        state, _ = env.reset()
        log(log_file, state['text_state'])
        while True:
            action = agent.act(state)
            print(action)
            next_state, reward, done, _, info = env.step(action)
            
            total_reward += reward
            print(f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")
            
            if done:
                num_success += 1
                log(log_file, f'SUCCESS, Total steps: {num_steps}, total reward: {total_reward}')
                break

            if num_steps >= max_num_steps:
                num_failure += 1
                print('Unable to finish task')
                log(log_file, f'FAIL, Total steps: {num_steps}, total reward: {total_reward}')
                break
            
            state = next_state
            num_steps += 1
        
        rewards.append(total_reward)

    log(log_file, f"Average reward: {np.mean(rewards)}")
    log(log_file, f'Num SUCCESS: {num_success}')
    log(log_file, f'Num FAIL: {num_failure}')


def main():
    if os.getenv("ALFWORLD_DATA") is None:
        raise Exception("Must set path in ALFWORLD_DATA environment variable to use ALFWorld Environment")

    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    AGENTS_TO_TEST = ['llm', 'mcts-llm']
    for agent_name in AGENTS_TO_TEST:
        log_file = open(f'{agent_name}_trial_logs.log', 'w')

        env = ALFWorldEnvironment()
        agent = get_agent(agent_name, env, args)
        run_trial(env, agent, log_file)

        log_file.close()

if __name__ == '__main__':
    main()
