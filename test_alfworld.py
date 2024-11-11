from agents.alfworld_llm_policy import ALFWorldLLMPolicyAgent
from environments.ALFWorldEnvironment import ALFWorldEnvironment
import numpy as np
from agents.random_agent import RandomAgent
from agents.mcts import MCTSAgent
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
        llm_agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
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

def run_single_trial(env, agent, log_file):
    num_steps = 0
    total_reward = 0
    max_num_steps = 40

    state, _ = env.reset()
    log(log_file, state['text_state'])
    while True:
        action = agent.act(state)
        print(action)
        next_state, reward, done, _, info = env.step(action)

        num_steps += 1
        total_reward += reward
        print(f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")
        
        if done:
            break
        if num_steps >= max_num_steps:
            print('Unable to finish task')
            break
        
        state = next_state
    
    return done, num_steps, total_reward
    


def main():
    if os.getenv("ALFWORLD_DATA") is None:
        raise Exception("Must set path in ALFWORLD_DATA environment variable to use ALFWorld Environment")

    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    AGENTS_TO_TEST = ['mcts-llm']

    FILES_TO_TEST = [
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-AlarmClock-None-Desk-307/trial_T20190907_072303_146844',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Book-None-Bed-312/trial_T20190908_103648_829231',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Bowl-None-Fridge-6/trial_T20190906_230933_751794',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Candle-None-Toilet-413/trial_T20190909_104025_525772',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-CellPhone-None-Bed-322/trial_T20190907_163932_211569',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-CreditCard-None-Shelf-316/trial_T20190909_092853_746076',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Laptop-None-Bed-302/trial_T20190908_112426_055221',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Pen-None-SideTable-309/trial_T20190907_051843_166835',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Pencil-None-SideTable-322/trial_T20190908_112624_358795',
        '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Pillow-None-ArmChair-206/trial_T20190909_011649_821132'
    ]

    for agent_name in AGENTS_TO_TEST:
        log_file = open(f'{agent_name}_trial_logs.log', 'a')

        rewards = []
        num_failure = 0
        num_success = 0

        for file in FILES_TO_TEST:
            log(log_file, '--------------------------------------------------')

            env = ALFWorldEnvironment(overwrite_env=file)
            agent = get_agent(agent_name, env, args)
            success, num_steps, total_reward = run_single_trial(env, agent, log_file)

            if success:
                num_success += 1
                log(log_file, f'SUCCESS, Total steps: {num_steps}, total reward: {total_reward}')
            else:
                num_failure += 1
                log(log_file, f'FAIL, Total steps: {num_steps}, total reward: {total_reward}')
        
            rewards.append(total_reward)

        log(log_file, f"Average reward: {np.mean(rewards)}")
        log(log_file, f'Num SUCCESS: {num_success}')
        log(log_file, f'Num FAIL: {num_failure}')

        log_file.close()

if __name__ == '__main__':
    main()
