import numpy as np
import yaml
import inquirer
import gym
import random
import os
from enum import Enum
import torch
from dotenv import load_dotenv, set_key, get_key

def create_env_file_if_needed():
    env_file = ".env"

    keys = [
        "OPENAI_API_KEY",
        "CUSTOM_BASE_URL",
        "CUSTOM_API_KEY",
        "CUSTOM_MODEL_ID",
        "USE_OPENAI_CUSTOM",
    ]
    
    print(f"Checking if {env_file} exists...")

    if not os.path.exists(env_file):
        print(f"{env_file} not found. Creating it...")
        open(env_file, 'w').close()

    load_dotenv(env_file)

    for key in keys:
        if get_key(dotenv_path=env_file, key_to_get=key) is None:
            print(f"{key} not found in {env_file}. Adding it...")
            set_key(env_file, key, "")
            print(f"{key} is added to {env_file}.")
        else:
            print(f"{key} already exists in {env_file}. No changes needed.")


create_env_file_if_needed()

from environments.ALFWorldEnvironment import ALFWorldEnvironment
from environments.ElevatorEnvironment import ElevatorEnvironment
from agents.random_agent import RandomAgent
from agents.nn_agent import NNAgent
from agents.mcts import MCTSAgent
from agents.llmzero import LLMZeroAgent
from agents.llm_policy import LLMPolicyAgent
from agents.elevator_expert import ElevatorExpertPolicyAgent

from agents.alfworld_mcts import ALFworldMCTSAgent
from agents.alfworld_llmzero import ALFWorldLLMZeroAgent
from agents.alfworld_llm_policy import ALFWorldLLMPolicyAgent

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class ENVIRONMENT(str, Enum):
    ELEVATOR = "Elevator"
    ALF_WORLD = "ALFWorld"

    @classmethod
    def list(self):
        return list(map(lambda c: c.value, self))


class AGENT(str, Enum):
    LLM_POLICY = "LLM Policy"
    RANDOM = "Random"
    MCTS = "MCTS"
    EXPERT = "Expert"
    MCTS_EXPERT = "MCTS Expert"
    LLM_MCTS = "LLM MCTS"
    NEURAL_NETWORK = "Neural Network"
    LLM_ZERO = "LLM Zero"

    @classmethod
    def list(self, env: ENVIRONMENT):
        if env == ENVIRONMENT.ELEVATOR:
            return list(map(lambda c: c.value, self))
        elif env == ENVIRONMENT.ALF_WORLD:
            return [AGENT.LLM_POLICY.value, AGENT.LLM_MCTS.value, AGENT.LLM_ZERO.value, AGENT.MCTS.value]


def load_env(env_str: ENVIRONMENT):
    print("env_str", env_str)
    if env_str == ENVIRONMENT.ELEVATOR.name:
        env = ElevatorEnvironment()
        cfg = yaml.safe_load(open("configs/elevator.yaml"))
    elif env_str == ENVIRONMENT.ALF_WORLD.name:
        env = ALFWorldEnvironment()
        cfg = yaml.safe_load(open("configs/alfworld.yaml"))
    else:
        raise Exception("Invalid environment selected.")
    return env, cfg


def load_agent(agent_str, env_str,  env: gym.Wrapper, cfg):
    if agent_str == AGENT.LLM_POLICY.name:
        if env_str == ENVIRONMENT.ELEVATOR.name:
            agent = LLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_policy"])
        elif env_str == ENVIRONMENT.ALF_WORLD.name:
            agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_policy"])
        else:
            raise ValueError("Invalid Environment")
    elif agent_str == AGENT.RANDOM.name:
        agent = RandomAgent(env, seed=SEED)
    elif agent_str == AGENT.MCTS.name:
        agent = MCTSAgent(env, policy=None, debug=True)
    elif agent_str == AGENT.EXPERT.name:
        agent = ElevatorExpertPolicyAgent()  # TO DO: abstract out the expert policy
    elif agent_str == AGENT.MCTS_EXPERT.name:
        mcts_args = cfg["mcts_expert"]["mcts_args"]
        agent = MCTSAgent(env, policy=ElevatorExpertPolicyAgent(),
                          debug=False, args=mcts_args)
    elif agent_str == AGENT.LLM_MCTS.name:
        mcts_args = cfg["llm_mcts"]["mcts_args"]
        if env_str == ENVIRONMENT.ELEVATOR.name:
            llm_agent = LLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
            agent = MCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)
        elif env_str == ENVIRONMENT.ALF_WORLD.name:
            llm_agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
            agent = ALFworldMCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)
        else:
            raise ValueError("Invalid Environment")
    elif agent_str == AGENT.NEURAL_NETWORK.name:
        agent = NNAgent(env, cfg=cfg["nn_agent"])
    elif agent_str == AGENT.LLM_ZERO.name:
        if env_str == ENVIRONMENT.ELEVATOR.name:
            agent = LLMZeroAgent(env)
        elif env_str == ENVIRONMENT.ALF_WORLD.name:
            agent = ALFWorldLLMZeroAgent(env)
        else:
            raise ValueError("Invalid Environment")
    else:
        raise ValueError("Invalid agent")

    return agent


def user_selection():
    env_question = [
        inquirer.List("env", "Select an environment", ENVIRONMENT.list()),
    ]
    env_answer = inquirer.prompt(env_question)
    
    agent_question = [
        inquirer.List("agent", "Select an agent", AGENT.list(env=ENVIRONMENT(env_answer["env"]))),
    ]
    agent_answer = inquirer.prompt(agent_question)

    selected_env = ENVIRONMENT(env_answer["env"]).name
    selected_agent = AGENT(agent_answer["agent"]).name
    
    # print('ENVIRONMENT(env_answer["env"]).name', ENVIRONMENT(env_answer["env"]).name)
    # print('AGENT(agent_answer["agent"]).name', AGENT(agent_answer["agent"]).name)

    return selected_env, selected_agent


def run():
    selected_env, selected_agent = user_selection()

    env, cfg = load_env(selected_env)
    agent = load_agent(selected_agent, selected_env, env, cfg)

    rewards = []

    for i in range(1):
        num_steps = 0
        total_reward = 0
        if isinstance(agent, RandomAgent) or isinstance(agent, NNAgent):
            state, _ = env.reset(seed=SEED)
        else:
            state, _ = env.reset()

        while True:
            action = agent.act(state)
            print(env.action_to_text(action))
            next_state, reward, done, _, info = env.step(action)

            total_reward += reward

            if done:
                rewards.append(total_reward)
                break

            state = next_state
            num_steps += 1

            print(
                f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")

    print("Average reward: ", np.mean(rewards))


if __name__ == "__main__":
    # check if .env file exist, if not create one and append something to it
    run()
