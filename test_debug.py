import numpy as np
from agents.llm_policy import LLMPolicyAgent
from agents.mcts import MCTSAgent
from agents.elvator_expert import ElevatorExpertPolicyAgent
from environments.elevator.ElevatorEnvironment import ElevatorEnvironment

env = ElevatorEnvironment()
agent = LLMPolicyAgent(env, device="cuda", debug=True, temp=1)
# agent = ElevatorExpertPolicyAgent()
# mcts_args = {
#             "num_simulations": 20,
#             "c_puct": 1000,    #should be proportional to the scale of the rewards 
#             "gamma": 0.95,
#             "max_depth": 30,
#             "num_rollouts": 1,
#             "backprop_T": 10,
#         }
# agent = MCTSAgent(env, policy=llm_agent, debug=True, args=mcts_args)
# agent = MCTSAgent(env, policy=ElevatorExpertPolicyAgent(), debug=True, args=mcts_args)

state = env.reset()

num_steps = 0

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = agent.act(state)
    
    state, reward, done, _ = env.step(action)
    
    print(f"action: {env.action_to_text(action)}")
    print(f"state:\n{env.state_to_text(state)}")
    print("reward: ", reward)
    
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    
    wait = input("Press Enter to continue...")