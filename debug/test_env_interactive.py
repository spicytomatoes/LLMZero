from pyRDDLGym.Elevator import Elevator
from utils import state_to_text, action_txt_to_idx, action_to_text
import numpy as np

env = Elevator(instance=5)
    
# interactive, wait for user input
state = env.reset()
state = env.disc2state(state)

num_steps = 0

print("Initial state:")
print(state)
print(state_to_text(state))

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = int(input("Enter action: "))
    print(f"Action: {action_to_text(action)}")
    next_state, reward, done, info = env.step(action)
    next_state = env.disc2state(next_state)
    print("Next state:")
    print(next_state)
    print(state_to_text(next_state))
    print(f"Reward: {reward}")
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    