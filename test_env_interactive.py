from environments.ElevatorEnvironment import ElevatorEnvironment 
import numpy as np

env = ElevatorEnvironment()
    
# interactive, wait for user input
state = env.reset()

num_steps = 0

print("Initial state:")
print(state)
print(env.state_to_text(state))

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = int(input("Enter action: "))
    print(f"Action: {env.action_to_text(action)}")
    next_state, reward, done, info = env.step(action)
    print("Next state:")
    print(next_state)
    print(env.state_to_text(next_state))
    print(f"Reward: {reward}")
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    