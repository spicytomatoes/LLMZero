def state_to_text(state_dict):
    state_text = ""
    
    num_person_waiting = [None for _ in range(5)]
    num_in_elavator = None
    door_state = None
    direction = None
    current_floor = None
    
    for feature, value in state_dict.items():
        if "num-person-waiting" in feature:
            num_person_waiting[int(feature[-1])] = value
        if "elevator-at-floor" in feature and value == True:
            current_floor = int(feature[-1]) + 1
        if feature == "elevator-dir-up___e0":
            direction = "up" if value == True else "down"
        if feature == "elevator-closed___e0":
            door_state = "closed" if value == True else "open"
        if feature == "num-person-in-elevator___e0":
            num_in_elavator = value
            
    state_text += f"There are "
    flag = False
    for i in range(5):
        if num_person_waiting[i] > 0:
            state_text += f"{num_person_waiting[i]} people waiting at floor {i+1}. "
            flag = True
            
    if not flag:
        state_text += "no one waiting at any floor."
    state_text += "\n"
        
    state_text += f"Elevator at floor {current_floor}.\n"
    state_text += f"There are {num_in_elavator} people in the elevator.\n"
    state_text += f"Elevator is moving {direction}.\n"
    state_text += f"Elevator door is {door_state}.\n"
    
    return state_text

def action_txt_to_idx(action_txt):
        if action_txt == "move":
            return 1
        elif action_txt == "open":
            return 5
        elif action_txt == "close":
            return 3
        elif action_txt == "nothing":
            return 0
        else:
            raise ValueError(f"Invalid action text {action_txt}")
        
def action_to_text(action):
    if action == 0 or action == 2 or action == 4:
        return "nothing"
    elif action == 1:
        return "move"
    elif action == 3:
        return "close door"
    elif action == 5:
        return "open door"
    else:
        raise ValueError(f"Invalid action {action}")