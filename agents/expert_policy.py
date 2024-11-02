import numpy as np

class CustomElevatorState:
    def __init__(self, num_person_waiting, num_in_elevator, door_state, direction, current_floor):
        self.num_person_waiting = num_person_waiting
        self.num_in_elevator = num_in_elevator
        self.door_state = door_state
        self.direction = direction
        self.current_floor = current_floor
        
        self.valid_actions = [0, 1, 2, 3]   # 0: nothing, 1: move, 2: close door, 3: open door

class ExpertPolicyAgent:
    def __init__(self, strategy='greedy', num_floors=5, elevator_capacity=10):
        self.strategy = strategy
        self.num_floors = num_floors
        self.elevator_capacity = elevator_capacity
        
        self.state = None        
        
    def act(self, state):
        '''
        Expert policy for the elevator environment, rule based
        '''
        
        self.state = self.parse_state(state)
        
        if self.strategy == 'greedy':
            return self.greedy_action()
        
        else:
            raise ValueError(f"Invalid strategy {self.strategy}")
        
    def parse_state(self, state):
        
        num_person_waiting = [None for _ in range(5)]
        num_in_elavator = None
        door_state = None
        direction = None
        current_floor = None
        
        for feature, value in state.items():
            if "num-person-waiting" in feature:
                num_person_waiting[int(feature[-1])] = int(value)
            if "elevator-at-floor" in feature and value == True:
                current_floor = int(feature[-1]) 
            if feature == "elevator-dir-up___e0":
                direction = "up" if value == True else "down"
            if feature == "elevator-closed___e0":
                door_state = "closed" if value == True else "open"
            if feature == "num-person-in-elevator___e0":
                num_in_elavator = value
                
        return CustomElevatorState(num_person_waiting, num_in_elavator, door_state, direction, current_floor)
    
    def greedy_action(self):
        '''
        Greedy policy
        '''
        
        # Case 0: If the door is open, close it
        if self.state.door_state == "open":
            return 2
        
        # Case 1: Elevator is empty, no one waiting, move to the ideal floor
        if self.state.num_in_elevator == 0 and sum(self.state.num_person_waiting) == 0:
            if self.state.direction == "up" and self.state.current_floor < self.num_floors - 1:
                return 1
            elif self.state.direction == "down" and self.state.current_floor < 2:
                # Elevator is near the bottom floor, go down to move up
                return 1
            else:
                return 0
            
        # Case 2: There are people waiting and elevator is moving up, pick up from the top floor if enought capacity
        elif self.state.direction == "up":
            assert self.state.num_in_elevator == 0, "Elevator should be empty if it is moving up"
            
            #calculate the total number of people waiting at the current floor and below
            ppl_below_current = sum(self.state.num_person_waiting[:self.state.current_floor+1])
    
            remaining_capacity = self.elevator_capacity - ppl_below_current
            
            #calculate the total number of people waiting above the current floor
            ppl_above = sum(self.state.num_person_waiting[self.state.current_floor+1:])
            
            if ppl_above > 0:
                #get the number of people waiting on the next floor
                next_floor = self.state.current_floor + 1
                if self.state.num_person_waiting[next_floor] <= remaining_capacity:
                    return 1
            else:
                #either no one is waiting or not enough capacity to pick up
                if self.state.num_person_waiting[self.state.current_floor] == 0:
                    # must be below the current floor, keep moving
                    return 1
                else:
                    # pick up the people waiting at the current floor
                    return 3
            
        # Case 3: Elevator going down, pick up people from the bottom floor if enough capacity
        elif self.state.direction == "down":
            remaining_capacity = self.elevator_capacity - self.state.num_in_elevator
            num_waiting = self.state.num_person_waiting[self.state.current_floor]
            
            # assuming some people can go in even if total waiting > remaining capacity
            if remaining_capacity > 0 and num_waiting > 0:
                # open the door
                return 3
            else:
                # move to the next floor
                return 1
                    
        else:
            raise ValueError(f"Invalid direction {self.state.direction}")
            
            
            
                
        