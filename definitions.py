import random
import logging
from collections import defaultdict



#there are 16*16*16*3 possible states, we will create a dictionary to create a mapping(for every policy)
# Initialize logging



logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

class State:
    def __init__(self,player_1,player_2,opponent,possesion,game_over=False):  
        if not (isinstance(player_1, int) and 1 <= player_1 <= 16):
            raise ValueError("player_1 postion must be an integer between 1 and 16")
        if not (isinstance(player_2, int) and 1 <= player_2 <= 16):
            raise ValueError("player_2 position must be an integer between 1 and 16")
        if not (isinstance(opponent, int) and 1 <= opponent <= 16):
            raise ValueError("opponent position must be an integer between 1 and 16")
        if possesion not in (0, 1, 2):
            raise ValueError("possesion must be 0, 1, or 2") # 0 meaning game is lost
        self.player_1 = player_1
        self.player_2 = player_2
        self.opponent = opponent
        self.possesion = possesion
        self.game_over = game_over
    #Board is a 4x4 matrix where each cell is a number 1 to 16 with  i(x coordinate) = number of cell % 4 -1 and j(y coordinate) = number of cell // 4 -1
    #out of bounds check is not applied hwere, it is applied in the take_action function
    @property
    def player_1_x(self):
        return int((self.player_1 - 1) % 4)

    @player_1_x.setter
    def player_1_x(self, value):
        self.player_1 = value + 1 + 4 * self.player_1_y

    @property
    def player_1_y(self):
        return (self.player_1 - 1) // 4

    @player_1_y.setter
    def player_1_y(self, value):
        self.player_1 = self.player_1_x + 1 + 4 * value

    @property
    def player_2_x(self):
        return int((self.player_2 - 1) % 4)

    @player_2_x.setter
    def player_2_x(self, value):
        self.player_2 = value + 1 + 4 * self.player_2_y

    @property
    def player_2_y(self):
        return (self.player_2 - 1) // 4

    @player_2_y.setter
    def player_2_y(self, value):
        self.player_2 = self.player_2_x + 1 + 4 * value

    @property
    def opponent_x(self):
        return (self.opponent - 1) % 4

    @opponent_x.setter
    def opponent_x(self, value):
        self.opponent = value + 1 + 4 * self.opponent_y

    @property
    def opponent_y(self):
        return (self.opponent - 1) // 4

    @opponent_y.setter
    def opponent_y(self, value):
        self.opponent = self.opponent_x + 1 + 4 * value
        
    def __str__(self):
        return f"Player 1: {self.player_1}, Player 2: {self.player_2}, Opponent: {self.opponent}, Possesion: {self.possesion}"


 
def take_action(state:State, action, opponent_action,p=0.1, q=0.6):
    # returns the next state s_t+1 and also the reward r_t+1
    if action not in range(0, 10):
        raise ValueError("Action must be an integer between 1 and 10")
    
    # Check if the action is valid
    if state.game_over :
        raise ValueError("Cannot take action when the game is over")
    
    ##---------- Move the opponent according to policy which is unkown to agent-----------------
    # Opponent action: 1=Left, 2=Right, 3=Up, 4=Down
    prev_opponent_pos = state.opponent
    op_dx, op_dy = 0, 0
    if opponent_action == 1:   # Left
        op_dx = -1
    elif opponent_action == 2: # Right
        op_dx = 1
    elif opponent_action == 3: # Up
        op_dy = -1
    elif opponent_action == 4: # Down
        op_dy = 1

    new_op_x = state.opponent_x + op_dx
    new_op_y = state.opponent_y + op_dy

    # Check bounds for opponent
    if 0 <= new_op_x < 4 and 0 <= new_op_y < 4:
        state.opponent = new_op_x + 1 + 4 * new_op_y
    else:
        raise ValueError("Opponent moved out of bounds")
    # prev_opponent_pos holds the previous position number of the opponent
    
    ##---------- PASSING-----------------

    #This case wasnt in the documentation , i have added it myself.
    # Determine which player is moving and if they have possession
    # If opponent moves onto the same square as the player who has the ball,
    # and we have moved the player who doesn't have the ball, then possession is lost
    if ((state.possesion == 1 and state.opponent == state.player_1 and action in [4, 5, 6, 7]) or
        (state.possesion == 2 and state.opponent == state.player_2 and action in [0, 1, 2, 3])):
        return State(state.player_1, state.player_2, state.opponent, 0,True), -1

    if action in [0, 1, 2, 3]:
        # Move player 1
        dx, dy = 0, 0
        if action == 0:   # Left
            dx = -1
        elif action == 1: # Right
            dx = 1
        elif action == 2: # Up
            dy = -1
        elif action == 3: # Down
            dy = 1
        new_x = state.player_1_x + dx
        new_y = state.player_1_y + dy
        new_pos = new_x + 1 + 4 * new_y
        # Check bounds
        if not (0 <= new_x < 4 and 0 <= new_y < 4):
            # Out of bounds, episode ends
            return State(state.player_1, state.player_2, state.opponent, 0,True), -1
        rand_val = random.random()
        if state.possesion == 1:
            # Player 1 has the ball
            # Check for collision or swap with opponent
            if (new_pos == state.opponent) or (state.opponent == state.player_1 and prev_opponent_pos == new_pos):
                if rand_val < 0.5 - p:
                    return State(new_pos, state.player_2, state.opponent, 1), 0
                else:
                    return State(state.player_1, state.player_2, state.opponent, 0,True), -1
            else:
                if rand_val < 1 - 2*p:
                    return State(new_pos, state.player_2, state.opponent, 1), 0
                else:
                    return State(state.player_1, state.player_2, state.opponent, 0,True), -1
        else:
            # Player 1 does not have the ball
            if rand_val < 1 - p:
                return State(new_pos, state.player_2, state.opponent, state.possesion), 0
            else:
                return State(state.player_1, state.player_2, state.opponent, 0,True), -1
    elif action in [4, 5, 6, 7]:
        # Move player 2
        dx, dy = 0, 0
        if action == 4:   # Left
            dx = -1
        elif action == 5: # Right
            dx = 1
        elif action == 6: # Up
            dy = -1
        elif action == 7: # Down
            dy = 1
        new_x = state.player_2_x + dx
        new_y = state.player_2_y + dy
        new_pos = new_x + 1 + 4 * new_y
        # Check bounds
        if not (0 <= new_x < 4 and 0 <= new_y < 4):
            # Out of bounds, episode ends
            return State(state.player_1, state.player_2, state.opponent, 0,True), -1
        rand_val = random.random()
        if state.possesion == 2:
            # Player 2 has the ball
            # Check for collision or swap with opponent
            if (new_pos == state.opponent) or (state.opponent == state.player_2 and prev_opponent_pos == new_pos):
                if rand_val < 0.5 - p:
                    return State(state.player_1, new_pos, state.opponent, 2), 0
                else:
                    return State(state.player_1, state.player_2, state.opponent, 0,True), -1
            else:
                if rand_val < 1 - 2*p:
                    return State(state.player_1, new_pos, state.opponent, 2), 0
                else:
                    return State(state.player_1, state.player_2, state.opponent, 0,True), -1
        else:
            # Player 2 does not have the ball
            if rand_val < 1 - p:
                return State(state.player_1, new_pos, state.opponent, state.possesion), 0
            else:
                return State(state.player_1, state.player_2, state.opponent, 0,True), -1
            

     #------ PASSING-----------------       
    elif action == 8:
        # Passing action
        if state.possesion == 1:
            passer_x, passer_y = state.player_1_x, state.player_1_y
            receiver_x, receiver_y = state.player_2_x, state.player_2_y
            new_state_possesion = 2
        elif state.possesion == 2:
            passer_x, passer_y = state.player_2_x, state.player_2_y
            receiver_x, receiver_y = state.player_1_x, state.player_1_y
            new_state_possesion = 1
        else:
            # No one has the ball, cannot pass
            return State(state.player_1, state.player_2, state.opponent, 0,True), -1

        # Calculate pass probability
        dist = max(abs(passer_x - receiver_x), abs(passer_y - receiver_y))
        pass_prob = q - 0.1 * dist

        # Check if opponent is in between
        def is_between(ax, ay, bx, by, cx, cy):
            # Check if c is on the line segment ab
            cross = (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)
            if cross != 0:
                return False
            dot = (cx - ax) * (bx - ax) + (cy - ay) * (by - ay)
            if dot < 0:
                return False
            sq_len = (bx - ax) ** 2 + (by - ay) ** 2
            if dot > sq_len:
                return False
            return True

        if is_between(passer_x, passer_y, receiver_x, receiver_y, state.opponent_x, state.opponent_y):
            pass_prob /= 2

        rand_val = random.random()
        if rand_val < pass_prob:
            # Successful pass
            return State(state.player_1, state.player_2, state.opponent, new_state_possesion), 0
        else:
            # Failed pass, game ends
            return State(state.player_1, state.player_2, state.opponent, 0,True), -1

    #------ SHOOTING-----------------   
    elif action == 9:
        # Shooting action
        if state.possesion == 1:
            shooter_x= state.player_1_x
        elif state.possesion == 2:
            shooter_x = state.player_2_x
        else:
            # No one has the ball, cannot shoot
            return State(state.player_1, state.player_2, state.opponent, 0,True), -1

        # Calculate base probability
        goal_prob = q - 0.2 * (3 - shooter_x)

        # Check if opponent is in front of the goal (positions 8 or 12)
        if state.opponent in [8, 12]:
            goal_prob /= 2

        rand_val = random.random()
        if rand_val < goal_prob:
        
           # logging.debug(f"Positive reward")

            # Goal scored, possession stays with shooter (could be changed as per game rules)
            return State(state.player_1, state.player_2, state.opponent, state.possesion,True), 20
        else:
            # Shot failed, game ends
            return State(state.player_1, state.player_2, state.opponent, 0,True), -1




def play(game_number:int,Opponent_policy,Value_function=defaultdict(float),Policy=defaultdict(int) ,initial_state=State(5,9,8,1),p=0.1,q=0.6,learning_rate=0.1,discount_factor=1.0,exploration_factor=0.8):

    epsolin= exploration_factor  # exploration factor for epsilon-greedy policy
    current_state=initial_state
    visit_count = defaultdict(int)  # to keep track of state visits
    while not current_state.game_over:
        visit_count[current_state] += 1
        if random.random() < epsolin:
            current_action = random.randint(0, 9)
        else:
            current_action = Policy[current_state]
        next_state,reward= take_action(current_state,current_action,Opponent_policy(current_state),p=p,q=q)
        logging.debug(f" Current state:{current_state.__str__()} , Action taken by Opponent={Opponent_policy(current_state)} Action taken by Agent={current_action} \n")

        alpha= learning_rate/(1+visit_count[current_state]**0.5)  # learning rate decreases with more visits
        #Implement TD(0)
        Value_function[current_state]+= alpha * (reward + discount_factor * Value_function[next_state] - Value_function[current_state])

        current_state = next_state

        # Check if the game is over
        if current_state.game_over:
            logging.debug(f"Game{game_number} ended. Final state: {current_state.__str__()} with reward: {reward}")

    return reward ,current_state,Value_function,Policy
