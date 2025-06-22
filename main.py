
from definitions import State, defaultdict, play, take_action
from algorithms.tdlambda import find_optimal_policy_td_lambda
from algorithms.iterations import find_optimal_policy
from algorithms.updated_td0 import find_optimal_policy_updated

import logging
import random
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)v 
#take intial state info
user_input = input("Enter list (e.g.[09,08,12,3]) or 'd' for default: ")

if user_input.strip().lower() == 'd':
    number_list = [8,12,1,1]
else:
    # Remove brackets and split by comma
    user_input = user_input.strip("[]").split(",")
    try:
        number_list = [int(x) for x in user_input]
        if len(number_list) != 4:
            raise ValueError("Please enter exactly four integers.")
    except ValueError:
        print("Invalid input. Please enter a list of integers.")
        exit()

#Inital state defined 
print("Initial state:", number_list)
initial_state=State(*number_list)

# Take p and q as inputs from user
try:
    p = float(input("Enter value for p (0 to 0.5): "))
    q = float(input("Enter value for q (0.6 to 1): "))
    if not (0 <= p <= 0.5):
        raise ValueError("p must be between 0 and 0.5.")
    if not (0.6 <= q <= 1):
        raise ValueError("q must be between 0.6 and 1.")
except ValueError as e:
    print(f"Invalid input: {e}")
    exit()

#Schemnatic Policy for the opponent
def opponent_policy(state):
    moves = []
    x, y = state.opponent_x, state.opponent_y
    if x > 0:
        moves.append(1)  # left
    if x < 3:
        moves.append(2)  # right
    if y > 0:
        moves.append(3)  # up
    if y < 3:
        moves.append(4)  # down
    return random.choice(moves)


#reward_per_episode, optimal_policy, optimal_value_function = find_optimal_policy(opponent_policy, initial_state=initial_state, p=p, q=q)
# reward_per_episode, optimal_policy, optimal_value_function = find_optimal_policy_td_lambda(
#     Opponent_policy=opponent_policy,
#     initial_state=initial_state,
#     p=p,
#     q=q
# )
reward_per_episode, optimal_policy, optimal_value_function = find_optimal_policy_updated(
    Opponent_policy=opponent_policy,
    initial_state=initial_state,
    p=p,
    q=q
)

print("Final value of initial state:", optimal_value_function[initial_state])

# Plotting the cumulative reward per episode
plt.plot(reward_per_episode)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward per Episode')
plt.grid()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.show()