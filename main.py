
# from definitions import State, defaultdict, play, take_action
# from algorithms.tdlambda import find_optimal_policy_td_lambda
# from algorithms.iterations import find_optimal_policy
# from algorithms.updated_td0 import find_optimal_policy_updated
# from algorithms.montecarlo1 import optimal_policy_monte_carlo

import logging
import random
import matplotlib.pyplot as plt
import pickle

from algorithms.finalattempt import find_best_value
from definitions import State
from opponent import random_opponent, greedy_opponent, defensive_opponent


# Set up logging
logging.basicConfig(level=logging.INFO)
#take intial state info
user_input = input("Enter list (e.g.[09,08,12,3]) or 'd' for default: ")

if user_input.strip().lower() == 'd':
    number_list = [5,9,8,1]
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
# try:
#     p = float(input("Enter value for p (0 to 0.5): "))
#     q = float(input("Enter value for q (0.6 to 1): "))
#     if not (0 <= p <= 0.5):
#         raise ValueError("p must be between 0 and 0.5.")
#     if not (0.6 <= q <= 1):
#         raise ValueError("q must be between 0.6 and 1.")
# except ValueError as e:
#     print(f"Invalid input: {e}")
#     exit()

# Ask user to pick an opponent
opponent_choice = input("Choose opponent: 'r' for random, 'g' for greedy, 'd' for defensive: ").strip().lower()
if opponent_choice == 'r':
    opponent_policy = random_opponent
elif opponent_choice == 'g':
    opponent_policy = greedy_opponent
elif opponent_choice == 'd':
    opponent_policy = defensive_opponent
else:
    print("Invalid choice. Please enter 'r', 'g', or 'd'.")
    exit()

#reward_per_episode, optimal_policy, optimal_value_function = find_optimal_policy(opponent_policy, initial_state=initial_state, p=p, q=q)
# reward_per_episode, optimal_policy, optimal_value_function = find_optimal_policy_td_lambda(
#     Opponent_policy=opponent_policy,
#     initial_state=initial_state,
#     p=p,
#     q=q
# )
# reward_per_episode, optimal_policy, optimal_value_function = find_optimal_policy_updated(
#     Opponent_policy=opponent_policy,
#     initial_state=initial_state,
#     p=p,
#     q=q
# )
#print("Final value of initial state:", optimal_value_function[initial_state])
# Graph 1: Varying p, fixed q=0.7
ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
q_fixed = 0.7
winning_chances_p = []
for p_val in ps:
    _, win_chance = find_best_value(
        opponent_policy=opponent_policy,
        initial_state=initial_state,
        episodes=275000,
        p=p_val,
        q=q_fixed,
        decay=0.1,
        discount=1,
        epison_0=0.75,
        exploration_decay=0.05,
        sample_actions=20
    )
    winning_chances_p.append(win_chance * 100)

plt.figure(figsize=(8, 5))
plt.plot(ps, winning_chances_p, marker='o')
plt.xlabel('p')
plt.ylabel('Winning Chances (%)')
plt.title('Winning Chances vs p (q=0.7)')
plt.grid(True)
plt.show()

# Graph 2: Varying q, fixed p=0.3
qs = [0.6, 0.7, 0.8, 0.9, 1]
p_fixed = 0.3
winning_chances_q = []
for q_val in qs:
    _, win_chance = find_best_value(
        opponent_policy=opponent_policy,
        initial_state=initial_state,
        episodes=275000,
        p=p_fixed,
        q=q_val,
        decay=0.1,
        discount=1,
        epison_0=0.75,
        exploration_decay=0.05,
        sample_actions=20
    )
    winning_chances_q.append(win_chance * 100)

plt.figure(figsize=(8, 5))
plt.plot(qs, winning_chances_q, marker='o')
plt.xlabel('q')
plt.ylabel('Winning Chances (%)')
plt.title('Winning Chances vs q (p=0.3)')
plt.grid(True)
plt.show()


# with open("value_function.pkl", "wb") as f:
#     pickle.dump(values, f)

# print("winning chances (%):", winning_chances*100)

# Plotting the cumulative reward per episode

