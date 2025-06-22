# This is where the magic happens
# GOAL - find the optimal policy for given initial state using Monte Carlo algorithm 
from definitions import *
import logging
import tqdm
import random
from collections import defaultdict
from numpy import argmax

logging.basicConfig(level=logging.INFO)

def optimal_policy_monte_carlo(
        Oponent_policy,
        Initial_state: State,
        p, q,
        episodes=100000,
        discount_factor=1,
        epsilon=0.1
):
    Q_value_estimates = defaultdict(float)  # maps (s,a) -> scalar
    Returns = defaultdict(list)             # maps (s,a) -> list of returns
    policy = defaultdict(lambda: None)
    cumulative_reward=0

    def initialize_policy(state):
        actions = list(range(10))
        prob = 1 / len(actions)
        return {a: prob for a in actions}

    for episode in tqdm.tqdm(range(episodes), desc="Episode:"):
        current_state = Initial_state.clone()  # clone if needed to reset environment
        game_sequence = []

        if episode%100:
            logging.debug(f" Episode{episode}, Cumulative reward {cumulative_reward}")
            

        # === Generate one episode ===
        while not current_state.game_over:
            if policy[current_state] is None:
                policy[current_state] = initialize_policy(current_state)

            actions, probs = zip(*policy[current_state].items())
            action = random.choices(actions, weights=probs)[0]
            opponent_action = Oponent_policy(current_state)
            next_state, reward = take_action(current_state, action, opponent_action, p, q)
            game_sequence.append((current_state, action, reward))
            cumulative_reward+=reward
            current_state = next_state

        # === Monte Carlo update (First-Visit) ===
        G = 0
        visited_pairs = set()

        for state, action, reward in reversed(game_sequence):
            G = discount_factor * G + reward

            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                Returns[(state, action)].append(G)
                Q_value_estimates[(state, action)] = sum(Returns[(state, action)]) / len(Returns[(state, action)])

                actions = list(range(10))
                if state.game_over:
                    continue
                best_action = max(actions, key=lambda a: Q_value_estimates[(state, a)])
                n = len(actions)
                policy[state] = {
                    a: (1 - epsilon + epsilon / n) if a == best_action else (epsilon / n)
                    for a in actions
                }

    # === Return the final greedy policy ===
    def final_policy(state):
        actions = list(range(10))
        q_vals = [Q_value_estimates[(state, a)] for a in actions]
        return actions[argmax(q_vals)]

    return final_policy,cumulative_reward
