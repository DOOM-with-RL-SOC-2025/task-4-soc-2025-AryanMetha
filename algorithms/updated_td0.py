#This is where the magic happens
#GOAL- find the optimal policy for given initial state using TD(0) algorithm 
from definitions import *
import logging
logging.basicConfig(level=logging.INFO)
from numpy import argmax
import math
import tqdm

samples=10 # numbers of saples taken per action to estimate q(s,a)


def get_epsilon(episode, ε_start=0.7, ε_min=0.01, decay_rate=0.03):
    return max(ε_min, ε_start / (1 + decay_rate * math.log(1 + episode)))

def find_optimal_policy_updated(
    Opponent_policy,
    Policy=defaultdict(int),
    Value_function=defaultdict(float),
    initial_state=State(5, 9, 8, 1),
    p=0.1, q=0.6,
    learning_rate=0.1, discount_factor=0.9
):
    num_episodes = 10000
    cumulative_reward_per_episode = []
    samples = 10  # number of samples per action for estimating Q(s,a)

    for episode in tqdm.tqdm(range(num_episodes), desc="Episodes"):
        eps = get_epsilon(episode)


        
        reward, final_state, Value_function, Policy = play(
            game_number=episode,
            Opponent_policy=Opponent_policy,
            Value_function=Value_function,
            Policy=Policy,
            initial_state=initial_state,
            p=p, q=q,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_factor=eps
        )

        cumulative_reward_per_episode.append(reward)
        if episode % 50 == 0:
            logging.info(f"Episode: {episode}, Cumulative Reward: {reward}")

        # Policy Improvement Step
        for state in Value_function.keys():
            if state.game_over:
                continue

            q_values = []
            for action in range(10):
                expected_q = 0
                for _ in range(samples):
                    next_state, r = take_action(state, action, Opponent_policy(state), p, q)
                    expected_q += r + discount_factor * Value_function.get(next_state, 0)
                q_values.append(expected_q / samples)

            Policy[state] = argmax(q_values)

    return cumulative_reward_per_episode, Policy, Value_function
