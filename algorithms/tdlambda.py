from collections import defaultdict
from numpy import argmax
import math
import random
import tqdm
import logging
from definitions import State, take_action

logging.basicConfig(level=logging.INFO)

# Slow-decaying exploration rate function
def get_exploration_rate(episode_number, start=0.7, min_value=0.01, decay_rate=0.03):
    return max(min_value, start / (1 + decay_rate * math.log(1 + episode_number)))


def find_optimal_policy_td_lambda(
    Opponent_policy,
    Policy=defaultdict(int),
    Value_function=defaultdict(float),
    initial_state=None,
    p=0.1,                          # slip probability
    q=0.6,                          # shot accuracy
    learning_rate=0.1,
    discount_factor=0.9,
    trace_decay=0.8,               # lambda
    
    total_episodes=10000,
    samples_per_action=10
):
    if initial_state is None:
        initial_state = State(5, 9, 8, 1)  # default position

    cumulative_rewards = []

    for episode in tqdm.tqdm(range(total_episodes), desc="Episodes"):
        total_reward = 0
        exploration_rate = get_exploration_rate(episode)
        eligibility_trace = defaultdict(float)
        current_state = initial_state

        while not current_state.game_over:
            if episode % 20 == 0 :
                logging.info(f"Episode: {episode}, Total Reward: {total_reward}")

            # Choose action (epsilon-greedy)
            if random.random() < exploration_rate:
                chosen_action = random.randint(0, 9)
            else:
                chosen_action = Policy[current_state]

            # Take the action and observe result
            opponent_action = Opponent_policy(current_state)
            next_state, reward = take_action(current_state, chosen_action, opponent_action, p=p, q=q)

            total_reward += reward

            # TD Target and Error
            predicted_value = Value_function[current_state]
            next_value = Value_function[next_state]
            target_value = reward + discount_factor * next_value
            prediction_error = target_value - predicted_value

            # Eligibility trace update
            eligibility_trace[current_state] += 1

            # Value updates for all visited states
            for state in eligibility_trace:
                Value_function[state] += learning_rate * prediction_error * eligibility_trace[state]
                eligibility_trace[state] *= discount_factor * trace_decay

            current_state = next_state

            if current_state.game_over:
                break

        cumulative_rewards.append(total_reward)

        # Policy Improvement
        for state in list(Value_function.keys()):
            if state.game_over:
                continue

            estimated_action_values = [0.0 for _ in range(10)]

            for action in range(10):
                for _ in range(samples_per_action):
                    opponent_action = Opponent_policy(state)
                    next_state, reward = take_action(state, action, opponent_action, p=p, q=q)
                    estimated_action_values[action] += reward + discount_factor * Value_function.get(next_state, 0)

                estimated_action_values[action] /= samples_per_action

            Policy[state] = argmax(estimated_action_values)

    return cumulative_rewards, Policy, Value_function
