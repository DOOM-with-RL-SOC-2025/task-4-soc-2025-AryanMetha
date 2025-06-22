from definitions import State,take_action,play
import logging
from collections import defaultdict
import tqdm
import random
from math import log
from numpy import argmax,exp




def find_best_value(
        opponent_policy,
        initial_state:State,  # Initial state for the game
        episodes,
        p,q,
        decay=0.1,
        discount=1,
        epison_0=0.75,
        exploration_decay=0.05 ,# Decay factor for epsilon:::My philoshophy is that it should start with a high value and decay over time, allowing for exploration in the beginning and exploitation later on
        sample_actions=20
):
    V_s= defaultdict(float)  # Value function for states

    
    def my_policy(state, V_s, epsilon_i,episode_number):

        #epsilon = max(0.01, epsilon_i /(1 + exploration_decay * log(1 + episode_number)))
        epsilon = max(0.05, epsilon_i * exp(-episode_number / 150000))


        if episode_number < 50000:
            sample_actions = 10
        elif episode_number < 150000:
            sample_actions = 15
        else:
            sample_actions = 20


        possible_actions = [i for i in range(10)]
        action_values = [0 for _ in range(10)]
        for action in possible_actions:
            # Sample actions multiple times to estimate the value of each action
            for _ in range(sample_actions):
                next_state, reward = take_action(state, action, opponent_policy(state), p=p, q=q)
                # Only use V_s[next_state] if next_state is already present
                if next_state in V_s:
                    action_values[action] += reward + discount * V_s[next_state]
                else:
                    action_values[action] += reward
            action_values[action] /= sample_actions  # Average the action values over samples

        best_action = argmax(action_values)

        if random.random() < epsilon:
            # Explore: choose a random action,avoid the best action 
            random_action = random.randint(0, 9)
            while random_action == best_action:
                random_action = random.randint(0, 9)
            return random_action
        else:
            return best_action

    cumulative_reward ,wins,losses= 0,0,0


    
    for episode in tqdm.tqdm(range(episodes), desc="Episodes"):
        current_state=initial_state
        while not current_state.game_over:
            #pick an action 
            action=my_policy(current_state, V_s, epison_0, episode_number=episode)
            #play the game
            previous_state = current_state
            current_state, reward = take_action(current_state, action, opponent_policy(current_state), p=p, q=q)
            #update the value function
            V_s[previous_state] += decay * (reward + discount * V_s[current_state] - V_s[previous_state])

            cumulative_reward += reward
            if reward>0:
                wins += 1
            elif reward < 0:
                losses += 1
        if (episode+1) % 100 == 0:
            logging.info(f"In {episode} to {episode-100} : wins: {wins}, losses: {losses}")
            cumulative_reward = 0
            wins=0
            losses=0
    # Convert defaultdict to a regular dict for easier handling later
    V_s = dict(V_s)
    initial_state_value = V_s.get(initial_state)
    winning_odds = (1+ initial_state_value) / 21 
    return V_s, winning_odds
        