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
 
def find_optimal_policy(Opponent_policy,Policy=defaultdict(int) ,Value_function = defaultdict(float),initial_state=State(5,9,8,1), p=0.1, q=0.6, learning_rate=0.1, discount_factor=0.9):
  
    #number of Td(0 updates) 
    num_updates=10
    #number of episodes
    num_episodes=5000

    cumulative_reward_per_episode =[]
    for episode in tqdm.tqdm(range(num_episodes), desc="Episodes"):
        cumulative_reward=0
        
        #policy interation
        for iteration in range(num_updates):
            if episode%10 ==0 and iteration==num_updates-1:
                logging.info(f"Episode: {episode}, Iteration: {iteration}, Cumulative Reward: {cumulative_reward}")



            reward,state_info,Value_function,Policy= play(iteration, Opponent_policy,Value_function,Policy,initial_state,p,q,learning_rate,discount_factor,exploration_factor=get_epsilon(episode))
            cumulative_reward += reward
        
        cumulative_reward_per_episode.append(cumulative_reward)
        #policy improvement
        for state in Value_function.keys():
            #find the best action for the current state
            if state.game_over:
                continue
            q_values = [0 for _ in range(10)]

            for possible_action in range(10):
                # try out each action #samples times
                for _ in range(samples):
                    next_state, reward = take_action(state, possible_action, Opponent_policy(state), p=p, q=q)
                    # Only use Value_function[next_state] if next_state is already present
                    
                    if next_state in Value_function:
                        q_values[possible_action] += reward + discount_factor * Value_function[next_state]
                    else:
                        q_values[possible_action] += reward  # ignore value function for unseen states
                q_values[possible_action] /= samples

            Policy[state] = argmax(q)

    return cumulative_reward_per_episode, Policy, Value_function
        
       












