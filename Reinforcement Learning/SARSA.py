from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":




    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0
       
        #TODO perform SARSA learning
        obs = env.reset()
        done = False
        obs_prime = None
        action = None
        reward = 0
        while(done == False):
            p = np.random.random_sample()
            if p < EPSILON: # random action
                action_prime = env.action_space.sample()  
            else:  # best action
                prediction = np.array([Q_table[(obs_prime,i)] for i in range(env.action_space.n)])
                action_prime =  np.argmax(prediction)
                
            obs_primeprime, reward_prime, done, info = env.step(action_prime)
            episode_reward += reward_prime
            if not done:
                Q_table[(obs, action)] += LEARNING_RATE*(reward + DISCOUNT_FACTOR*Q_table[(obs_prime, action_prime)] - Q_table[(obs, action)])
            if done:
                Q_table[(obs_prime, action_prime)] += LEARNING_RATE*(reward_prime - Q_table[(obs_prime, action_prime)])
            obs = obs_prime
            obs_prime = obs_primeprime
            action = action_prime
            reward = reward_prime

        # appending the reward in this episode
        episode_reward_record.append(episode_reward)

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )

        # At the end of episode update epsilon?
        EPSILON = EPSILON * EPSILON_DECAY
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



