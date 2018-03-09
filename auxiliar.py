import tensorflow as tf
import numpy as np
import random

def discount_rewards(rewards, discount_factor):
    dis_rewards = []
    prev_dis_rew = 0
    for r in reversed(rewards):
        prev_dis_rew = r + prev_dis_rew * discount_factor
        dis_rewards.append(prev_dis_rew)
        
    dis_rewards = dis_rewards[::-1]
    
    return dis_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    return [(discounted_rewards - flat_rewards.mean())/flat_rewards.std() for discounted_rewards in all_discounted_rewards]

def get_batch(dataset, batch_size):
    dataset = np.array(dataset)
    return dataset[random.sample(range(0, len(dataset)), batch_size)]