# Name: Wyatt Habinski
# MacID: 400338858

import numpy as np
from multi_armed_bandit import *

def explore_then_commit(list_of_arms, T):
    """Explore-Then-Commit algorithm for multi-armed bandits.

    Args:
        list_of_arms (list): A list of arms.
        T (int): Number of rounds.

    Returns:
        pulls: A dictionary with arms as keys and the number of pulls as values.

    """
    pulls = {arm: 0 for arm in list_of_arms}        # initialize dictionary of number of pulls for each arm
    rewards = {arm: [] for arm in list_of_arms}     # initialize dictionary of rewards for each arm

    n = len(list_of_arms)                               # number of arms n
    m = int(2 * ((T/n)**(2/3)) * np.log(T)**(1/3))      # number of exploration pulls per arm m
    
    for arm in list_of_arms:                # iterate through all arms
        for _ in range(m):                  # perform all m exploration pulls for each arm
            reward = arm.pull()             # pull arm and get reward
            rewards[arm].append(reward)     # add to list of respective arm's rewards
            pulls[arm] += 1                 # increase running total of times respective arm has been pulled

    empirical_means = {arm: np.mean(rewards[arm]) for arm in list_of_arms}      # compute and store empirical means for each arm in dictionary
    
    best_arm = max(empirical_means, key = empirical_means.get)      # select the arm with highest empirical mean
    
    exploitation_pulls = T - (m*n)          # calculate remaining number of exploitation pulls 
    for _ in range(exploitation_pulls):     # perform all remaining exploitation pulls
        reward = best_arm.pull()            # pull arm and get reward
        pulls[best_arm] += 1                # increase running total of times best arm has been pulled

    return pulls    # return pulls per arm dictionary


def ucb(list_of_arms, T):
    """Upper Confidence Bound (UCB) algorithm for multi-armed bandits.

    Args:
        list_of_arms (list): A list of arms.
        T (int): Number of rounds.

    Returns:
        pulls: A dictionary with arms as keys and the number of pulls as values.
        last_pull_time: A dictionary with arms as keys and the time of the last pull as values.
    """

    pulls = {arm: 0 for arm in list_of_arms}                # initialize dictionary of pulls for each arm
    last_pull_time = {arm: 0 for arm in list_of_arms}       # initialize dictionary of last pull time for each arm
    
    rewards = {arm: 0 for arm in list_of_arms}              # initialize dictionary of rewards for each arm
    empirical_means = {arm: [] for arm in list_of_arms}     # initialize dictionary of empirical means for each arm
    
    # pull each arm once to begin
    for arm in list_of_arms:            # iterate through all arms
        reward = arm.pull()             # pull arm and get reward
        pulls[arm] += 1                 # increase running total of times respective arm has been pulled
        rewards[arm] += reward          # add reward to respective arms total reward
        empirical_means[arm] = rewards[arm] / pulls[arm]    # calculate empirical mean for respective arm
        last_pull_time[arm] = 1         # update last time respective arm was pulled
    
    t = len(list_of_arms)               # initialize number of total pulls to n
    
    while t < T:                                        # iterate through all pulls until total pulls T is reached
        t += 1                                          # increment t by 1
        ucbs = {arm: 0 for arm in list_of_arms}         # initialize dictionary of ucbs for each arm
    
        # calculate ucb for each arm
        for arm in list_of_arms:                        # iterate through each arm
            m_t = pulls[arm]                            # get number of pulls for respective arm m_t
            r_t = np.sqrt(2 * np.log(T) / m_t)          # calculate confidence radius r_t for respective arm
            ucbs[arm] = empirical_means[arm] + r_t      # calculate ucb for respective arm and add to dictionary
        
        best_arm = max(ucbs, key = ucbs.get)        # select the arm with highest ucb
        
        reward = best_arm.pull()                    # pull best arm and get reward
        pulls[best_arm] += 1                        # increase running total of times best arm has been pulled
        rewards[best_arm] += reward                 # add reward to best arms total reward
        empirical_means[best_arm] = rewards[best_arm] / pulls[best_arm]     # calculate empirical mean for best arm
        last_pull_time[best_arm] = t                # update last time best arm was pulled
    
    return pulls, last_pull_time    # return dictionary of number of pulls for each arm and dictionary of last time each arm was pulled


def calculate_regret(true_means, total_pulls):
    mu_star = max(true_means.values())
    regret = 0
    for arm in total_pulls:
        pulls = total_pulls[arm]
        regret += (mu_star - true_means[arm]) * pulls
    return regret
