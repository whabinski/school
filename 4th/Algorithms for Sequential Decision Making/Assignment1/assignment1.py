# Name: Wyatt Habinski
# MacID: 400338858

import numpy as np

def uniform_sampling(arms, alpha, epsilon_0, delta_0, pull_reduction=1):
    """The uniform sampling algorithm
    Args:
        arms: list of arm
        alpha: maximal arm reward
        epsilon_0: error
        delta_0: failure probability
        pull_reduction: factor to reduce the number of pulls
    Returns:
        best_arm_index: the index of best arm in the input list (1-based)
        num_pulls: total number of pulls that was used by the algorithm
    """
    
    best_arm_index = None       # return value for best arms index
    num_pulls = 0               # return value for total number of arms pulled throughout entire algorithm
    
    n = len(arms)               # number of arms n
    epsilon = epsilon_0/2       # epsilon
    delta = delta_0/n           # delta
    
    m = int((((alpha**2) * np.log(2/delta)) / (2 * epsilon**2)) / pull_reduction)   # number of arms pulled each round m
    
    rewards = {a.index:[] for a in arms}        # initialize dictionary to store each arms rewards across all pulls
    
    for t in range(m):                          # iterate through all m arm pulls
        for a in arms:                          # iterate through all n arms
            r = a.pull()                        # pull an arm and recieve reward
            rewards[a.index].append(r)          # add reward to respective arm in dictionary
            num_pulls += 1                      # increase running count of total arm pulls
        
    empirical_means = {a.index: np.mean(rewards[a.index]) for a in arms}        # calculate empirical mean for all arms and store in dictionary
    best_arm_index = max(empirical_means, key=empirical_means.get)              # get index for arm with highest empirical mean

    return best_arm_index, num_pulls    # return best arm index and total number of arm pulls


def successive_elimination(arms, alpha, list_of_gaps, delta_0, pull_reduction=1):
    """The successive elimination algorithm
    Args:
        arms: list of arm
        alpha: maximal arm reward
        list_of_gaps: list of gaps between the best arm and the other arms; Delta_i for i >= 2 will be in index [i] of this list. The values in locations [0] and [1] are ignored by this function.
        delta_0: failure probability
        pull_reduction: factor to reduce the number of pulls

    Returns:
        best_arm_index: the index of best arm in the input list (1-based)
        num_pulls: total number of pulls that was used by the algorithm
    """
    
    best_arm_index = None       # return value for best arms index
    num_pulls = 0               # return value for total number of arms pulled throughout entire algorithm
    removed_arms = []           # list of eliminated arms

    n = len(arms)               # number of arms n
    S = set(arms)               # create set of arms
    
    t = 0                       
    T_0 = 0
    T_j = T_0
    
    for i in range(1,n):                        # iterate through elimination rounds
        if len(S) == 1:                         # return the last remaining arm
            break
        
        epsilon_i = list_of_gaps[n-i+1]/2       # episolon
        delta_i = delta_0/(2*(n-1))             # delta
        
        m = (((alpha**2) * np.log(2/delta_i)) / (2 * epsilon_i**2))     # calculate m(episol, delta, alpha)

        T_i = max(1, int((m - T_j) / pull_reduction))   # number of arms pulled each round. Subtract m by T_j for (episol, delta)-PAC estimate. Ensuring > 0
        T_j += T_i                                      # keep running total of T_j
        t = 0                                           # resest number of arm pulls per eliminatioin round
        
        rewards = {a.index: [] for a in S}      # initialize reward dictionary every new elimination round
        
        for i in range(T_i):                    # iterate through all T_i arm pulls
            t += 1                              # keep running count of number of arm pulls
            for a in S:                         # iterate through every arm
                r = a.pull()                    # pull arm a and recieve reward r
                rewards[a.index].append(r)      # add reward to respective arm in dictionary
                num_pulls += 1                  # increase running count of total arm pulls for entire algorithm

        empirical_means = {a.index: np.sum(rewards[a.index]) / t for a in S}    # calculate empirical mean for all arms and store in dictionary

        worst_arm_index = min(empirical_means, key=empirical_means.get)         # find worst empirical mean in current set
        
        S = {a for a in S if a.index != worst_arm_index}        # remove element from set if it is the worst arm
        removed_arms.append(worst_arm_index)                    # add removed arm to list of removed arms

    best_arm_index = list(S)[0].index       # the best arm is the last remaining arm

    return best_arm_index, num_pulls #, removed_arms   # return best arm index and total number of algorithm arm pulls (optional: removed_arms)



#-----------------------------------------------------------------------------------------------------------------------------------
# Tests for Question 4
#-----------------------------------------------------------------------------------------------------------------------------------

from multi_armed_bandit import *
import time

def single_tests(list_of_arms, alpha, epsilon_0, delta_0, gaps, pull_reductions=1):
        true_means = [arm._get_true_mean() for arm in list_of_arms]

        print("Expected output:")
        print("The true means of the bandits are ", true_means)
        print(
            "The index of the best bandit is",
            list_of_arms[np.argmax(true_means)].index,
            "with true mean",
            max(true_means),
        )
        print("====================================")
        print("Running Uniform sampling algorithm")
        start = time.time()
        best_arm_1, num_pulls = uniform_sampling(
            list_of_arms, alpha, epsilon_0, delta_0, pull_reduction=1
        )
        print("Time taken: ", time.time() - start)
        print("Uniform sampling Algorithm's output:")
        print("The index of the best bandit is ", best_arm_1)
        print("Total number of pulls: ", num_pulls)
        print("====================================")
        print("Running Successive elimination algorithm")
        start = time.time()
        best_arm_2, num_pulls_2 = successive_elimination(
            list_of_arms, alpha, gaps, delta_0, pull_reduction=1
        )
        print("Time taken: ", time.time() - start)
        print("Successive elimination algorithm's output:")
        print("The index of the best bandit is ", best_arm_2)
        print("Total number of pulls: ", num_pulls_2)

def batch_tests(batches, best_arm_index, list_of_arms, alpha, epsilon_0, delta_0, gaps, pull_reductions=1):
    
    successful_runs_uniform = 0
    successful_runs_elimination = 0
    
    for batch in range (batches):
            best_arm_uniform, num_pulls_uniform = uniform_sampling(list_of_arms, alpha, epsilon_0, delta_0, pull_reduction=1)
            best_arm_elimination, num_pulls_eliminiation = successive_elimination(list_of_arms, alpha, gaps, delta_0, pull_reduction=1)
            
            if best_arm_uniform == best_arm_index:
                successful_runs_uniform += 1
                
            if best_arm_elimination == best_arm_index:
                successful_runs_elimination +=1
            
    print("Total Runs: ", batches)
    print("Successfule Uniform Sampling Runs: ", successful_runs_uniform)
    print("Successfule Successive Elimination Runs: ", successful_runs_elimination)

def removal_sequence_test(batches, best_arm_index, list_of_arms, alpha, delta_0, gaps, pull_reductions=1):
    
    successful_runs_elimination = 0

    for batch in range (batches):
            best_arm_elimination, num_pulls_eliminiation, removed_arms = successive_elimination(list_of_arms, alpha, gaps, delta_0, pull_reduction=1)
                
            if best_arm_elimination == best_arm_index:
                successful_runs_elimination +=1
                print(removed_arms)
                    

if __name__ == "__main__":
    # Define the arms
    list_of_arms = [
        bernoulli_arm(0.65, index=1),
        bernoulli_arm(0.66, index=2),
        gaussian_arm(9, 2, index=3),
        gaussian_arm(8.9, 2.2, index=4),
        uniform_arm_continuous(0, 1.4, index=5),
        uniform_arm_continuous(0, 4, index=6)
    ]

    # Define the parameters
    alpha = 15.5
    epsilon_0 = 0.1
    delta_0 = 0.2

    # The below list of gaps starting from Delta_0.
    gaps = [None, 0, 0.1, 7, 8.3, 8.34, 8.35]

    #single_tests(list_of_arms, alpha, epsilon_0, delta_0, gaps, pull_reductions=1)
    #batch_tests(100, 3, list_of_arms, alpha, epsilon_0, delta_0, gaps, pull_reductions=20)
    
    # need to add removed_arms to return statement of successive_elimination
    #removal_sequence_test(100, 3, list_of_arms, alpha, delta_0, gaps, pull_reductions=1)

    