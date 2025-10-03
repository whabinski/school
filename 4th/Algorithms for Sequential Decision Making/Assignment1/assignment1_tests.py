from multi_armed_bandit import *
from assignment1 import uniform_sampling
from assignment1 import successive_elimination
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

    