from multi_armed_bandit import *
from assignment2 import explore_then_commit
from assignment2 import ucb
from assignment2 import calculate_regret
import time
import numpy as np
import matplotlib.pyplot as plt

def regret_tests(list_of_arms, T):
    true_means = {arm: arm._get_true_mean() for arm in list_of_arms}

    print("Expected output:")
    print("The true means of the bandits are ", true_means)
    print(
        "The index of the best bandit is",
        list_of_arms[np.argmax(true_means)].index,
        "with true mean",
        max(true_means, key=true_means.get),
    )
    print("====================================")
    
    print("Running Explore then Commit Algorithm")
    start = time.time()
    etc_pulls = explore_then_commit(list_of_arms, T)
    print("Time taken: ", time.time() - start)
    print("Explore then Commit's output:")
    print("The pulls per arm are: ", etc_pulls)
    etc_regret = calculate_regret(true_means, etc_pulls)
    print("Regret for Explore then Commit: ", etc_regret)
    print("====================================")
    
    print("Running UCB Algorithm")
    start = time.time()
    ucb_pulls, last_pull_times = ucb(list_of_arms, T)
    print("Time taken: ", time.time() - start)
    print("UCB's output:")
    print("The pulls per arm are: ", ucb_pulls)
    print("The last pull times are:: ", last_pull_times)
    ucb_regret = calculate_regret(true_means, ucb_pulls)
    print("Regret for UCB: ", ucb_regret)
    print("====================================")

    return etc_regret, ucb_regret


def etc_ucb_regret_comparison():

    list_of_arms = [
        bernoulli_arm(0.6, index=1),
        bernoulli_arm(0.48, index=2),
        bernoulli_arm(0.5, index=3)
    ]

    etc_regrets = []
    ucb_regrets = []

    Ts = np.arange(500, 20001, 500)
    
    for T in Ts:
        etc_regret, ucb_regret = regret_tests(list_of_arms, T)
        etc_regrets.append(etc_regret)
        ucb_regrets.append(ucb_regret)
    
    return etc_regrets, ucb_regrets, Ts

def plot_comparison(x_axis1, x_axis2, y_axis1, y_axis2 ):
    
    plt.figure(figsize=(14, 8))
    plt.plot(x_axis1, y_axis1, marker='o', label='Explore-Then-Commit')
    plt.plot(x_axis2, y_axis2, marker='s', label='UCB')
    
    plt.xlabel("Time Horizon (T)")
    plt.ylabel("Cumulative Regret")
    plt.title("ETC vs UCB Algorithm Regret Comparison")
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_log_log_comparison(x_axis1, x_axis2, y_axis1, y_axis2 ):
    
    plt.figure(figsize=(14, 8))
    plt.loglog(x_axis1, y_axis1, marker='o', label='Explore-Then-Commit')
    plt.loglog(x_axis2, y_axis2, marker='s', label='UCB')
    
    plt.xlabel("Log(Time Horizon T)")
    plt.ylabel("Log(Cumulative Regret)")
    plt.title("Log-Log ETC vs UCB Algorithm Regret Comparison")
    plt.legend()
    plt.grid()
    plt.show()
    
def UCB_last_pulls_analysis():
    list_of_arms = [
    bernoulli_arm(0.09, index=1),
    uniform_arm_continuous(0, 0.3, index=2),
    uniform_arm_continuous(0, 0.35, index=3),
    uniform_arm_continuous(0, 0.4, index=4),
    uniform_arm_continuous(0, 0.8, index=5),
    uniform_arm_continuous(0.5, 0.6, index=6)
    ]
    
    true_means = {arm: arm._get_true_mean() for arm in list_of_arms}

    print("Expected output:")
    print("The true means of the bandits are \n\n", true_means)

    T = 10000
    ucb_pulls, last_pull_times = ucb(list_of_arms, T)

    # Print results
    print("\n\nLast time each arm was pulled by UCB:")
    for arm in sorted(last_pull_times, key=last_pull_times.get, reverse=True):
        print(f"Arm {arm.index}: Last pulled at t = {last_pull_times[arm]}")

    # Determine the order of elimination
    sorted_eliminations = sorted(last_pull_times.items(), key=lambda x: x[1])
    elimination_order = [arm.index for arm, _ in sorted_eliminations]

    print("\nOrder of eliminations based on last pull time:")
    print(elimination_order)

if __name__ == "__main__":
    #etc_regrets, ucb_regrets, Ts = etc_ucb_regret_comparison()
    #plot_comparison(Ts, Ts, etc_regrets, ucb_regrets)
    #plot_log_log_comparison(Ts, Ts, etc_regrets, ucb_regrets)
    
    UCB_last_pulls_analysis()
    
        
