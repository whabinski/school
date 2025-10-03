import matplotlib.pyplot as plot
import random

from BST import *
from lab6 import *

# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

# Creates a near sorted list by creating a random list, sorting it, then doing a random number of swaps
def create_near_sorted_list(length, max_value, swaps):
    L = create_random_list(length, max_value)
    L.sort()
    for _ in range(swaps):
        r1 = random.randint(0, length - 1)
        r2 = random.randint(0, length - 1)
        L[r1], L[r2] = L[r2], L[r1]
    return L

diffData = []

tests = 100
totalSwaps = 600
currentSwaps = 0
totalElements = 550

total1 = 0
total2 = 0

while currentSwaps <= totalSwaps:

    for _ in range(tests):
        L = create_near_sorted_list(totalElements,totalElements,currentSwaps)

        t1 = BST()
        t2 = RBTree()

        for x in L:
            t1.insert(x)
            t2.insert(x)

        total1 += t1.get_height()
        total2 += t2.get_height()
        
    
    
    diffData.append(total1/total2)
    currentSwaps += 60


# ******************* MatPlot *******************

plot.plot([60*i for i in range(11)], diffData, label="")
plot.xlabel('Number of Swaps')
plot.ylabel('Difference in Height of Tree')
plot.title("Height of Tree vs Swaps")
plot.legend()
plot.show()
