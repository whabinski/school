from BST import *
from lab6 import *

import matplotlib.pyplot as plot
import random

# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

totalElements = 10000
tests = 100

total1 = 0
total2 = 0


for _ in range(tests):
    L = create_random_list(totalElements, totalElements)

    t1 = BST()
    t2 = RBTree()

    for x in L:
        t1.insert(x)
        t2.insert(x)

    total1 += t1.get_height();
    total2 += t2.get_height();


print("BST Average Height: ", total1/tests)
print("RBT Average Height: ", total2/tests)

'''
L = [10, 5,30, 40, 9, 2,]

t1 = BST()
t2 = RBTree()

for x in L:
    t1.insert(x)
    t2.insert(x)

print(t1.get_height())
print(t1)
print(t2.get_height())
print(t2)
'''