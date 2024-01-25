from bad_sorts import *
from good_sorts import *
import matplotlib.pyplot as plot

# ******************* Dual Quicksort *******************

def dual_quicksort(L):
    copy = dual_quicksort_copy(L)
    for i in range(len(L)):
        L[i] = copy[i]

def dual_quicksort_copy(L):
    if len(L) < 2:
        return L

    if (L[0] <= L[1]):
        smallerPivot = L[0]
        greaterPivot = L[1]
    else: 
        smallerPivot = L[1]
        greaterPivot = L[0]
    
    left, middle, right = [], [], []

    for num in L[2:]:
        if num < smallerPivot:
            left.append(num)
        elif num > greaterPivot:
            right.append(num)
        else:
            middle.append(num)
    
    return quicksort_copy(left) + [smallerPivot] + quicksort_copy(middle) + [greaterPivot] + quicksort_copy(right)

# ******************* Experiement *******************

quickData = []
dual_quickData = []

tests = 100

increments = 0
totalElements = 1000

total1 = 0
total2 = 0
total3 = 0

while increments <= totalElements:

    for _ in range(tests):
        L = create_random_list(totalElements, totalElements)
        L2 = L.copy()

        start = timeit.default_timer()
        quicksort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        dual_quicksort(L2)
        end = timeit.default_timer()
        total2 += end - start
    
    quickData.append(total1/tests)
    dual_quickData.append(total2/tests)
    increments += 100

# ******************* MatPlot *******************
plot.plot([100*i for i in range(11)], quickData, label='Quick Sort')
plot.plot([100*i for i in range(11)], dual_quickData, label='Dual Quick Sort')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Original vs Optimized Quick Sort Test")
plot.legend()
plot.show()