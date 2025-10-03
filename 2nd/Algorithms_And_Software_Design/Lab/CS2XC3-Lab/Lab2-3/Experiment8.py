from bad_sorts import *
from good_sorts import *
import matplotlib.pyplot as plot

# ******************* Experiment8 Testing When Insertion Sort is Superior *******************

insertionData = []
quickData = []
mergeData = []

totalElements = 17
increments = 0
tests = 100

total1 = 0
total2 = 0
total3 = 0

while increments <= totalElements:

    for _ in range(tests):
        L = create_random_list(increments, increments)
        L2 = L.copy()
        L3 = L.copy()

        start = timeit.default_timer()
        insertion_sort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        quicksort(L2)
        end = timeit.default_timer()
        total2 += end - start

        start = timeit.default_timer()
        mergesort(L3)
        end = timeit.default_timer()
        total3 += end - start
    
    insertionData.append(total1/tests)
    quickData.append(total2/tests)
    mergeData.append(total3/tests)
    increments += 1

# ******************* MatPlot *******************
# **** Comment out to run the inferior graph ****

plot.plot([i for i in range(18)], insertionData, label='Insertion Sort')
plot.plot([i for i in range(18)], quickData, label='Quick Sort')
plot.plot([i for i in range(18)], mergeData, label='Merge Sort')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Insertion vs Quick vs Merge Sorts")
plot.legend()
plot.show()

# ******************* Experiment8 Testing When Insertion Sort is Inferior *******************

insertionData = []
quickData = []
mergeData = []

totalElements = 70
increments = 0
tests = 100

total1 = 0
total2 = 0
total3 = 0

while increments <= totalElements:

    for _ in range(tests):
        L = create_random_list(increments, increments)
        L2 = L.copy()
        L3 = L.copy()

        start = timeit.default_timer()
        insertion_sort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        quicksort(L2)
        end = timeit.default_timer()
        total2 += end - start

        start = timeit.default_timer()
        mergesort(L3)
        end = timeit.default_timer()
        total3 += end - start
    
    insertionData.append(total1/tests)
    quickData.append(total2/tests)
    mergeData.append(total3/tests)
    increments += 7

# ******************* MatPlot *******************
# **** Comment out to run the superior graph ****

plot.plot([7*i for i in range(11)], insertionData, label='Insertion Sort')
plot.plot([7*i for i in range(11)], quickData, label='Quick Sort')
plot.plot([7*i for i in range(11)], mergeData, label='Merge Sort')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Insertion vs Quick vs Merge Sorts")
plot.legend()
plot.show()