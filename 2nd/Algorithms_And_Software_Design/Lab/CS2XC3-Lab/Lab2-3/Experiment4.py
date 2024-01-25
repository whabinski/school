from bad_sorts import *
from good_sorts import *
import matplotlib.pyplot as plot

quickData = []
heapData = []
mergeData = []

totalElements = 1000
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
        quicksort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        heapsort(L2)
        end = timeit.default_timer()
        total2 += end - start

        start = timeit.default_timer()
        mergesort(L3)
        end = timeit.default_timer()
        total3 += end - start
    
    quickData.append(total1/tests)
    heapData.append(total2/tests)
    mergeData.append(total3/tests)
    increments += 100

# ******************* MatPlot *******************
plot.plot([100*i for i in range(11)], quickData, label='Quick Sort')
plot.plot([100*i for i in range(11)], heapData, label='Heap Sort')
plot.plot([100*i for i in range(11)], mergeData, label='Merge Sort')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Sorting Algorithm Test 2")
plot.legend()
plot.show()