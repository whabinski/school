from bad_sorts import *
from good_sorts import *
import matplotlib.pyplot as plot

quickData = []
heapData = []
mergeData = []

tests = 100
totalSwaps = 600
currentSwaps = 0
totalElements = 500

total1 = 0
total2 = 0
total3 = 0

while currentSwaps <= totalSwaps:

    for _ in range(tests):
        L = create_near_sorted_list(totalElements,totalElements,currentSwaps)
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
    currentSwaps += 60


# ******************* MatPlot *******************
plot.plot([60*i for i in range(11)], quickData, label='Quick Sort')
plot.plot([60*i for i in range(11)], heapData, label='Heap Sort')
plot.plot([60*i for i in range(11)], mergeData, label='Merge Sort')
plot.xlabel('Number of Swaps')
plot.ylabel('Time Taken (s)')
plot.title("Good Sorts vs Swaps Test")
plot.legend()
plot.show()