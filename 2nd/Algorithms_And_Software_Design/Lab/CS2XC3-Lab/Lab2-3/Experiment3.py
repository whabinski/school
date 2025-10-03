from bad_sorts import *
import matplotlib.pyplot as plot

selectionData = []
insertionData = []
bubbleData = []

swaps = 500
increments = 0
tests = 10

total1 = 0
total2 = 0
total3 = 0

while increments <= swaps:

    for _ in range(tests):
        L = create_near_sorted_list(5000,5000,increments)
        L2 = L.copy()
        L3 = L.copy()

        start = timeit.default_timer()
        insertion_sort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        bubble_sort(L2)
        end = timeit.default_timer()
        total2 += end - start

        start = timeit.default_timer()
        selection_sort(L3)
        end = timeit.default_timer()
        total3 += end - start
    
    insertionData.append(total1/tests)
    bubbleData.append(total2/tests)
    selectionData.append(total3/tests)
    increments += 50

# ******************* MatPlot *******************
plot.plot([50*i for i in range(11)], selectionData, label='Selection Sort')
plot.plot([50*i for i in range(11)], insertionData, label='Insertion Sort')
plot.plot([50*i for i in range(11)], bubbleData, label='Bubble Sort')
plot.xlabel('Number of Swaps')
plot.ylabel('Time Taken (s)')
plot.title("Swaps vs Time Bad Algorithms")
plot.legend()
plot.show()