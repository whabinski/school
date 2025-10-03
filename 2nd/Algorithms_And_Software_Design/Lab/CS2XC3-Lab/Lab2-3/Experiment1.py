from bad_sorts import *
import matplotlib.pyplot as plot

selectionData = []
insertionData = []
bubbleData = []

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
    increments += 100

# ******************* MatPlot *******************
plot.plot([100*i for i in range(11)], selectionData, label='Selection Sort')
plot.plot([100*i for i in range(11)], insertionData, label='Insertion Sort')
plot.plot([100*i for i in range(11)], bubbleData, label='Bubble Sort')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Sorting Algorithm Test")
plot.legend()
plot.show()