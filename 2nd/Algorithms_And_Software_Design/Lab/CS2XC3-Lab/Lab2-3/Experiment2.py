from bad_sorts import *
import matplotlib.pyplot as plot

# ******************* Selection sort code *******************

def selectionsort2(L):
    start = 0
    end = len(L) - 1 
    while start < end:
        
        min_index = find_min_index2(L, start, end)
        swap(L, start, min_index)
        max_index = find_max_index(L, start, end)
        swap(L, end, max_index)
        start += 1
        end -= 1
        

def find_min_index2(L, start, end):
    min_index = start
    for i in range(start, end+1):
        if L[i] < L[min_index]:
            min_index = i
    return min_index

def find_max_index(L, start, end):
    max_index = start
    for i in range(start, end+1):
        if L[i] > L[max_index]:
            max_index = i
    return max_index

# ******************* Experiment2 SELECTION SORT Testing data *******************


selection1Data = []
selection2Data = []

increments = 0
totalElements = 1000
tests = 100

total1 = 0
total2 = 0

while increments <= totalElements:

    for _ in range(tests):
        L = create_random_list(increments, increments)
        L2 = L.copy()

        start = timeit.default_timer()
        selection_sort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        selectionsort2(L2)
        end = timeit.default_timer()
        total2 += end - start
    
    selection1Data.append(total1/tests)
    selection2Data.append(total2/tests)
    increments += 100

# ******************* MatPlot for Selection *******************
# **** Comment out to run the bubble sort graph ****

plot.plot([100*i for i in range(11)], selection1Data, label='Selection Sort 1')
plot.plot([100*i for i in range(11)], selection2Data, label='Selection Sort 2')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Original vs Optimized Selection Sort Test")
plot.legend()
plot.show()


# ******************* Bubble sort code *******************

def bubblesort2(L):
    for i in range (len(L)):
        temp = L[0]
        for j in range(len(L)-1):
            if temp > L[j+1]:
                L[j] = L[j+1]
            else :
                L[j] = temp
                temp = L[j+1]
        L[-1] = temp

# ******************* Experiment2 BUBBLE SORT Testing data *******************

bubble1Data = []
bubble2Data = []

increments = 0
totalElements = 1000
tests = 100

total1 = 0
total2 = 0

while increments <= totalElements:

    for _ in range(tests):
        L = create_random_list(increments, increments)
        L2 = L.copy()

        start = timeit.default_timer()
        bubble_sort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        bubblesort2(L2)
        end = timeit.default_timer()
        total2 += end - start
    
    bubble1Data.append(total1/tests)
    bubble2Data.append(total2/tests)
    increments += 100

# ******************* MatPlot for Bubble *******************
# **** Comment out to run the selection sort graph ****

plot.plot([100*i for i in range(11)], bubble1Data, label='Bubble Sort 1')
plot.plot([100*i for i in range(11)], bubble2Data, label='Bubble Sort 2')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Original vs Optimized Bubble Sort Test")
plot.legend()
plot.show()