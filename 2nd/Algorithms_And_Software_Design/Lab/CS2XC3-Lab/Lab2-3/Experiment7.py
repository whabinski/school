from bad_sorts import *
from good_sorts import *
import matplotlib.pyplot as plot

<<<<<<< HEAD
def mergesort(L):
    if len(L) <= 1:
        return
    mid = len(L) // 2
    left, right = L[:mid], L[mid:]

    mergesort(left)
    mergesort(right)
    temp = merge(left, right)

    for i in range(len(temp)):
        L[i] = temp[i]


def merge(left, right):
    L = []
    i = j = 0

    while i < len(left) or j < len(right):
        if i >= len(left):
            L.append(right[j])
            j += 1
        elif j >= len(right):
            L.append(left[i])
            i += 1
        else:
            if left[i] <= right[j]:
                L.append(left[i])
                i += 1
            else:
                L.append(right[j])
                j += 1
    return L

def  bottom_up_mergesort(L):
    windowSize = 1

    while windowSize < len(L) :
        left = 0
        right = windowSize
        while left < len(L) :
            L = merge2(L, windowSize, left, right)
            left += windowSize*2
            right = left + windowSize

        windowSize = windowSize*2
    return L
    

#assumes sections are sorted
def merge2(L, windowSize, left, right):
    #left = start of left list
    #right = start of right list

    i = 0
    j = 0

    while right+j < len(L):
        if (L[left+i] > L[right+j]):
            #shift the values
            L = L[:left+i] + [L[right+j]] + L[left+i:right+j] + L[right+j+1:]
            i += 1
        else :
            j += 1    
    
        print(L)
    return L

L = [7, 4, 1, 7]

L = bottom_up_mergesort(L)
print(L)



=======

def bottom_up_mergesort(L):

    windowSize = 1	
    listSize = len(L)										
    
    while (windowSize < listSize):
        left = 0;
        while (left < listSize):
            right = min(left+(windowSize*2-1), listSize-1)		
            middle = min(left+windowSize-1,listSize-1)			
            bottom_up_merge(L, left, middle, right)
            left += windowSize*2
        windowSize *= 2
    return L
	
def bottom_up_merge(L, left, middle, right):
    
	num1 = middle - left + 1
	num2 = right - middle
	
	leftList = [0] * num1
	rightList = [0] * num2
  
	for i in range(0, num1):
		leftList[i] = L[left + i]
	for i in range(0, num2):
		rightList[i] = L[middle + i + 1]

	i = 0
	j = 0
	k = left
   
	while i < num1 and j < num2:
		if leftList[i] <= rightList[j]:
			L[k] = leftList[i]
			i += 1
		else:
			L[k] = rightList[j]
			j += 1
		k += 1

	while i < num1:
		L[k] = leftList[i]
		i += 1
		k += 1

	while j < num2:
		L[k] = rightList[j]
		j += 1
		k += 1

# ******************* Experiment2 BUBBLE SORT Testing data *******************

mergeData = []
bottomUpMergeData = []

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
        mergesort(L)
        end = timeit.default_timer()
        total1 += end - start

        start = timeit.default_timer()
        bottom_up_mergesort(L2)
        end = timeit.default_timer()
        total2 += end - start
    
    mergeData.append(total1/tests)
    bottomUpMergeData.append(total2/tests)
    increments += 100

# ******************* MatPlot for Bubble *******************

plot.plot([100*i for i in range(11)], mergeData, label='Tradition Meger Sort')
plot.plot([100*i for i in range(11)], bottomUpMergeData, label='Bottom Up Merge Sort')
plot.xlabel('List Size')
plot.ylabel('Time Taken (s)')
plot.title("Original vs BottomUp Merge Sort Test")
plot.legend()
plot.show()
>>>>>>> origin/main
