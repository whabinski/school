from XC3Tree import *
import matplotlib.pyplot as plot

def h(i):
    t = XC3Tree(i)
    childHeights = [0]
    
    if (len(t.children) != 0):
        for c in t.children:
            d = c.degree
            childHeights.append(h(d))
    

    height = 1 + max(childHeights)
    return height
    
max_degree = 25
heights = []

for d in range(max_degree+1):
    heights.append(h(d) - 1)

print(heights)
# ******************* MatPlot *******************
plot.plot([i for i in range(max_degree+1)], heights, label='XC3-Tree')
plot.xlabel('Degree of Tree')
plot.ylabel('Height of Tree')
plot.title("Height of Tree vs Degree of Tree")
plot.legend()
plot.show()

