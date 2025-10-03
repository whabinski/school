from XC3Tree import *
import matplotlib.pyplot as plot


def nodes(i):
    t = XC3Tree(i)
    return t.count_nodes()
    
max_degree = 25
l = []

for d in range(max_degree+1):
    l.append(nodes(d))

#print(l)

# ******************* MatPlot *******************
plot.plot([i for i in range(max_degree+1)], l, label='XC3-Tree')
plot.xlabel('Degree of Tree')
plot.ylabel('Nodes in the Tree')
plot.title("Nodes in the Tree vs Degree of Tree")
plot.legend()
plot.show()

