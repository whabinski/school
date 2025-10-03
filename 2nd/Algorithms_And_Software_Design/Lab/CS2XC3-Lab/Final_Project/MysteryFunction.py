from final_project_part1 import *
from part1 import *
import matplotlib.pyplot as plot
import timeit
import math

mysteryData = []

tests = 20
n = 5
maxN = 100
total1 = 0

while n <= maxN:

    for _ in range(tests):
        G = create_random_complete_graph(n,n^2)
        
        start = timeit.default_timer()
        mystery(G)
        end = timeit.default_timer()
        total1 += end - start
        
    mysteryData.append(total1/tests)
    
    n += 1


# ******************* MatPlot *******************

x = list(map(math.log, [i for i in range(5, maxN+1)]))
y = list(map(math.log, mysteryData))

print((y[-1] - y[0]) / (x[-1] - x[0]))

plot.plot(x, y)
plot.xlabel('Number of Verticies')
plot.ylabel('Time')
plot.title("Runtime of the Mystery Function")
plot.legend()
plot.show()