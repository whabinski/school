# All pairs shortest paths runtimes
# 10 000 Random pairs

from final_project_part1 import *
from part2 import *
from part3 import *
import matplotlib.pyplot as plot
import timeit
import random

G = createLondonGraph()
stationsList = list(G.adj.keys())

sampleData = 10000
dijkstraData = []
aStarData = []

for _ in range(sampleData):
    station1 = random.choice(stationsList)
    station2 = random.choice(stationsList)

    start = timeit.default_timer()
    dijkstra(G,station1)
    end = timeit.default_timer()
    dijkstraTime = end - start
    print("Dijkstra" + str(dijkstraTime))
    
    heuristic = heuristicFunction(G, station2)
    start = timeit.default_timer()
    a_star(G,station1,station2,heuristic)
    end = timeit.default_timer()
    aStarTime = end - start
    print("aStar" + str(aStarTime))
    
    dijkstraData.append(dijkstraTime)
    aStarData.append(aStarTime)

# ******************* MatPlot *******************

plot.plot([i for i in range(sampleData)], aStarData, label="A*")
plot.plot([i for i in range(sampleData)], dijkstraData, label="Dijkstra")
plot.xlabel('')
plot.ylabel('Runtime')
plot.title("Runtime of Dijkstra vs A*")
plot.legend()
plot.show()
