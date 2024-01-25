# Varying relaxations value
# Use a realistic graph (less nodes than edges) that has a fixed number of nodes and edges

# Compare size of approximations to real shortest path

from final_project_part1 import *
from part1 import *
import matplotlib.pyplot as plot

shortestData = []
dijkstraData = []
bellmanfordData = []

tests = 100

nodes = 30 # fixed
relaxtions = 1 # varied
increment = 1
maxRelaxtions = 10

while relaxtions <= maxRelaxtions:
    
    totalShortest = 0
    totalDijkstra = 0
    totalBellman = 0

    for _ in range(tests):
        G = create_random_complete_graph(nodes,nodes^2)
        
        shortest_dist = total_dist(dijkstra(G, 0))
        dijkstra_dist = total_dist(dijkstra_approx(G, 0, relaxtions))
        bellman_dist = total_dist(bellman_ford_approx(G, 0, relaxtions))

        totalShortest += shortest_dist
        totalDijkstra += dijkstra_dist
        totalBellman += bellman_dist
        
    shortestData.append(totalShortest/tests)    
    dijkstraData.append(totalDijkstra/tests)
    bellmanfordData.append(totalBellman/tests)
    
    relaxtions += increment


# ******************* MatPlot *******************

plot.plot([i for i in range(maxRelaxtions)], shortestData, label="Shortest Path")
plot.plot([i for i in range(maxRelaxtions)], dijkstraData, label="Dijkstra Approx.")
plot.plot([i for i in range(maxRelaxtions)], bellmanfordData, label="Bellman Ford Approx.")
plot.xlabel('Number of Permitted Relaxations')
plot.ylabel('Total Distance of The Shortest Path Found')
plot.title("Varying Relaxations vs Shortest Path Distance")
plot.legend()
plot.show()
