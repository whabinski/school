# Varying nodes in the graph
# Use a realistic graph (less nodes than edges) that has a fixed proportion of nodes and edges and a fixed number of relaxtions

# Compare size of approximations to real shortest path

from final_project_part1 import *
from part1 import *
import matplotlib.pyplot as plot

shortestData = []
dijkstraData = []
bellmanfordData = []

tests = 100

relaxtions = 3 # fixed
nodes = 5 # varied
increment = 1
maxNodes = 30

while nodes <= maxNodes:
    
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
    
    nodes += increment


# ******************* MatPlot *******************

plot.plot([i for i in range(5, maxNodes +1)], shortestData, label="Shortest Path")
plot.plot([i for i in range(5, maxNodes +1)], dijkstraData, label="Dijkstra Approx.")
plot.plot([i for i in range(5, maxNodes +1)], bellmanfordData, label="Bellman Ford Approx.")
plot.xlabel('Number of Nodes')
plot.ylabel('Total Distance of The Shortest Path Found')
plot.title("Varying Nodes with Constant Relaxtions(3) vs Shortest Path Distance")
plot.legend()
plot.show()