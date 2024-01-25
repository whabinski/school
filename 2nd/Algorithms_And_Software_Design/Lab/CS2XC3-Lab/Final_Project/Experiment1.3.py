# Varying edges in the graph
# Use a realistic graph (less nodes than edges) that has a fixed number of nodes and relaxtions while having varying edges

# Compare size of approximations to real shortest path

from final_project_part1 import *
from part1 import *
import matplotlib.pyplot as plot

shortestData = []
dijkstraData = []
bellmanfordData = []

tests = 100

relaxtions = 3 # fixed
nodes = 50 # fixed
edges = 78 # minimal amount of edges for a complete directed graph
increment = 100
maxEdges = 1500 

while edges <= maxEdges:
    
    totalShortest = 0
    totalDijkstra = 0
    totalBellman = 0

    for _ in range(tests):
        G = create_minimal_edge_complete_graph(nodes,nodes^2)
        for _ in range(edges):
            add_new_edge(G,nodes^2)
        
        shortest_dist = total_dist(dijkstra(G, 0))
        dijkstra_dist = total_dist(dijkstra_approx(G, 0, relaxtions))
        bellman_dist = total_dist(bellman_ford_approx(G, 0, relaxtions))

        totalShortest += shortest_dist
        totalDijkstra += dijkstra_dist
        totalBellman += bellman_dist
        
    shortestData.append(totalShortest/tests)    
    dijkstraData.append(totalDijkstra/tests)
    bellmanfordData.append(totalBellman/tests)
    
    edges += increment


# ******************* MatPlot *******************

plot.plot([i for i in range(78, maxEdges, increment)], shortestData, label="Shortest Path")
plot.plot([i for i in range(78, maxEdges, increment)], dijkstraData, label="Dijkstra Approx.")
plot.plot([i for i in range(78, maxEdges, increment)], bellmanfordData, label="Bellman Ford Approx.")
plot.xlabel('Number of Edges')
plot.ylabel('Total Distance of The Shortest Path Found')
plot.title("Varying Edges with Constant Relaxtions(3) vs Shortest Path Distance")
plot.legend()
plot.show()