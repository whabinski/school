import min_heap

from final_project_part1 import *
from part2 import *
from part3 import *
import matplotlib.pyplot as plot
import random

# An altered version of A* which slightly modifies the predecessor dictionary for easier computation
def a_star_pred(G, s, d, h):
    pred = {} #Predecessor dictionary
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
        #pred[node] = None
    Q.decrease_key(s, h[s]) #changed
    pred[s] = s

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        
        #added - When the destination node is found
        if current_node == d:
            try:
                if pred[current_node] == None:
                    return (pred, [])
            except:
                return None
            
            path = [] #Path of shortest path
            while current_node != s:
                path.insert(0,current_node)
                current_node = pred[current_node]
            path.insert(0,s)
            return pred
        
        #Meat of the Algorithm
        dist[current_node] = current_element.key + h[current_node]
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour) + h[neighbour]) #changed
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour) + h[neighbour] #changed
                pred[neighbour] = current_node

    return pred

G = createLondonGraph()
stationsList = list(G.adj.keys())
numStations = len(stationsList)

sampleData = 1000
dijkstraData = []
aStarData = []

for _ in range(sampleData):
    station1 = random.choice(stationsList)
    station2 = random.choice(stationsList)

    heuristic = heuristicFunction(G, station1)

    predA = a_star_pred(G, station2, station1, heuristic)
    if (predA != None): 
        aStarData.append(len(predA))
        dijkstraData.append(numStations)

# ******************* MatPlot *******************

plot.plot([i for i in range(sampleData)], aStarData, label="A*")
plot.plot([i for i in range(sampleData)], dijkstraData, label="Dijkstra")
plot.xlabel('')
plot.ylabel('Number of Nodes in Predecessor Dictionary')
plot.title("Number of Nodes Visited")
plot.legend()
plot.show()