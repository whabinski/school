from final_project_part1 import *
import min_heap

def a_star(G, s, d, h):
    pred = {} #Predecessor dictionary
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
        pred[node] = None
    Q.decrease_key(s, h[s]) #changed
    pred[s] = s

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        
        #added - When the destination node is found
        if current_node == d:
            if pred[current_node] == None:
                return (pred, [])
            path = [] #Path of shortest path
            while current_node != s:
                path.insert(0,current_node)
                current_node = pred[current_node]
            path.insert(0,s)
            return (pred, path)
        
        #Meat of the Algorithm
        dist[current_node] = current_element.key + h[current_node]
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour) + h[neighbour]) #changed
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour) + h[neighbour] #changed
                pred[neighbour] = current_node

    return (pred, path)

'''
G = DirectedWeightedGraph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_edge(1,0,2)
G.add_edge(1,3,11)
G.add_edge(0,3,8)
G.add_edge(0,2,3)
G.add_edge(2,1,100)
G.add_edge(2,3,4)
G.add_edge(3,4,100)
h = {0 : 1111, 1 : 0, 2 : 0, 3 : 0, 4 : 0}
print(a_star(G, 1, 5, h))
'''