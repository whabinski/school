from final_project_part1 import *
from min_heap import *
import random

def dijkstra_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    num_relaxes = {} # ADDED 
    Q = MinHeap([])
    nodes = list(G.adj.keys())

    for node in nodes:
        Q.insert(Element(node, float("inf")))
        dist[node] = float("inf")
        num_relaxes[node] = k #ADDED
    Q.decrease_key(source, 0)

    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour] and num_relaxes[neighbour] > 0 :
                num_relaxes[neighbour] -= 1 #ADDED
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist

def bellman_ford_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    num_relaxes = {} # ADDED 
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
        num_relaxes[node] = k #ADDED
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour) and num_relaxes[neighbour] > 0:
                    num_relaxes[neighbour] -= 1 #ADDED
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist

####################################################################################
# The following functions are used for Experiment 1.3

def create_minimal_edge_complete_graph(n,upper): 
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)

    nodes = list(G.adj.keys())
    
    while len(nodes) != 0:
        current_node = nodes[0]
        
        if len(nodes) == 1:
            if (len(G.adj[current_node])) > 0:
                return G
            neighbour = random.randint(1,n-1)
            while neighbour == current_node:
                neighbour = random.randint(1,n-1)
            G.add_edge(current_node,neighbour,random.randint(1,upper))
            G.add_edge(neighbour,current_node,random.randint(1,upper)) 
        else:
            neighbour = random.choice(nodes)
            while neighbour == current_node:
                neighbour = random.choice(nodes)
            G.add_edge(current_node,neighbour,random.randint(1,upper))
            G.add_edge(neighbour,current_node,random.randint(1,upper))
        nodes.remove(current_node)
    
    return G

# Count total edges in a graph
def total_edges(G):
    nodes = list(G.adj.keys())
    total_edges = 0
    
    for x in range(len(nodes)):
        total_edges += len(G.adj[x])
    return total_edges

def add_new_edge(G,upper):
    nodes = list(G.adj.keys())
    numNodes = len(nodes)-1
    node1 = random.randint(0,numNodes)
    node2 = random.randint(0,numNodes)
    
    while node2 in G.adj[node1]:
        node1 = random.randint(0,numNodes)
        node2 = random.randint(0,numNodes)
    
    G.add_edge(node1,node2,random.randint(1,upper))
    
    return
