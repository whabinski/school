import copy
import random
from graphs import *

# APPROX1 - largest degree mvc
def approx1(G):  
    
    copyG = copy.deepcopy(G)
    c = set()
    
    while(not is_vertex_cover(copyG,c)):
        v = 0
        for node in copyG.adj:
            if len(copyG.adj[node]) >= len(copyG.adj[v]):
                v = node
        c.add(v)
        remove_all_edges(copyG,v)
    
    return c

# APPROX2 - random vertex mvc
def approx2(G): 
    
    copyG = copy.deepcopy(G)
    c = set()
    nodeList = []
    
    for node in copyG.adj.keys():
        nodeList.append(node)
    
    while(not is_vertex_cover(copyG,c)):
        index = random.randint(0, len(nodeList) - 1)
        node = nodeList[index]
        nodeList.remove(node)
        c.add(node)
        
    return c

# APPROX3 - random edges mvc
def approx3(G):   
     
    copyG = copy.deepcopy(G)
    c = set()
    
    while(not is_vertex_cover(copyG,c)):
        edge = random_edge(copyG)
        c.add(edge[0])
        c.add(edge[1])
        remove_all_edges(copyG, edge[0])
        remove_all_edges(copyG, edge[1])
    
    return c

# Remove all edges of a node ***************************************
def remove_all_edges(G, node):
    adj_nodes = G.adj[node]
    G.adj[node] = []
    for n in adj_nodes:
        try:
            G.adj[n].remove(node)
        except:
            pass
        
# Remove all edges of a node ***************************************
def random_edge(G):
    
    numNodes = G.number_of_nodes()
    has_edge = False
    for i in list(G.adj.keys()):
        if (len(G.adj[i])) > 0:
            has_edge = True
    
    if (has_edge == False):
        return None

    while True:
        u = random.choice(list(G.adj.keys()))
        
        if (len(G.adj[u]) > 0):
            v = random.choice(list(G.adj[u]))
            break
        
    return (u,v) 

# TESTING *********************************************

'''
g = Graph(7)
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(2,4)
g.add_edge(3,4)
g.add_edge(3,5)
g.add_edge(4,5)
g.add_edge(4,6)
g.add_edge(0,6)

v = Graph(2)
v.add_edge(0,1)



print("\nADJ\n")
print(g.adj)

print("\nMVC\n")
print(MVC(v))

print("\nAPPROX1\n")
print(approx1(v))

print("\nAPPROX2\n")
print(approx2(v))

print("\nAPPROX3\n")
print(approx3(v))
'''