from collections import deque
import random

#Undirected graph using an adjacency list
class Graph:

    def __init__(self, n):
        self.adj = {}
        for i in range(n):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        return node2 in self.adj[node1]

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            if (node1 != node2):
                self.adj[node2].append(node1) #this change was made (self loops allowed??)

    def number_of_nodes(self):
        return len(self.adj)


#Breadth First Search
def BFS(G, node1, node2):
    Q = deque([node1])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if node == node2:
                return True
            if not marked[node]:
                Q.append(node)
                marked[node] = True
    return False

#Breadth First Search ***************************************
def BFS2(G, node1, node2):
    Q = deque([(node1, [node1])])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node, path = Q.popleft()
        for node in G.adj[current_node]:
            if node == node2:
                path.append(node)
                return path
            if not marked[node]:
                Q.append((node, path + [node]))
                marked[node] = True
    return []

def BFS3(G, node1):
    Q = deque([node1])
    marked = {node1 : True}
    pred = {}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if not marked[node]:
                Q.append(node)
                marked[node] = True
                pred[node] = current_node
    return pred


#Depth First Search ***************************************
def DFS(G, node1, node2):
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                if node == node2:
                    return True
                S.append(node)
    return False

def DFS2(G, node1, node2):
    S = [(node1, [node1])]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node, path = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                if node == node2:
                    path.append(node)
                    return path
                S.append((node, path + [node]))
    return []

def DFS3(G, node1):
    S = [(node1, None)]
    marked = {}
    pred = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node, previous = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            if not current_node == node1:
                pred[current_node] = previous
            for node in G.adj[current_node]:
                S.append((node, current_node))
    return pred

#Has Cycle ***************************************
def has_cycle(G):
    marked = {}
    for node in G.adj:
        marked[node] = False
    for node in G.adj:
        if not marked[node]:
            if has_cycle_helper(G, node, marked, None):
                return True
    return False

def has_cycle_helper(G, node, marked, previous):
    marked[node] = True
    for adj in G.adj[node]:
        if not marked[adj]:
            if has_cycle_helper(G, adj, marked, node):
                return True
        elif adj != previous:
            return True
    return False

#Is Connected ***************************************
def is_connected(G):
    L = DFS3(G, 0)
    if len(L) == G.number_of_nodes() - 1:
            return True
    return False

#Create Random Graph ***************************************
# we allow self loops in our implementation
# we do not allow duplicate edges 
def create_random_graph(i, j):
    G = Graph(i)
    edges = []
    count = 0
    while count < j :
        u = random.randint(0, i-1)
        v = random.randint(0, i-1)

        if not((u,v) in edges) and not((v,u) in edges):
            edges.append((u,v))
            count += 1
            G.add_edge(u, v)
    return G

#Minimum Vertex Covers ***************************************
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])

def is_vertex_cover(G, C):
    for start in G.adj:
        for end in G.adj[start]:
            if not(start in C or end in C):
                return False
    return True

def MVC(G):
    nodes = [i for i in range(G.number_of_nodes())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

#Maximum Independant Set ***************************************
def is_independant_set(G, C):
    
    for start in C:
        for end in C:
            if (end in G.adj[start]):
                return False
    
    return True

def MIS(G):
    nodes = [i for i in range(G.number_of_nodes())]
    subsets = power_set(nodes)
    max_IS = []
    
    for subset in subsets:
        if is_independant_set(G, subset):
            if len(subset) > len(max_IS):
                max_IS = subset
    return max_IS
        
#********* TESTING **********

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
g.add_edge(6,5)
'''
