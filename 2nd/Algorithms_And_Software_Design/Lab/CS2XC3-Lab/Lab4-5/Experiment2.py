from graphs import *
import matplotlib.pyplot as plot

NODES = 100
MAX_EDGES = NODES*7
start_edges = 50
inc = 2
data = []

tests = 200 #m

edges = start_edges

while edges <= MAX_EDGES:
    num_connected = 0 
    for _ in range(tests):
        G = create_random_graph(NODES, edges)

        if is_connected(G):
            num_connected += 1

    edges += inc
    data.append(num_connected/tests)

# ******************* MatPlot *******************
plot.plot([start_edges + inc*i for i in range(int((MAX_EDGES - start_edges)/inc) + 1)], data)
plot.xlabel('Number of Edges')
plot.ylabel('Connected Probability')
plot.title("Number of Edges vs Connected Probability")
plot.show()