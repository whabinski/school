from graphs import *
import matplotlib.pyplot as plot

NODES = 200
MAX_EDGES = 200
inc = 2
edges = 0
data = []

tests = 200 #m

while edges <= MAX_EDGES:
    count_cycles = 0 
    for _ in range(tests):
        G = create_random_graph(NODES, edges)

        if has_cycle(G):
            count_cycles += 1

    edges += inc
    data.append(count_cycles/tests)


# ******************* MatPlot *******************
plot.plot([inc*i for i in range(int(MAX_EDGES/inc) + 1)], data)
plot.xlabel('Number of Edges')
plot.ylabel('Cycle Probability')
plot.title("Number of Edges vs Cycle Probability")
plot.show()