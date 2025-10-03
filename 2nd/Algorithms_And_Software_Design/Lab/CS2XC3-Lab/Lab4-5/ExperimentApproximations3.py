from graphs import *
from Approximations import *
import matplotlib.pyplot as plot

# APPROXIMATION EXPERIMENT 3 ************************************
# Measures the accuracy of Approximations relative to # of nodes

# Constant proportion of edges with increasing nodes (0 to 15)
# 1000 Test graphs per edge increase

def number_of_edges(nodes):
    edges = (nodes * (nodes-1))/2 + nodes
    return edges

nodes = 0
MAX_NODES = 15
inc = 1

tests = 1000

approx1Data = []
approx2Data = []
approx3Data = []

while nodes <= MAX_NODES:
    
    totalApprox1 = 0
    totalApprox2 = 0
    totalApprox3 = 0
    
    for _ in range(tests):
        edges = int((number_of_edges(nodes))/2)
        G = create_random_graph(nodes, edges)
        
        mvcSize = len(MVC(G))
        approx1Size = len(approx1(G));
        approx2Size = len(approx2(G));
        approx3Size = len(approx3(G));
    
        if (approx1Size == mvcSize):
            totalApprox1 += 1
        if (approx2Size == mvcSize):
            totalApprox2 += 1
        if (approx3Size == mvcSize):
            totalApprox3 += 1
    
    
    approx1Data.append(totalApprox1/tests)
    approx2Data.append(totalApprox2/tests)
    approx3Data.append(totalApprox3/tests)
    
    nodes += inc
        
    
# ******************* MatPlot for Experiment 3 *******************
plot.plot([inc*i for i in range(int(MAX_NODES/inc) + 1)], approx1Data, label = 'Approx1')
plot.plot([inc*i for i in range(int(MAX_NODES/inc) + 1)], approx2Data, label = 'Approx2')
plot.plot([inc*i for i in range(int(MAX_NODES/inc) + 1)], approx3Data, label = 'Approx3')
plot.legend()
plot.xlabel('Number of Nodes')
plot.ylabel('Accurate Approximations (%)')
plot.title('Accuracy of Approximations vs # of Nodes')
plot.show()
