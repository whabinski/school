from graphs import *
from Approximations import *
import matplotlib.pyplot as plot

# APPROXIMATION EXPERIMENT 2 ************************************
# Measures the accuracy of Approximations relative to # of edges

# Constant Nodes with increasing edges (1 to 36)
# 1000 Test graphs per edge increase

NODES = 8
MAX_EDGES = 36
edges = 1
inc = 5

tests = 1000

approx1Data = []
approx2Data = []
approx3Data = []

while edges <= MAX_EDGES:
    
    totalApprox1 = 0
    totalApprox2 = 0
    totalApprox3 = 0
    
    for _ in range(tests):
        G = create_random_graph(NODES, edges)
        
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
    
    edges += inc
        
    
# ******************* MatPlot for Experiment 2 *******************
plot.plot([inc*i for i in range(int(MAX_EDGES/inc) + 1)], approx1Data, label = 'Approx1')
plot.plot([inc*i for i in range(int(MAX_EDGES/inc) + 1)], approx2Data, label = 'Approx2')
plot.plot([inc*i for i in range(int(MAX_EDGES/inc) + 1)], approx3Data, label = 'Approx3')
plot.legend()
plot.xlabel('Number of Edges')
plot.ylabel('Accurate Approximations (%)')
plot.title('Accuracy of Approximations vs # of Edges')
plot.show()