from graphs import *
from Approximations import *
import matplotlib.pyplot as plot

# APPROXIMATION EXPERIMENT 1 ************************************
# Measures the ratio of Approximation sizes over MVC size relative to # of edges

# Constant Nodes with increasing edges (1 to 36)
# 1000 Test graphs per edge increase

NODES = 8
MAX_EDGES = 36
edges = 1
inc = 5

tests = 1000

mvcData = []
approx1Data = []
approx2Data = []
approx3Data = []

while edges <= MAX_EDGES:
    
    totalMVC = 0;
    totalApprox1 = 0;
    totalApprox2 = 0;
    totalApprox3 = 0;
    
    for _ in range(tests):
        G = create_random_graph(NODES, edges)
        
        totalMVC += len(MVC(G));
        totalApprox1 += len(approx1(G));
        totalApprox2 += len(approx2(G));
        totalApprox3 += len(approx3(G));
    
    approx1Data.append(totalApprox1/totalMVC)
    approx2Data.append(totalApprox2/totalMVC)
    approx3Data.append(totalApprox3/totalMVC)
    
    edges += inc
        
    
# ******************* MatPlot for Experiment 1 *******************

plot.plot([inc * i for i in range(int(MAX_EDGES/inc) + 1)], approx1Data, label = 'Approx1')
plot.plot([inc * i for i in range(int(MAX_EDGES/inc) + 1)], approx2Data, label = 'Approx2')
plot.plot([inc * i for i in range(int(MAX_EDGES/inc) + 1)], approx3Data, label = 'Approx3')
plot.legend()
plot.xlabel('Number of Edges')
plot.ylabel('Relative Size (To MVC)')
plot.title("Approximations Size compared to MVC")
plot.show()
