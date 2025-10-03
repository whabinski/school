# All pairs shortest path lines experiment

from final_project_part1 import *
from part2 import *
from part3 import *
import matplotlib.pyplot as plot
import timeit



G = createLondonGraph()

stations = pd.read_csv("london_stations.csv")
connections = pd.read_csv("london_connections.csv")

# Function to get the line that corresponds to the connection of two stations
def getLine(s1,s2):
    lines = []
    
    connection1 = connections[(connections['station1'] == s1) & (connections['station2'] == s2)]
    connection2 = connections[(connections['station1'] == s2) & (connections['station2'] == s1)]
    
    for x in range(len(connection1)):
        lines.append(connection1.iloc[x][2])
        
    for x in range(len(connection2)):
        lines.append(connection2.iloc[x][2])
    
    return lines

# Function to return a list of all lines used in a given path
def getAllLines(path):
    lines = []
    
    for x in range(len(path) - 1):
    
        for x in getLine(path[x], path[x+1]):
            lines.append(x)

    return lines

# Function to compute how many distinct lines were in the given set of lines
def getNumLines(lines):
    return len(set(lines))


####### Start of testing data #######
stationsList = list(G.adj.keys())

sampleData = 500

oneLine = []
twoLines = []
multipleLines = []

for _ in range(sampleData):
    station1 = random.choice(stationsList)
    station2 = random.choice(stationsList)
    
    heuristic = heuristicFunction(G, station2)
    start = timeit.default_timer()
    aStar = a_star(G,station1,station2,heuristic)
    end = timeit.default_timer()
    aStarTime = end - start
    print(aStarTime)
    
    path = aStar[1]
    lines = getAllLines(path)
    numLines = getNumLines(lines)
    
    if numLines == 1:
        oneLine.append(aStarTime)
    elif numLines == 2:
        twoLines.append(aStarTime)
    else:
        multipleLines.append(aStarTime)
    
oneLineExtraSpace = sampleData - len(oneLine)
twoLinesExtraSpace = sampleData - len(twoLines)
multipleLinesExtraSpace = sampleData - len(multipleLines)

oneLineExtra = [None for x in range(oneLineExtraSpace)]
oneLine = oneLine + oneLineExtra

twoLinesExtra = [None for x in range(twoLinesExtraSpace)]
twoLines = twoLines + twoLinesExtra

multipleLinesExtra = [None for x in range(multipleLinesExtraSpace)]
multipleLines = multipleLines + multipleLinesExtra

print(len(oneLine))
print(len(twoLines))
print(len(multipleLines))


# ******************* MatPlot *******************

plot.plot([i for i in range(sampleData)], oneLine, label="One Line")
plot.plot([i for i in range(sampleData)], twoLines, label="Two Lines")
plot.plot([i for i in range(sampleData)], multipleLines, label="Multiple Lines")
plot.xlabel('')
plot.ylabel('Runtime')
plot.title("Runtime of A* over 1 vs 2 vs Multiple Lines")
plot.legend()
plot.show()
