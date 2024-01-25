import pandas as pd
import math
from final_project_part1 import *

#Reead the london stations file
stations = pd.read_csv("london_stations.csv")
# 302 rows
# index column is the id of each row / station
#generated_column    id    latitude    longitude   name    display_name    zone    total_lines     rail

#Read the london connections file
connections = pd.read_csv("london_connections.csv")
# 406 rows
# index column is randomly generated for each row
#generated_index    station1    station2    line    time


#stations.iloc[x] will return the values of the row in stations at index x
#id ---- stations.iloc[x][0] will return the value
#lat --- stations.iloc[x][1] will return the value
#lon --- stations.iloc[x][2] will return the value

#connections.iloc[x] will return the values of the row in stations at index x
#station1 --- connections.iloc[x][0] will return the value
#station2 --- connections.iloc[x][1] will return the value

# Used to get the distance between two points in the euclidean plane
# A helper function to distanceBetween(s1,s2)
def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx**2 + dy**2)
    return distance

# Get the distance between two stations
# Used to get the weights for each edge
def distanceBetween(s1,s2):
    distance = 0
    
    # To Find a row with Id value == 1
    #row = stations[stations['id'] == 1]
    # Given that row, find the value of the rows 
    # ***[0] must be there, then [x] corresponds to the column
    #value = row.iloc[0][1]
    
    #Finds row of s1
    station1Row = stations[stations['id'] == s1]
    #s1 latitude
    s1Lat = station1Row.iloc[0][1]
    #s2 latitude
    s1Lon = station1Row.iloc[0][2]
    s1Pos = (s1Lat, s1Lon) 
    
    station2Row = stations[stations['id'] == s2]
    s2Lat = station2Row.iloc[0][1]
    s2Lon = station2Row.iloc[0][2]
    s2Pos = (s2Lat, s2Lon) 
    
    distance = euclidean_distance(s1Pos, s2Pos)
    
    return float(distance)

# Used to create the heuristic function for any start node and destination node
# We recompute the destination and station rows in this function instead of using distanceBetween function
#   in order to reduce the amount of times destinationPos is computed as it will be the same for every iteration
def heuristicFunction(G, destination):
    stationsList = list(G.adj.keys())
    h = {}
    
    destinationRow = stations[stations['id'] == destination]
    destinationLat = destinationRow.iloc[0][1]
    destinationLon = destinationRow.iloc[0][2]
    destinationPos = (destinationLat, destinationLon) 
    
    for station in stationsList:
        
        stationRow = stations[stations['id'] == station]
        stationLat = stationRow.iloc[0][1]
        stationLon = stationRow.iloc[0][2]
        stationPos = (stationLat, stationLon)
        
        distance = euclidean_distance(stationPos, destinationPos)
        h[station] = float(distance)
        
    return h 

def createLondonGraph():
    #We know there are 302 stations
    G = DirectedWeightedGraph()

    # add stations
    for x in range(len(stations)):
        stationId = stations.iloc[x][0]
        G.add_node(stationId)
    
    # add connections
    for x in range(len(connections)):
        s1 = connections.iloc[x][0]
        s2 = connections.iloc[x][1]
        
        weight = distanceBetween(s1,s2)
    
        G.add_edge(s1,s2,weight)
        G.add_edge(s2,s1,weight)
    
    return G

# Only used for testing purposes
# Prints all weights of every possible edge 
def checkAllDistances():
    for x in range(1, 303):
        for i in range(1,303):
            if i == 189 or x == 189:
                pass
            else:
                n = g.w(x,i)
                if n == None:
                    pass
                else:
                    print(n)

g = createLondonGraph()








