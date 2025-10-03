# Name: Wyatt Habinski
# MacId: habinskw
import numpy as np


"""
Runs perceptron algorith for 3rd dimensional space

Args:
    examples: List of 3-dimensional vectors
    labels: List of labels

Returns:
    mistakes: Number of mistakes
    weights: List of weight vectors [w^(1), ..., w^(T+1)]
"""
def perceptron_3d(examples, labels):
    
    T = len(examples)               # total number of examples
    w_t = np.zeros(3)               # initialize first default weight vector for dimension = 3 as all 0s
    
    mistakes = 0                    # initialize running count of mistakes
    weights = [w_t.copy()]          # initialize weight list and add first default wweight
    
    for t in range(T):              # iterate through all examples
        x_t = examples[t]           # current example
        y_t = labels[t]             # current label
        
        prediction = 1 if np.dot(w_t, x_t) > 0 else -1  # prediction of current example using current weight vector

        if prediction != y_t:       # check if prediction is wrong
            w_t = w_t + y_t * x_t   # update weight
            mistakes += 1           # update mistake counter
    
        weights.append(w_t.copy())  # add weight to weights list

    return mistakes, weights        # return mistake count and weights list

"""
Transforsm 2 dimensional examples to 3rd dimension and runs perceptron

Args:
    examples_2d: List of 2-dimensional examples
    labels: List of labels

Returns:
    mistakes: Number of mistakes
    weights: List of weight vectors [w^(1), ..., w^(T+1)]
"""
def perceptron_trans(examples_2d, labels):

    examples_trans = np.array([[x[0], x[1], 1] for x in examples_2d])   # adds 3rd dimension with values of 1 to all examples

    return perceptron_3d(examples_trans, labels)    # calls perceptron as return

"""
Calculates margin given weight vector

Args:
    w: 3-dimensional weight vector
    examples: List of 3-dimensional vectors
    labels: List of labels

Returns:
    margin: The margin if it exists, or -1 if it doesn't
"""
def calculate_margin(w, examples, labels):
    
    T = len(examples)               # total number of examples
    margins = []                    # initialize margins list
    
    for t in range(T):              # iterate through all examples
        x_t = examples[t]           # current example
        y_t = labels[t]             # current label
        
        prediction = 1 if np.dot(w, x_t) > 0 else -1    # prediction of current example using current weight vector
        
        if prediction != y_t:       # check if prediction is correct
            return -1               # return -1 if even one prediction is incorrect
        
        margin = abs(np.dot(w,x_t)) / np.linalg.norm(w)     # calculate margin for current example
        margins.append(margin)                              # add margin to list of margins
    
    return min(margins)     # return the smallest margin

"""
    Finds intersection of decision boundry defined by w vector and a 1 x 1 unit square

    If the line has no intersection with the square, the algorithm should return an empty list [].
    If the line intersects the square at only one point (x1,y1), the algorithm should return [(x1, y1), (x1, y1)].
    If the line intersects the square at two different points (x1,y1) and (x2,y2), the algorithm should return [(x1, y1), (x2, y2)].
    If the line lies exactly on the boundary of the square, the algorithm should return the extreme points of that boundary—specifically, its intersections with the other two boundaries—in the format.[(x1, y1), (x2, y2)].

    Args:
        w: 3-dimensional weight vector

    Returns:
        intersections: List of 2D points [(x1, y1), (x2, y2)]
        origin_label: The predicted label (1 or -1) for the origin (0,0)

    """
def find_intersections(w):
    w1, w2, w3 = w[0], w[1], w[2]       # initialize w's based on weight vector
    intersections = []                  # initialize intersections list

    if w1 == 0 and w2 == 0 and w3 == 0:         # check if all weight values are 0
        return [], None                         # return empty list and None if w = [0,0,0]

    
    # Check intersection along x1 values using x2 = (-w1 * x1 - w3) / w2
    for x1 in [-1, 1]:                              # iterate through all values within the x1 values of the box
        if w2 != 0:                                 # make sure division by 0 is not possible
            x2 = (-w1 * x1 - w3) / w2               # calculate x2
            if -1 <= x2 <= 1:                       # check if x2 is within the values
                intersections.append((x1, x2))      # append the point to intersections list

    # Check intersection along x2 values using x1 = (-w2 * x2 - w3) / w1
    for x2 in [-1, 1]:                              # iterate through all values within the x2 values of the box
        if w1 != 0:                                 # make sure division by 0 is not possible
            x1 = (-w2 * x2 - w3) / w1               # calculate x1
            if -1 <= x1 <= 1:                       # check if x1 is within the values
                intersections.append((x1, x2))      # append the point to intersections list

    # Case 2 - if line only intersects with 1 point, return it twice 
    if len(intersections) == 1:                     # check if the intersection only has 1 point
        intersections.append(intersections[0])      # duplicate the point

    # Compute label at the origin
    origin = np.array([0, 0, 1])                            # origin is sing(w * (0,0,1)) = sign(w2)
    origin_label = 1 if np.dot(w, origin) > 0 else -1       # assign label to origin

    return intersections, origin_label              # return intersections list and origin label
