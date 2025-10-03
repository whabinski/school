import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from assignment3 import perceptron_3d, perceptron_trans, calculate_margin, find_intersections


def extract_data(file):
    s = pd.read_csv(file)
    examples = s[["x1", "x2"]].values
    labels = s["label"].values
    
    return examples, labels
    
def plot(examples, labels, title):
    for x, y in zip(examples, labels):
        if y == 1:
            plt.scatter(x[0], x[1], color="red", marker="x", label="Positive" if "Positive" not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(x[0], x[1], color="blue", marker="o", label="Negative" if "Negative" not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot the origin with square marker, colored by its label
    origin_label = labels[(examples[:, 0] == 0) & (examples[:, 1] == 0)]
    if len(origin_label) > 0:
        color = "red" if origin_label[0] == 1 else "blue"
        plt.scatter(0, 0, color=color, marker="s", label="Origin")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    
def plot_with_w(examples, labels, w, title="Decision Boundary"):
    for x, y in zip(examples, labels):
        if y == 1:
            plt.scatter(x[0], x[1], color="red", marker="x", label="Positive" if "Positive" not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(x[0], x[1], color="blue", marker="o", label="Negative" if "Negative" not in plt.gca().get_legend_handles_labels()[1] else "")

    # Decision boundary
    x_vals = np.linspace(-1.5, 1.5, 100)
    y_vals = -(w[0]/w[1]) * x_vals - (w[2]/w[1])
    plt.plot(x_vals, y_vals, label="Decision Boundary", color='blue')

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    
def plot_decision_boundary_from_intersections(w, examples, labels):
    intersections, origin_label = find_intersections(w)

    # Plot the dataset
    for x, y in zip(examples, labels):
        color = "red" if y == 1 else "blue"
        marker = "x" if y == 1 else "o"
        plt.scatter(x[0], x[1], color=color, marker=marker,
                    label="Positive" if y == 1 and "Positive" not in plt.gca().get_legend_handles_labels()[1]
                    else "Negative" if y == -1 and "Negative" not in plt.gca().get_legend_handles_labels()[1]
                    else "")

    # Plot the decision boundary
    if len(intersections) >= 2:
        (x1_1, x2_1), (x1_2, x2_2) = intersections[:2]
        plt.plot([x1_1, x1_2], [x2_1, x2_2], color="black", label="Decision Boundary")

    # Plot origin with its label color
    origin_color = "green" if origin_label == 1 else "green"
    plt.scatter(0, 0, color=origin_color, marker="s", label="Origin")

    plt.title("Decision Boundary from find_intersections")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.show()
    
def plot_decision_line(w):
    intersections, origin_label = find_intersections(w)
    print(f"Intersections: {intersections}, \nOrigin: {origin_label}")
    
    # Plot the square
    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]
    plt.plot(square_x, square_y, color='black', label='Boundary Square')

    # Plot decision boundary if there are intersections
    if len(intersections) > 0:
        x_vals = [pt[0] for pt in intersections]
        y_vals = [pt[1] for pt in intersections]
        plt.plot(x_vals, y_vals, color='blue', label='Decision Line')

    # Plot origin with predicted label color
    color = 'red' if origin_label == 1 else 'blue'
    plt.scatter(0, 0, color=color, s=100, marker='x', label='Origin Prediction')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.title(f"Decision Boundary for w = {w}")
    plt.grid(True)
    plt.show()
    
def test_intersection():
    #good_w = [0.85694716, 0.98136326, -1.0]
    case_1 = [0, 1, -2]
    case_2 = [1, 1, -2]
    case_3 = [1, -1, 0]
    case_4 = [0, 1, -1]
    case_origin = [1.0, -1.0, 2.0]
    plot_decision_line(case_3)
#--------------------------------------------------------------------------------------------------

def a(s1_examples, s2_examples, s1_labels, s2_labels):
    # Plot each dataset
    plot(s1_examples, s1_labels, "S1")
    plot(s2_examples, s2_labels, "S2")
    
def c(s1_examples, s2_examples, s1_labels, s2_labels):
    # Run Perceptron
    s1_mistakes, s1_weights = perceptron_trans(s1_examples, s1_labels)
    s2_mistakes, s2_weights = perceptron_trans(s2_examples, s2_labels)

    # Print Mistakes
    print(f"S1 mistakes: {s1_mistakes}")
    print(f"S2 mistakes: {s2_mistakes}")
    
def d(s1_examples, s1_labels):
    # Run Perceptron on s1
    s1_mistakes, s1_weights = perceptron_trans(s1_examples, s1_labels)
    
    print(f"S1 mistakes: {s1_mistakes}")
    
    # w = w^(t+1)
    w = s1_weights[-1]
    
    print(f"S1 w: {w}")
    
    plot_with_w(s1_examples, s1_labels, w)
    
    # Translate from dimension 2 to dimension 3
    s1_examples_trans = np.array([[x[0], x[1], 1] for x in s1_examples])
    
    print(s1_examples_trans)
    
    # Calculate margin
    margin = calculate_margin(w, s1_examples_trans, s1_labels)
    
    # Print margin
    print(f"S1 Margin: {margin}")
    
def d_sklearn(s1_examples, s1_labels):
    # Run Perceptron on s1
    s1_mistakes, s1_weights = perceptron_3d_sklearn(s1_examples, s1_labels)
    
    print(f"S1 mistakes: {s1_mistakes}")
    
    # w = w^(t+1)
    w = s1_weights[-1]
    
    print(f"S1 w: {w}")
    
    plot_with_w(s1_examples, s1_labels, w)
    
    # Translate from dimension 2 to dimension 3
    s1_examples_trans = np.array([[x[0], x[1], 1] for x in s1_examples])
    
    print(s1_examples_trans)
    
    # Calculate margin
    margin = calculate_margin(w, s1_examples_trans, s1_labels)
    
    # Print margin
    print(f"S1 Margin: {margin}")
    
def e(s3_examples, s3_labels):
    # Run Perceptron
    s3_mistakes, s3_weights = perceptron_trans(s3_examples, s3_labels)

    # Print Mistakes
    print(f"S3 mistakes: {s3_mistakes}")
    
#--------------------------------------------------------------------------------------------------    
    
from sklearn.linear_model import Perceptron

def perceptron_3d_sklearn(examples, labels):
    
    examples_trans = np.array([[x[0], x[1], 1] for x in examples])

    clf = Perceptron(tol=1e-3, max_iter=1000, random_state=0, fit_intercept=False)
    clf.fit(examples_trans, labels)

    # Predict on training data
    predictions = clf.predict(examples_trans)
    mistakes = np.sum(predictions != labels)

    # Final weight vector (since we don't track every step, just return final)
    weights = [clf.coef_[0]]

    return mistakes, weights    

import matplotlib.pyplot as plt

    
# Load datasets
s1_examples, s1_labels = extract_data("data1.csv")
s2_examples, s2_labels = extract_data("data2.csv")
s3_examples, s3_labels = extract_data("data3.csv")

#a(s1_examples, s2_examples, s1_labels, s2_labels)

#c(s1_examples, s2_examples, s1_labels, s2_labels)

#d(s1_examples, s1_labels)
#d_sklearn(s1_examples,s1_labels)

#e(s3_examples, s3_labels)



