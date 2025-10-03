import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

# Author: Wyatt Habinski
# Created: Sep 18, 2024
# Purpose: This python file includes an OLS methods for training Polynomial Regression 
#          to predict the age of abolones given their features      
#          The cost function that is used is MAE    
 
# Usage: python3 habinskw_part_II.py

# Dependencies: None
# Python Version: 3.6+

# References: 
# Week 1 habinskw_part_I.py 

# Modification History:
# - Version 1 - 


## PART II ## --------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function to import appropriate data from csv file for data analysis
def import_data():
    df = pd.read_csv("training_data.csv")

    # set features and target
    features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = df[features].values
    Y = df['Rings'].values + 1.5  # Adding 1.5 to compute age
    
    return X, Y, features

# function to split training and test data
# for testing purposes only
def split_data(X, Y):
    # Split data into training and test sets (80% train, 20% test)
    train_size = int(0.8 * len(X))
    # training sets
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    # test sets
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    
    return X_train, Y_train, X_test, Y_test

# function to calculate mse
def calculate_mse(Y_actual, Y_predicted):
    return np.mean((Y_actual - Y_predicted) ** 2)

# function to calculate rmse
def calculate_rmse(Y_actual, Y_predicted):
    return np.sqrt(calculate_mse(Y_actual, Y_predicted))

# function to calculate mae
def calculate_mae(Y_actual, Y_predicted):
    return np.mean(np.abs(Y_actual - Y_predicted))

# function to predict model, and calculate cost functions
def evaluate_model(pr, X, Y, beta):

    Y_predicted = pr.predict(X, beta) # predict values using the model
    # calculate cost functions
    mse = calculate_mse(Y, Y_predicted)
    rmse = calculate_rmse(Y, Y_predicted)
    mae = calculate_mae(Y, Y_predicted)
    
    # print cost functions
    #print(f'MSE: {mse}')
    #print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    return Y_predicted

# Report beta values (coefficients)
def report_beta(beta):
    print(f"B': {beta.T}")
    print(f"B' length: {len(beta)}")

# Scatter plot of true vs predicted values
def scatter_matrix(X, Y, Y_predicted, features):
    plt.figure(figsize=(15, 10))  # set plot size
    
    # create subplot for 7 feature vs age plots
    for i, feature in enumerate(features):
        plt.subplot(3, 3, i + 1)  # 3 rows, 3 columns, each plot is a subplot
        plt.scatter(np.take(X, i, axis=1), Y, c='blue', label='Actual values', alpha=0.5) # plot actual values
        plt.scatter(np.take(X, i, axis=1), Y_predicted, c='red', label='Predicted values', alpha=0.5) # plot predicted values
        
        # set the title, x-label, and y-label
        plt.title(f'{feature} vs Age')
        plt.xlabel(feature)
        plt.ylabel('Age')
    
    plt.tight_layout()
    plt.show()

# polynomial regression class that implements OLS method for training
class PolynomialRegression:
    # constructor that converts input lists to numpy arrays 
    def __init__(self, x_: list, y_: list, degree: int = 2):
        self.input = np.array(x_)  # feature vector
        self.target = np.array(y_)  # target variable
        self.degree = degree # polynomial degree

    # function to preprocess data by organizing into a polynomial feature array
    def preprocess(self):
        
        n = len(self.input)
        X = np.ones((n, 1))  # Start with bias (intercept) column
        # for each degree in the polynomial, add the corresponding column in the matrix
        for d in range(1, self.degree + 1):
            X = np.column_stack((X, self.input ** d))
        
        # arrange target in matrix format then transpose
        Y = (np.array([self.target])).T
        
        return X, Y # return both matricies

    # function to train model using ordinary least squres (OLS) method 
    def train_ols(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        
    # function to predict the target values used trained beta
    def predict(self, X, beta):
        #predict using element wise multiplication
        Y_hat = X*beta.T
        return np.sum(Y_hat,axis=1)

# -------------------------------------------------------------

# set seed for numpys random number generator
np.random.seed(42)

# import csv data file
# generate target and feature vectors
X, Y, features = import_data()

# instantiate the polynomial regression model
pr = PolynomialRegression(X, Y, degree=2)

# preprocess the data (generate polynomial features)
X_poly, Y_poly = pr.preprocess()

# train the model using OLS
beta = pr.train_ols(X_poly, Y_poly)
# print the beta values
report_beta(beta)

# evaluate the model
Y_pred = evaluate_model(pr, X_poly, Y, beta)

# Scatter plot of actual vs predicted values
scatter_matrix(X, Y, Y_pred, features)