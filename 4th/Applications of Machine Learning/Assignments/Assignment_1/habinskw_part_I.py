import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

# Author: Wyatt Habinski
# Created: Sep 18, 2024
# Purpose: This python file includes OLS and Gradient Decent methods for training Linear Regression 
#          Testing differences between GD and OLS
 
# Usage: python3 habinskw_part_I.py

# Dependencies: None
# Python Version: 3.6+

# References: 
# Week 1 linear_regression.py 

# Modification History:
# - Version 1 - added linear regression implementation


## PART I ## --------------------------------------------------------------------------------------------------------------------------

global df, best_gd # global variables

# function to import appropriate data from csv file for data analysis
def import_data():
    global df
    # import csv data
    data = pd.read_csv("gdp-vs-happiness.csv")

    # filter data to only display rows from 2018 and drop unused columns
    by_year = (data[data['Year']==2018]).drop(columns=["Continent","Population (historical estimates)","Code"])
    # remove rows where columns have missing values
    df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]


# function to generate and return target variable and feature vector
def get_variables():
    # initialize empty lists to store happiness and GDP values 
    happiness=[]
    gdp=[]
    # iterate over each row of dataframe
    for row in df.iterrows(): 
        if row[1]['Cantril ladder score']>4.5: #append respective values to lists for gdp and happiness where happiness score is above 4.5
            happiness.append(row[1]['Cantril ladder score'])
            gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])
            
    return happiness, gdp # return both lists

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
    
# function to plot predicted values trained by gradient decent
def plot_many_gds(lr, X, Y):
    global best_gd
    
    # initialize the alpha and iteration values for the best gradient descent
    best_gd = {
        'mse': float('inf'),
        'alpha': 0,
        'echo': 0
    }
    
    alpha_vals = [1.0, 1.0, 0.1, 0.0000001, 0.01, 0.001, 0.0001, 0.001] # list of test alpha values
    epoch_vals = [10, 100, 10000, 1000, 100, 1, 10000, 1000] # list of test epoch (iteration) values
    predict_vals = [] # initiate list of lines

    # loop through every (8) of alpha and correspong epoch value
    for i in range(len(alpha_vals)):
        beta = lr.train_gd(X,Y,alpha_vals[i],epoch_vals[i]) # get beta value for each respective pair of alpha and epoch values 
        Y_predict = { # create a record to keep track of predicted values, and what alpha and epoch value was used
        'lr': lr.predict(X,beta), # predict the values using the respective beta value
        'alpha': alpha_vals[i],
        'epoch': epoch_vals[i]
        }
        predict_vals.append(Y_predict)
        # print each respective alpha, epoch, and B' values
        print(f"Alpha: {Y_predict['alpha']}, Epoch: {Y_predict['epoch']}, B': {beta}")
        
        # calculate mse and check if its current lowest
        mse = lr.calculate_mse(Y, Y_predict['lr'])
        if mse < best_gd['mse']:
            # set best gd values
            best_gd['mse'] = mse
            best_gd['alpha'] = Y_predict['alpha']
            best_gd['epoch'] = Y_predict['epoch']

    print(f"Best MSE = {best_gd['mse']} with alpha = {best_gd['alpha']} and epoch = {best_gd['epoch']}")

    # access the 1st column (the 0th column is all 1's)
    X_ = X[...,1].ravel()

    # set the plot and plot size
    fig, ax = plt.subplots()
    fig.set_size_inches((15,8))

    # display the X and Y points
    ax.scatter(X_,Y)

    # iterate over list of predicted values with their respective alpha and epoch values
    for x in predict_vals:
        # display each respective line on the graph
        ax.plot(X_,x['lr'], label=f"alpha ={x['alpha']} epoch ={x['epoch']} ")

    #set the title, x-label, and y-label
    ax.set_xlabel("GDP per capita")
    ax.set_ylabel("Happiness")
    ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)\n Gradien Descent Comparisons")

    #show the legend and plot
    ax.legend()
    plt.show()
    
# function to plot predicted values trained by gradient decent vs trainned by ols
def plot_ols_vs_gd(lr, X, Y):
    global best_gd
    
    beta_gd = lr.train_gd(X,Y,best_gd['alpha'],best_gd['epoch']) # train data via gd method
    beta_ols = lr.train_ols(X,Y) # train data via ols method
    
    Y_predict_gd = lr.predict(X,beta_gd) # prediced values from gd method
    Y_predict_ols = lr.predict(X,beta_ols) # predicted values from ols method
    
    # access the 1st column (the 0th column is all 1's)
    X_ = X[...,1].ravel()

    # set the plot and plot size
    fig, ax = plt.subplots()
    fig.set_size_inches((15,8))

    # display the X and Y points
    ax.scatter(X_,Y)

    ax.plot(X_,Y_predict_gd, label="gd") # plot predicted values from gd method
    print(f"Gradient Descent: Alpha: {best_gd['alpha']}, Epoch: {best_gd['epoch']}, B': {beta_gd}") # print alpha, epoch and B' values for gd
    ax.plot(X_,Y_predict_ols, label="ols") # plot predicted values from ols method
    print(f"OLS: B': {beta_ols}") # print B' values for ols
    
    print(f"GD MSE = {lr.calculate_mse(Y, Y_predict_gd)}")
    print(f"OLS MSE = {lr.calculate_mse(Y, Y_predict_ols)}")

    #set the title, x-label, and y-label
    ax.set_xlabel("GDP per capita")
    ax.set_ylabel("Happiness")
    ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)\n Gradient Descent vs OLS")

    #show the legend and plot
    ax.legend()
    plt.show()
    
# linear regression class that implements both OLS and Gradient Descent for training
class linear_regression():
    
    # constructor that converts input lists to numpy arrays 
    def __init__(self,x_:list,y_:list):
        
        self.input = np.array(x_) # feture vector
        self.target = np.array(y_) # target variable

    # function to preprocess data by normalizing and arranging in matrix format
    def preprocess(self,):

        # normalize the feature vector
        xmean = np.mean(self.input) # calculate mean
        xstd = np.std(self.input) # calculate standard deviation
        x_train = (self.input - xmean)/xstd # standardize

        # arrange in matrix format with initial column of all 1's
        X = np.column_stack((np.ones(len(x_train)),x_train))

        # normalize the target variable
        ymean = np.mean(self.target) # calculate mean
        ystd = np.std(self.target) # calculate standard deviation
        y_train = (self.target - ymean)/ystd # standardize

        #arrange in matrix format then transpose
        Y = (np.column_stack(y_train)).T

        return X, Y # return both matricies

    # function to calculate mse
    def calculate_mse(self, y_actual, y_predicted):
        mse = 0 # initialize at 0
        n = len(y_actual)
        for i in range(n):
            mse = mse + ((y_actual[i] - y_predicted[i]))**2 # mse formula
        return mse / n
    
    # function to train model using ordinary least squres (OLS) method 
    def train_ols(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    # function to train model using gradient descent (GD) method 
    def train_gd(self, X, Y, alpha, iterations):
        beta = np.random.randn(2,1) # initialize beta matrix (2x1) with random values 
        n = len(X) # number of data points 
        
        # iterate based on number of iterations given
        for _ in range(iterations):
            gradients = 2/n * (X.T).dot(X.dot(beta) - Y) # gradient of cost function with respect to beta
            beta = beta - alpha * gradients # update weights based on learning rate
        
        return beta # return optomized beta

    # function to predict the target values used trained beta
    def predict(self, X, beta):
        #predict using element wise multiplication
        Y_hat = X*beta.T
        return np.sum(Y_hat,axis=1) # sum resulting features across all columns to get predicted values


# -------------------------------------------------------------

# set seed for numpys random number generator
np.random.seed(42)

# import csv data file
import_data()
# generate target variable and feature vector
happiness, gdp = get_variables()

#instantiate the linear_regression model and preprocess inputs
lr = linear_regression(gdp, happiness)
X,Y = lr.preprocess()

# plot both graphs
plot_many_gds(lr, X, Y)
plot_ols_vs_gd(lr, X, Y)







