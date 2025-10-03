import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn

import sklearn.impute                                   # provides methods for handling missing data
import sklearn.preprocessing                            # provides methods for data normalization
from sklearn.svm import SVC                             # provides implementation of SVM Classifier
from sklearn.model_selection import KFold               # provides implementation of KFold cross validation methods
from sklearn.metrics import confusion_matrix            # provides methods to compute confusion matrixes
from sklearn.metrics import ConfusionMatrixDisplay      # provides methods to display confusion matrixes

# Author: Wyatt Habinski
# Created: October 1, 2024
# Purpose: develop, evaluate and visualize a classification model that predicts solar flare events using support vector machine classifiers
 
# Usage: python3 habinskw.py

# Dependencies: None
# Python Version: 3.6+

# References: 
# starter code

# Modification History:
# - Version 1 - implemented svm model with evaluation and visualization methods

class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self,):
        
        self.data_old = 'data-2010-15/' # old dataset path
        self.data_new = 'data-2020-24/' # new dataset path
        self.data_age = None # model's dataset path
        
        # dictionary to store feature sets
        self.feature_sets = {
            'FS-I': None,   # main time change features (columns 1-18)
            'FS-II': None,  # main time change features (columns 19-90)
            'FS-III': None, # historical activity features
            'FS-IV': None   # min max features
        }
        
        self.labels = None # model's labels for positive and negative samples
        self.feature_matrix = None # models' 2D feature matrix
        self.trained_model = None # trained SVM model

    # preprocess() function:
    #  1) normalizes the data, 
    #  2) removes missing values
    #  3) assign labels to target 
     # Preprocess function to normalize, clean, and reorder data based on data_order
    def preprocess(self):
        
        # load all positive and negative class features from the npy files
        
        # positive and negative main_timechange files
        pos_main_timechange = np.load(f'{self.data_age}pos_features_main_timechange.npy', allow_pickle=True)
        neg_main_timechange = np.load(f'{self.data_age}neg_features_main_timechange.npy', allow_pickle=True)
        
        # positive and negative historical files
        pos_historical = np.load(f'{self.data_age}pos_features_historical.npy', allow_pickle=True)
        neg_historical = np.load(f'{self.data_age}neg_features_historical.npy', allow_pickle=True)
        
        # positive and negative maxmin files
        pos_maxmin = np.load(f'{self.data_age}pos_features_maxmin.npy', allow_pickle=True)
        neg_maxmin = np.load(f'{self.data_age}neg_features_maxmin.npy', allow_pickle=True)
        
        # positive and negative class files
        pos_class = np.load(f'{self.data_age}pos_class.npy', allow_pickle=True)
        neg_class = np.load(f'{self.data_age}neg_class.npy', allow_pickle=True)
        
        # data order file
        data_order = np.load(f'{self.data_age}data_order.npy', allow_pickle=True)

        # load all feature sets for both positive and negative classes
        # combine positive and negative into same feature set in form of concatenating array (positive first, then negative)
        self.feature_sets['FS-I'] = np.concatenate((pos_main_timechange[:, :18], neg_main_timechange[:, :18]), axis=0) #Create FS-I
        self.feature_sets['FS-II'] = np.concatenate((pos_main_timechange[:, 18:90], neg_main_timechange[:, 18:90]), axis=0) #Create FS-II
        self.feature_sets['FS-III'] = np.concatenate((pos_historical,neg_historical), axis=0) #Create FS-III
        self.feature_sets['FS-IV'] = np.concatenate((pos_maxmin, neg_maxmin), axis=0) #Create FS-IV

        # instantiate simple imputer for handling missing data 
        # instantiate standard scalar for normalizing data
        si = sklearn.impute.SimpleImputer(strategy='mean') # replace missing values with the mean of the column
        ss = sklearn.preprocessing.StandardScaler() # standardize features
        
        # handle missing data and normalize each feature set
        # reorder feature set based on data order file
        for fs_key in ['FS-I', 'FS-II', 'FS-III', 'FS-IV']:
            self.feature_sets[fs_key] = si.fit_transform(self.feature_sets[fs_key]) # apply simple imputer for missing data
            self.feature_sets[fs_key] = ss.fit_transform(self.feature_sets[fs_key]) # apply standard scaler for normalization
            self.feature_sets[fs_key] = self.feature_sets[fs_key][data_order] # reorder based on data_order
        
        # create positive and negative labels
        labels = np.concatenate((np.ones(len(pos_class)), np.zeros(len(neg_class))))  # 1 for positive, 0 for negative
        self.labels = labels[data_order]  # reorder labels based on data_order
        
        return
    
    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2 D array of corresponding features 
    # for both positive and negative observations.
    # this array will be input to the svm model
    # For instance, if the input is FS-I, the output is a 2-d array with features corresponding to 
    # FS-I for both negative and positive class observations
    def feature_creation(self, fs_value):
        feature_list = [] # initialize empty list to hold the selected feature sets

        # Iterate over the selected feature sets
        for fs in fs_value:
            # Append the respective feature set to the list of feature sets
            feature_list.append(self.feature_sets[fs])
        
        # Concatenate the feature sets along the second axis (columns) to create one 2D array of all feature sets
        self.feature_matrix = np.concatenate(feature_list, axis=1)
         
        return 
    
    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # the svm is trained on training set and tested on test set
    # the output is the average accuracy across all train test splits.
    def cross_validation(self):
        
        tss_scores = [] # array to store the TSS values for each fold
        cm_total = np.array([[0, 0], [0, 0]]) # to store confusion matrix values for all folds
        
        # k-fold cross-validation with k=10 splits
        # Split the data into 10 differnt train and test sets
        kf = KFold(n_splits=10, shuffle=True)

        # split the feature matrix data using k fold
        # iterate over each fold of training and test sets
        for train_index, test_index in kf.split(self.feature_matrix):
            x_train = self.feature_matrix[train_index] # training features for respective fold
            x_test = self.feature_matrix[test_index] # testing features for respective fold
            y_train = self.labels[train_index] # training predictions for respective fold
            y_test = self.labels[test_index] # testing predictions for respective fold
            
            # train the model using the current fold
            self.training(x_train, y_train)
            
            # predict using the current test set
            y_predict = self.trained_model.predict(x_test)
            
            # calculate the tss scores for this fold
            tss = self.tss(y_test,y_predict) 
            tss_scores.append(tss) # add to list for all folds
            
            # calculate the confusion matrix for this fold
            cm = confusion_matrix(y_test, y_predict)
            cm_total += cm # add to total for all folds
            
        # calculate mean and standard deviation of tss scores
        tss_mean = np.mean(tss_scores) # average tss score for all folds
        tss_std = np.std(tss_scores) # standard deviation tss score for all folds
        
        # return mean tss, standard deviation tss, tss scores for each fold, and combined confusion matrix for all folds
        return tss_mean, tss_std, tss_scores, cm_total 
    
    #training() function trains a SVM classification model on input features and corresponding target
    def training(self, x_train, y_train):

        # initialize and train the SVM model
        self.trained_model = SVC()  # initialize svm classifier
        self.trained_model.fit(x_train, y_train) # train the model using fit, with training features and labels
        
        return

    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    # using the TSS measure given in the document
    def tss(self, y_test, y_predict):
        
        # compute the confusion matrix to get count of tn, fp, fn, tp
        cm = confusion_matrix(y_test, y_predict)
        tn, fp, fn, tp = cm.ravel()

        # tss calculation
        tss_score = (tp / (tp + fn)) - (fp / (fp + tn))
            
        # return tss score
        return tss_score

# feature_experiment() function executes experiments with all 4 feature sets.
# svm is trained (and tested) on 2010 dataset with all 4 feature set combinations
# the output of this function is : 
#  1) TSS average scores (mean std) for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e 10)
#
# Above 3 charts are produced for all feature combinations
#  4) The function prints the best performing feature set combination
def feature_experiment():
    
    # define all possible feature set combinations given FS-I, FS-II, FS-III, and FS-IV
    # total 15 combinations
    feature_set_combinations = [
        ['FS-I'], ['FS-II'], ['FS-III'], ['FS-IV'],  # 4
        ['FS-I', 'FS-II'], ['FS-I', 'FS-III'], ['FS-I', 'FS-IV'], ['FS-II', 'FS-III'], ['FS-II', 'FS-IV'], ['FS-III', 'FS-IV'], # 6
        ['FS-I', 'FS-II', 'FS-III'], ['FS-I', 'FS-III', 'FS-IV'], ['FS-I', 'FS-II', 'FS-IV'], ['FS-II', 'FS-III', 'FS-IV'], # 4
        ['FS-I', 'FS-II', 'FS-III','FS-IV'] # 1 
    ]
    
    # initialize to track feature set with best tss score
    best_score = {'fs': None, 'mean': 0, 'std': 100}
    # initialize dictionaries used for plotting graphs
    confusion_matrixes = {'cm': [], 'titles': []} # to store a list of all confusion matrices and their respective titles
    scatter_plots = {'sp': [], 'titles': []} # to store a list of all scatter plots and their respective titles
    
    # iterate through all feature set combinations and perform cross validations
    for fs in feature_set_combinations:
        
        model = my_svm() # initialize svm model
        model.data_age = model.data_old # set data path for data age (use old data for this experiment)
        
        model.preprocess() # preprocess the data
        model.feature_creation(fs) # create feature matrix for respective feature set
        tss_mean, tss_std, tss_scores, cm_total = model.cross_validation() # perform cross validation and store values
        
        # update best performing feature set if the current one is better
        # 'better' means within 0.03 or better of the mean, and within 0.03 or better of the standard deviation
        if (tss_mean >= best_score['mean'] - 0.03) and (tss_std < best_score['std'] + 0.03):
            best_score = {'fs': fs, 'mean': tss_mean, 'std': tss_std}
        
        # add confusion matrix values to list of confusion matrixes for future plotting
        confusion_matrixes['cm'].append(cm_total) # add amtrix
        confusion_matrixes['titles'].append(f"Confusion Matrix for {fs}") # add title
        
        # add scatter plot values to list of scatter plots for future graphing
        scatter_plots['sp'].append({'x': range(1, len(tss_scores) + 1), 'y': tss_scores}) # add scatter plot values as another dictionary of x and y values
        scatter_plots['titles'].append(f"TSS Scores across folds for {fs}") # add title
        
        # print out tss scores for each respective feature set
        print(f"Feature Set Combination: {fs}")
        print(f"Mean TSS: {tss_mean:.4f}")
        print(f"Standard Deviation of TSS: {tss_std:.4f}")
        print("----------------------------------------------------------------------------------------------------------------")

    # call plot_on_grid helper for both confusion matrixes and scatter plots to display graphs
    plot_on_grid(confusion_matrixes['cm'], confusion_matrixes['titles'], "ConfusionMatrix")
    plot_on_grid(scatter_plots['sp'], scatter_plots['titles'], "Scatter")
    print(f"Best Performing Feature Set: {best_score['fs']}")
    print("----------------------------------------------------------------------------------------------------------------")
    
    # return the best performing feature set
    return best_score['fs']


# data_experiment() function executes 2 experiments with 2 data sets.
# svm is trained (and tested) on both 2010 data and 2020 data
# the output of this function is : 
#  1) TSS average scores for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e. 10)
# above 3 charts are produced for both datasets
# feature set for this experiment should be the 
# best performing feature set combination from feature_experiment()
def data_experiment(best_fs):
    
    # list of datasets to experiment with
    experiments = ['2010-2015', '2020-2024']
    
    # create a 2x2 grid of plots
    # each row will have 1 confusion matrix and 1 scatter plot corresponding to the experiment
    fig, ax = plt.subplots(2, 2, figsize=(10, 10)) 
    
    # for both experiments, create, train, and test the model
    for i, exp in enumerate(experiments):
        model = my_svm() # initialize model
        # set appropriate datapath
        if exp == '2010-2015':
            model.data_age = model.data_old # older dataset
        else:
            model.data_age = model.data_new # newer dataset
            
        model.preprocess() # preprocess the model
        model.feature_creation(best_fs) # create futureset using the predefined 'best' featureset from earlier experiment
        
        tss_mean, tss_std, tss_scores, cm_total = model.cross_validation() # perform cross validation and store values
        
        # plot confusion matrix
        ax_cm = ax[i, 0]  # confusion matrix will go in first column (0) and respective row i
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_total) # create confusion matrix display using values already obtained
        disp.plot(ax=ax_cm) # plot on the specified axis
        ax_cm.set_title(f"Confusion Matrix for {exp}") # confusion matrix title
        
        # plot scatter plot of tss scores
        ax_scatter = ax[i, 1]  # scatter plot will go in second column (1) and respective row i
        ax_scatter.scatter(range(1, len(tss_scores) + 1), tss_scores) # create scatter plot given x and y values
        ax_scatter.set_xlabel('Fold Number') # x label
        ax_scatter.set_ylabel('TSS Score') # y label
        ax_scatter.set_title(f"TSS Scores across folds for {exp}") # scatter plot title
        
        # print out tss scores for each respective data set
        print(f"Experiment for dataset: {exp}")
        print(f"Mean TSS: {tss_mean:.4f}")
        print(f"Standard Deviation of TSS: {tss_std:.4f}")
        print("----------------------------------------------------------------------------------------------------------------")
    
    plt.tight_layout()  # tighter layout of graphs to prevent overlapping
    plt.show()  # show all plots on a single window

# helper function to plot given graphs on one window
# supports confusion matrixes or scatter plots
def plot_on_grid(plots, titles, plot_type='Scatter'):
    
    num_plots = len(plots) # number of plots to display
    cols = 3 # number of columns
    rows = (num_plots + cols - 1) // cols  # adjust number of rows based on number of plots

    fig, ax = plt.subplots(rows, cols, figsize=(10, 10)) # create grid of subplots of specific size (10x10)
    ax = ax.flatten()  # flatten axes used for simplified looping
    
    # plot the graphs
    if plot_type == 'ConfusionMatrix':
        # if its a list of confusion matrixes
        # iterate through all graphs
        for i, cm in enumerate(plots):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm) # create confusion matrix for respective matrix values
            disp.plot(ax=ax[i], colorbar=False)  # plot confusion matrix in subplot
            ax[i].set_title(titles[i], fontsize=8)  # title
    else:
        # if its a list of scatter plots
        # iterate through all graphs
        for i, sp in enumerate(plots):
            x = sp['x'] # get the graphs x values
            y = sp['y'] # get the graphs y values
            ax[i].scatter(x, y)  # plot scatter plot in the subplot
            ax[i].set_xlabel('Fold Number', fontsize=7) # x label
            ax[i].set_ylabel('TSS Score', fontsize=7) # y label
            ax[i].set_title(titles[i], fontsize=8) # title
    
    # hide any unused subplots ( if there are more cells in the grid than plots) to precent weird layout
    for i in range(num_plots, cols * rows):
        fig.delaxes(ax[i]) # delete the extra suplot

    plt.tight_layout()  # tighter layout of graphs to prevent overlapping
    plt.show()  # show all plots on a single window

#------------------------------------------------------------------------------------------------------------------------------------------

# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets

np.random.seed(42) # initialize seed for randomization

best_fs = feature_experiment() # call first experiment and return best feature set
data_experiment(best_fs) # call second experiment using the best feature set




