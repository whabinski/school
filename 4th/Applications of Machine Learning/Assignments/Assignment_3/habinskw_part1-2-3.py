
# Author: Wyatt Habinski
# Created: Oct 31, 2024
# License: MIT License
# Purpose: This python file includes a calss for an svm classification model using different training
    # methods such as stochastic and mini batch gradient descent with different optomization strategies

# Usage: python3 habinskw_part1-2-3.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation
# - Tutorial 3 Assignment3 boilerplate code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


class svm_():
    # Initialize svm model
    def __init__(self,learning_rate,epoch,C_value,X,Y):

        #initialize the variables
        self.input = X                         # input feature matrix
        self.target = Y                        # target labels
        self.learning_rate =learning_rate      # step size for gradient descent updates
        self.epoch = epoch                     # the number of iterations to be trained
        self.C = C_value                       # the regularization parameter

        #initialize the weight matrix based on number of features 
        # bias and weights are merged together as one matrix
        # you should try random initialization
        
        self.weights = np.random.randn(X.shape[1]) # Random values
        #self.weights = np.zeros(X.shape[1])    # All zeros (optional)

    # Pre process input data via standardization
    def pre_process(self,):

        #using StandardScaler to normalize the input
        scalar = StandardScaler().fit(self.input)         # scalar
        X_ = scalar.transform(self.input)                 # apply scalar to transform feature matrix

        Y_ = self.target                                  # labels remain the same

        return X_,Y_, scalar
    
    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    # Computes loss using hinge and regularization
    def compute_loss(self,X,Y):
        # calculate hinge loss
        # hinge loss implementation- start
        # Part 1
        hinge_loss = np.maximum(0, 1 - Y * np.dot(X, self.weights)).sum()       # hinge loss formula
        regularization = 0.5 * np.dot(self.weights, self.weights)               # regularization formula
        loss = (self.C * hinge_loss) + regularization                           # total loss
        # hinge loss implementatin - end
                
        return loss
    
    # Stochastic gradient descent with early stopping
    def stochastic_gradient_descent(self,X,Y,validation_set):
        print("---------------------------------------------------------------------------------------- \nBeginning stochatsic gradient gradient descent")
    
        threshold = 0.001                       # threshold for early stopping
        early_stopping_found = False            # flag if early stopping parameters are found
        previous_loss = float('inf')            # set initial loss to inf
        validation_losses = []                  # initialize list to store validation losses
        training_losses = []                    # initialize list to store training losses
        
        X_val, Y_val = validation_set           # validation input feature matrix and output labels

        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(0, self.epoch + 1):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            # update weights for each training sample 
            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)
                
            # calculate training and validation losses
            current_loss = self.compute_loss(features, output) 
            training_loss = current_loss
            validation_loss = self.compute_loss(X_val, Y_val)

            # record losses every 1/10th of epochs for tracking
            if epoch % (self.epoch // 10) == 0:
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                print("Epoch is: {} and Loss is : {}".format(epoch, training_loss))

            #check for convergence -start
            
            # Part 1
            # early stopping condition checks if cahnge in loss is below threshold
            if early_stopping_found == False and abs(previous_loss - current_loss) < threshold:
                 print("Early stopping for stochastic gradient descent at epoch: {}".format(epoch))
                 early_stopping_found = True
            previous_loss = current_loss

            #check for convergence - end
        
        return training_losses, validation_losses # return list of training lossses and validation losses
        

    # Mini batch gradient descent
    def mini_batch_gradient_descent(self,X,Y,batch_size,validation_set):
        print("---------------------------------------------------------------------------------------- \nBeginning mini batch gradient descent")

        # mini batch gradient decent implementation - start

        # Part 2
        validation_losses = []      # initialize list to store validation losses
        training_losses = []        # initialize list to store training losses
        
        X_val, Y_val = validation_set       # validation input feature matrix and output labels

        # execute the mini batch gradient descent function for defined epochs
        for epoch in range(0, self.epoch): 
            # Shuffle features and output for random batches each epoch
            features, output = shuffle(X, Y)

            # Split features and output into mini batches
            feature_batches = np.array_split(features, len(features) // batch_size)
            output_batches = np.array_split(output, len(output) // batch_size)

            # Iterate through each mini-batch
            for X_batch, Y_batch in zip(feature_batches, output_batches):
                # Compute gradient for the batch by summing each sample's gradient
                gradient_sum = np.zeros(self.weights.shape) # Initialize the gradient sum for this batch
                for i, feature in enumerate(X_batch):
                    gradient_sum += self.compute_gradient(feature, Y_batch[i])

                # Update weights using the average gradient for the batch
                gradient_avg = gradient_sum / len(X_batch)
                self.weights = self.weights - (self.learning_rate * gradient_avg) # update weights
                    
                # calculate training and validation losses
                current_loss = self.compute_loss(features, output)
                training_loss = current_loss
                validation_loss = self.compute_loss(X_val, Y_val)

            # record losses every 1/10th of epochs for tracking
            if epoch % (self.epoch // 10) == 0:
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                print("Epoch is: {} and Loss is : {}".format(epoch, training_loss))

        # mini batch gradient decent implementation - end
        
        return training_losses, validation_losses # return list of training lossses and validation losses

    # Finds sample witht the smallest SVM loss
    def sampling_strategy(self, X, Y):
        # Implementation of sampling strategy - start

        min_loss = float('inf')             # set initial min loss to infinity
        max_loss = 0                        # set initial max loss to 0
        min_X = X[0]                        # initializze as first sample in X
        min_Y = Y[0]                        # initialize as first label in Y

        # Iterate over each sample
        for (x, y) in zip(X, Y):
            loss = self.compute_loss(np.array([x]), np.array([y]))      # compute loss for the current sample
            
            # update min X and min Y if the new minimum loss is found
            if loss < min_loss:
                min_loss = loss
                min_X = x  
                min_Y = y  
                
        # Implementation of sampling strategy - end
        return min_X, min_Y     # return the X and Y with the smallest loss

    # Evaluates model performance using accuracy, precision, and recall metrics
    def predict(self,X_test,Y_test):

        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
        
        #compute accuracy
        accuracy= accuracy_score(Y_test, predicted_values)
        print("Accuracy on test dataset: {}".format(accuracy))

        #compute precision - start
        precision= precision_score(Y_test, predicted_values, pos_label=1)
        print("Precision on test dataset: {}".format(precision))
        #compute precision - end

        #compute recall - start
        recall= recall_score(Y_test, predicted_values, pos_label=1)
        print("Recall on test dataset: {}".format(recall))
        #compute recall - end

# Train and validate svm model using stochastic gradient descent
# Analyze early stopping regularization technique
def part_1(X_train,y_train,validation_set):
    print("PART I -----------------------------------------------------------------------------")
    # initialize model parameters
    C = 0.001                               # regularization parameter
    learning_rate = 0.00001                 # step size for gradient descent
    epoch = 2500                            # number of training iterations

    X_test, y_test = validation_set         # validation input feature matrix and output labels
  
    #intantiate the support vector machine class above 
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    X_train, y_train, scalar = my_svm.pre_process()     # pre process input and output training data
    X_test = scalar.transform(X_test)                   # use same scalar to pre process validation data           
    validation_set = X_test, y_test                     # update variables for validation data
    
    # train model using stochastic gradient descent
    training_losses, validation_losses = my_svm.stochastic_gradient_descent(X_train, y_train, validation_set)
    
    # test models success metrics
    print("Testing model accuracy for Stochastic Gradient Descent...")
    my_svm.predict(X_test,y_test)
    
    # visualize training vs validation loss using a plot
    plot_train_vs_val(training_losses, validation_losses, "Stochastic Gradient Descent")

    return my_svm

# Compare 2 different SVM models by using 2 seperate techniques: stochastic and mini batch gradient descent
def part_2(X_train,y_train,validation_set):
    print("PART II -----------------------------------------------------------------------------")
    # initialize model parameters
    C = 0.001                       # regularization parameter
    learning_rate = 0.001           # step size for gradient descent
    epoch = 5000                    # number of training iterations
    batch_size = 100                # batch size for mini batch gradient descent
    
    # validation set for mini batch training
    validation_set_batch = validation_set
    X_test_batch, y_test_batch = validation_set_batch
  
    #intantiate the support vector machine class above
    mini_batch_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    X_train_batch, y_train_batch, scalar = mini_batch_svm.pre_process()     # pre process training data
    X_test_batch = scalar.transform(X_test_batch)                           # use same scalar to pre process validation data
    validation_set_batch = X_test_batch, y_test_batch                       # update validation set

    # train model for mini batch
    training_losses_batch, validation_losses_batch = mini_batch_svm.mini_batch_gradient_descent(X_train_batch, y_train_batch, batch_size, validation_set_batch)

    print("Mini Batch Gradient Descent Training ended...")
    print("Weights after mini batch gradient descent are: {}".format(mini_batch_svm.weights))
    
    print("Testing model accuracy for Mini Batch Gradient Descent...")

    # evaluate mini batch model using accuracy, precision, and recall metrics
    mini_batch_svm.predict(X_test_batch,y_test_batch)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # validation set for stochastic training
    validation_set_stoch = validation_set
    X_test_stoch, y_test_stoch = validation_set_stoch
    
    #intantiate the support vector machine class above
    stochastic_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    X_train_stoch, y_train_stoch,scalar = stochastic_svm.pre_process()          # pre process training data
    X_test_stoch = scalar.transform(X_test_stoch)                               # use same scalar to pre process validation data
    validation_set = X_test_stoch, y_test_stoch                                 # update validation set

    # train model for stochastic 
    training_losses_stoch, validation_losses_stoch = stochastic_svm.stochastic_gradient_descent(X_train_stoch, y_train_stoch, validation_set_stoch)

    print("Stochastic Gradient Descent Training ended...")
    print("Weights after stochastic gradient descent are: {}".format(stochastic_svm.weights))
    
    print("Testing model accuracy for Stochastic Gradient Descent...")
    
    # evaluate stochastic model using accuracy, precision, and recall metrics
    stochastic_svm.predict(X_test_stoch,y_test_stoch)

    # plot data
    plot_train_vs_val(training_losses_batch, validation_losses_batch, "Mini Batch Gradient Descent")                # mini batch visualization plot
    plot_train_vs_val(training_losses_stoch, validation_losses_stoch, "Stochastic Gradient Descent")                # stochastic visualization plot
    
    # plot all 4 training losses on 1 plot
    plt.plot(range(len(training_losses_batch)), training_losses_batch, label="Mini Batch Training Loss")
    plt.plot(range(len(validation_losses_batch)), validation_losses_batch, label="Mini Batch Validation Loss")
    plt.plot(range(len(training_losses_stoch)), training_losses_stoch, label="Stochastic Training Loss")
    plt.plot(range(len(validation_losses_stoch)), validation_losses_stoch, label="Stochastic Validation Loss")
    plt.xlabel("Epoch (1/10th)")
    plt.ylabel("Loss")
    plt.title("Stochastic vs Mini Batch Losses vs Epochs")
    plt.legend()
    plt.show()
    
    return

def part_3(X_train,y_train,validation_set):
    print("PART III -----------------------------------------------------------------------------")
    # initialize model parameters
    C = 0.001                               # regularization parameter
    learning_rate = 0.001                   # step size for gradient ddescent
    epoch = 5000                            # number of training iterations
    sample_size = 10                        # initial size of samples
    batch_size = max(1,sample_size // 5)    # batch size for mini batch
    previous_loss = float('inf')            # initialize min loss to inf
    threshold = 0.001                       # threshold for finding satisfactory performance
    
    total_training_losses = []              # list of training lossses
    total_validation_losses = []            # list of validation losses
    total_samples = 0                       # initialize smaple count to 0
    
    X_test, y_test = validation_set         # validation input feature matrix and output labels
    
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    X_train, y_train, scalar = my_svm.pre_process()     # pre procces training data
    X_test = scalar.transform(X_test)                   # use same scalar to pre process validation data
    validation_set = X_test, y_test                     # update validation set
    
    # randomly select initial samples from training data
    initial_indices = np.random.choice(len(X_train), sample_size, replace=False)    
    X_initial = X_train[initial_indices]
    Y_initial = y_train[initial_indices]
    # remove those training samples from the remaining pool of samples
    X_train_remaining = np.delete(X_train, initial_indices, axis=0)
    y_train_remaining = np.delete(y_train, initial_indices, axis=0)
    total_samples += sample_size

    # train model using mini batch
    training_loss, validation_loss = my_svm.mini_batch_gradient_descent(X_initial, Y_initial, batch_size, validation_set=(X_train_remaining, y_train_remaining))
    #training_loss, validation_loss = my_svm.stochastic_gradient_descent(X_initial, Y_initial, validation_set=(X_train_remaining, y_train_remaining))
    total_training_losses.extend(training_loss)         # update training losses
    total_validation_losses.extend(validation_loss)     # update validation losses
    
    # active learning loop to add samples until convergence
    while abs(previous_loss - min(training_loss)) > threshold: 
        # select sample with highest uncertainty
        next_X, next_y = my_svm.sampling_strategy(X_train_remaining, y_train_remaining)
        
        # add selected sample to training set
        X_initial = np.vstack([X_initial, next_X])
        Y_initial = np.vstack([Y_initial, next_y])

        # remove selected sample from the remaining pool
        idx_to_remove = np.where((X_train_remaining == next_X).all(axis=1))[0][0]
        X_train_remaining = np.delete(X_train_remaining, idx_to_remove, axis=0)
        y_train_remaining = np.delete(y_train_remaining, idx_to_remove, axis=0)
    
        # train model again using mini batch
        training_loss, validation_loss = my_svm.mini_batch_gradient_descent(X_initial, Y_initial, batch_size, validation_set=(X_train_remaining, y_train_remaining))
        #training_loss, validation_loss = my_svm.stochastic_gradient_descent(X_initial, Y_initial, validation_set=(X_train_remaining, y_train_remaining))
        previous_loss = min(training_loss)

        # update loss lists
        total_training_losses.extend(training_loss)
        total_validation_losses.extend(validation_loss)
        total_samples += 1  # increment total samples
    
    print("The minimum number of samples used are:",total_samples)
    
    print("Testing model accuracy for Uncertainty Sampling Using Mini Batch Gradient Descent...")
    
    # evaluate stochastic model using accuracy, precision, and recall metrics
    my_svm.predict(X_test,y_test)
    
    # plot losses
    plot_train_vs_val(total_training_losses, total_validation_losses, "Uncertainty Sampling")

    return my_svm

# plot the traingin vs validation losses for repsective training methods
def plot_train_vs_val(training_losses, validation_losses, title):
    
    plt.plot(range(len(training_losses)), training_losses, label="Training Loss")           # training loss line
    plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")     # validation loss line
    plt.xlabel("Epoch (1/10th)")                                                            # x label
    plt.ylabel("Loss")                                                                      # y label
    plt.title("Training and Validation Loss Over Epochs for {}".format(title))              # plot title
    plt.legend()                                                                            # make plot legend
    plt.show()                                                                              # show plot

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Set random seed 
np.random.seed(42) 

#Load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv('data1.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#segregate inputs and targets
#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

#converting categorical variables to integers 
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

# part1
#my_svm = part_1(X_train, y_train, (X_test,y_test))

# part 2
#my_svm = part_2(X_train, y_train, (X_test,y_test))

# part 3
my_svm = part_3(X_train, y_train, (X_test,y_test))
