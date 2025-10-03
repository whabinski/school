# Author: Wyatt Habinski
# Created: Nov 25, 2024
# License: MIT License
# Purpose: This python file includes 2 parts. 
    # Part 1 involves training an image classification model among the FASHION MINST dataset, using a convolutional neural network
    # Part 2 involves another classification model amoung the COMPAS dataset, using a logistic regression model and a bias mitigation strategy

# Usage: pyhton3 habinskw_part1_part2

# Dependencies: None
# Python Version: 3.12.7

# Modification History:
# - Version 1 

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor
import gzip

# -----------------------------------------------------------------------------------------------------------------------
# PART 1 ----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# Load and preprocess Fashion-MNIST dataset ------------------------------------------------------------------------------------------------------------

# function to load images from idx file
def load_images(file):
    with gzip.open(file, 'rb') as f:    # unzip compressed files with gzip
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28).copy()    #skip header (16 bytes) and reshape into 28x28
    return images

# function to load labels from idx file
def load_labels(file):
    with gzip.open(file, 'rb') as f:    # unzip compressed files with gzip
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)    # skip header (8 bytes)
    return labels

# function to load and preprocess fashion mnist dataset
def load_FASHION_MNIST(train_images_path, train_labels_path, test_images_path, test_labels_path, batch_size):  
    train_images = load_images(train_images_path)       # load training images
    train_labels = load_labels(train_labels_path)       # load training labels
    test_images = load_images(test_images_path)         # load test images
    test_labels = load_labels(test_labels_path)         # load test labels
    
    # define transformations to apply to images
    transform = Compose([
        ToTensor()          # convert images to pytorch tensors
    ])
    
    train_images_transformed_list = []  # initialize empty list of transformed train images
    test_images_transformed_list = []   # initialize empty list of transformed test images

    for image in train_images:
        train_images_transformed_list.append(transform(image))  # apply transformation and append to list

    for image in test_images:
        test_images_transformed_list.append(transform(image))  # apply transformation and append to list

    train_images_transformed = torch.stack(train_images_transformed_list)   # create torch stack of transformed training images
    test_images_transformed = torch.stack(test_images_transformed_list)     # create torch stack of transformed test images

    train_images, val_images, train_labels, val_labels = train_test_split(train_images_transformed, train_labels, test_size=0.2, random_state=42)   # split data into train and val sets
    
    train_dataset = list(zip(train_images, train_labels))                                   # combine train images and labels
    val_dataset = list(zip(val_images, val_labels))                                         # combine val images and labels
    test_dataset = list(zip(test_images_transformed, test_labels))                          # combine test images and labels
    
    batches = create_batches(train_dataset, val_dataset, test_dataset, batch_size)          # create batches for train, val, and test sets
    return batches

# function to create batches using dataloaders
def create_batches(train_dataset, val_dataset, test_dataset, batch_size):
    train_batches = DataLoader(train_dataset, batch_size=batch_size)        # create dataloader of respective batch size for train set
    val_batches = DataLoader(val_dataset, batch_size=batch_size)            # create dataloader of respective batch size for val set
    test_batches = DataLoader(test_dataset, batch_size=batch_size)          # create dataloader of respective batch size for test set
    return train_batches, val_batches, test_batches

# Convolutional Neural Network ------------------------------------------------------------------------------------------------------------

# convolutional neural network for image classification
class CNN(nn.Module):
    def __init__(self):
        
        super(CNN, self).__init__()     # initialize parent torch.nn.Module class
        
        # Convolutional Layers
        self.conv_10 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size=3)     # conv_10 layer with 1 input channel, 10 output channels, and a 3x3 kernel
        self.conv_5 = nn.Conv2d(in_channels = 10, out_channels = 5, kernel_size=3)      # conv_5 layer with 10 input channels, 5 output channels, and a 3x3 kernel
        self.conv_16 = nn.Conv2d(in_channels = 5, out_channels = 16, kernel_size=3)     # conv_16 layer with 5 input channels, 16 output channels, and a 3x3 kernel

        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # max pooling operation with 2x2 kernel 

        # Fully Connected Layers
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)  # fully connected layer 1 with flattened output from pooling (16 * 4 * 4) as input, 120 output units
        self.fc_2 = nn.Linear(120, 84)          # fully connected layer 2 with 120 inputs and 84 outputs
        self.fc_3 = nn.Linear(84, 10)           # fully connected layer 3 with 84 inputs and 10 outputs (10 classes)

    # forward pass of CNN
    def forward(self, x):
        x = F.relu(self.conv_10(x))     # apply Conv_10 and ReLU
        x = self.maxpool(x)             # apply Max Pooling
        x = F.relu(self.conv_5(x))      # apply Conv_5 and ReLU
        x = F.relu(self.conv_16(x))     # apply Conv_16 and ReLU
        x = self.maxpool(x)             # apply Max Pooling
        x = x.view(-1, 16 * 4 * 4)      # flatten tensor for Fully Connected Layers
        x = F.relu(self.fc_1(x))        # apply first Fully Connected Layer and ReLU
        x = F.relu(self.fc_2(x))        # apply second Fully Connected Layer and ReLU
        x = self.fc_3(x)                # apply third Fully Connected Layer
        
        return x

# Train, Evaluate, and Visualize ------------------------------------------------------------------------------------------------------------

# function to train the model using epochs, batches, gradient descent, and cross entropy loss
def train_model_cnn(model, train_batches, val_batches, cross_entropy_loss, sdg, epochs):

    training_losses = []        # initialize list to store training losses
    training_accuracies = []    # initialize list to store training accuracies
    validation_losses = []      # initialize list to store validation losses
    validation_accuracies = []  # initialize list to store validation accuracies
    
    for epoch in range(1, epochs + 1):                                # iterate through all epochs
        model.train()                                                 # set torch.nn model into train mode
        epoch_loss, correct_predictions, total_samples = 0, 0, 0      # initialize tracking variables

        for images, labels in train_batches:            # iterate through each batch's respective images and labels
            sdg.zero_grad()                             # initialize gradients to zero
            outputs = model(images)                     # call torch.nn forward pass
            loss = cross_entropy_loss(outputs, labels)  # compute loss using cross entropy loss
            loss.backward()                             # back propogation 
            sdg.step()                                  # stochastic gradient descent to calculate weights

            epoch_loss += loss.item()                               # cumulative total epoch loss by adding individual batch loss
            predictions = outputs.max(1)[1]                         # get predictions from logits
            correct_predictions += (predictions == labels).sum()    # count number of correct predictions
            total_samples += len(labels)                            # cumulative total number of samples by adding individual batch sizes

        training_accuracy = (correct_predictions.item() / total_samples) * 100      # calucalte epoch's training accuracy
        avg_training_loss = epoch_loss / len(train_batches)                         # average loss over all batches

        validation_loss, validation_accuracy = evaluate_model_cnn(model, val_batches, cross_entropy_loss)     # evaluate model using validation set
        
        training_losses.append(avg_training_loss)           # append each epochs training loss to list of losses
        training_accuracies.append(training_accuracy)       # append each epochs training accuracy to list of accuracies
        validation_losses.append(validation_loss)                  # append each epochs validation loss to list of losses
        validation_accuracies.append(validation_accuracy)          # append each epochs validation accuracy to list of accuracies

        # print training loss, training accuracy, validation loss, and validation accuracy for each epoch
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_training_loss:.4f}, Train Accuracy: {training_accuracy:.2f}%, Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.2f}%")

    return training_losses, training_accuracies, validation_losses, validation_accuracies

# function to evaluate trained model using a validation set
def evaluate_model_cnn(model, val_batches, cross_entropy_loss):
    
    model.eval()                                                        # set torch.nn model into eval mode
    validation_loss, correct_predictions, total_samples = 0, 0, 0       # initialize tracking variables

    for images, labels in val_batches:
        outputs = model(images)                                             # call torch.nn forward pass
        validation_loss += cross_entropy_loss(outputs, labels).item()       # compute loss using cross entropy loss
        
        predictions = outputs.max(1)[1]                                     # get predictions from logits
        correct_predictions += (predictions == labels).sum()                # count number of correct predictions
        total_samples += len(labels)                                        # cumulative total number of samples by adding individual batch sizes

    validation_accuracy = (correct_predictions.item() / total_samples) * 100    # calucalte epoch's training accuracy
    avg_validation_loss = validation_loss / len(val_batches)                    # # average loss over batch
    
    return avg_validation_loss, validation_accuracy

# Plots training vs validation losses and accuracies
def plot_loss_accuracy(training_losses, training_accuracies, validation_losses, validation_accuracies):
    epochs = range(1, len(training_losses) + 1)     # x-axis values are # of epochs
    plt.figure(figsize=(12, 5))                     # set plot size 12 units wide, 5 units tall

    # Plot loss for trainng vs validation sets
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label="Train Loss")           # plot training loss
    plt.plot(epochs, validation_losses, label="Validation Loss")    # plot validatioin loss
    plt.xlabel("Epochs")                                            # x-axis title
    plt.ylabel("Loss")                                              # y-axis title
    plt.title("Training vs Validation Loss Over Epochs")            # plot title
    plt.legend()                                                    # add legend

    # Plot accuracy for trainng vs validation sets
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, label="Train Accuracy")           # plot training accuracy
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")    # plot validation accuracy
    plt.xlabel("Epochs")                                                    # x-axis title
    plt.ylabel("Accuracy (%)")                                              # y-axis title
    plt.title("Training vs Validation Accuracy Over Epochs")                # plot title
    plt.legend()                                                            # add legend

    plt.tight_layout()      # adjust layout appropriately
    plt.show()              # display plots

# -----------------------------------------------------------------------------------------------------------------------
# PART 2 ----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# Load and preprocess compas dataset ------------------------------------------------------------------------------------------------------------

# function to load COMPAS scores dataset
def load_compas(file_path):
    data = pd.read_csv(file_path)   # read and load the dataset
    return data

# function to assign labels to dataset
def assign_labels(data):
    data['label'] = (data['score_text'] == 'High').astype(int)  # assign 1 if score_text = High, otherwise 0
    return data

# function to drop irrelevant columns
def drop_columns(data):
    irrelevant_cols = [
        'id', 'name', 'first', 'last', 'compas_screening_date',
        'score_text', 'decile_score', 'c_case_number', 'c_offense_date', 'c_arrest_date',
        'c_jail_in', 'c_jail_out', 'r_case_number', 'r_offense_date', 'r_charge_desc',
        'r_jail_in', 'r_jail_out', 'is_recid', 'is_violent_recid', 'num_r_cases',
        'num_vr_cases', 'vr_case_number', 'vr_offense_date', 'vr_charge_desc', 
        'type_of_assessment', 'v_decile_score', 'v_score_text', 'screening_date'
    ]
    data = data.drop(columns=irrelevant_cols)   # drop columns from dataset
    return data

# function to preprocess Features
def preprocess_features_lr(train_features, test_features, train_labels, test_labels):
    
    # handle categorical columns
    categorical_columns = ['sex', 'race', 'age_cat', 'c_charge_degree']                             # categorical columns to be processed
    # replace NaN values
    for col in categorical_columns:
            mode_value_train = train_features[col].mode()[0]                                        # calculate mode of the training column
            train_features[col] = train_features[col].fillna(mode_value_train)                      # replace NaN with mode for train features
            mode_value_test = test_features[col].mode()[0]                                          # calculate mode of the test column
            test_features[col] = test_features[col].fillna(mode_value_test)                         # replace NaN with mode for test features
    # one hot encoding
    encoder = OneHotEncoder()                                                                       # initialize sklearns one hot encoder
    train_features_encoded = encoder.fit_transform(train_features[categorical_columns]).toarray()   # fit and apply encoder to training set
    test_features_encoded = encoder.transform(test_features[categorical_columns]).toarray()         # apply encoder to testing set

    # handle numerical columns
    continuous_columns = ['age', 'juv_fel_count', 'priors_count', 'days_b_screening_arrest']        # numerical columns to be processed
    # replace NaN values
    for col in continuous_columns:
        mean_value_train = train_features[col].mean()                                               # calculate mean value of the train features
        train_features[col] = train_features[col].fillna(mean_value_train)                          # replace NaN with mean for train features
        mean_value_test = test_features[col].mean()                                                 # calculate mean value of the test features
        test_features[col] = test_features[col].fillna(mean_value_test)                             # replace NaN with mean for test features
    # standardize numerical columns
    scaler = StandardScaler()                                                                       # initialize sklearns standard scalar
    train_features_sclaed = scaler.fit_transform(train_features[continuous_columns])                # fit and apply scalar to training set
    test_features_scaled = scaler.transform(test_features[continuous_columns])                      # apply scalar to testing set

    # recombine categorical and numerical columns
    train_features_processed = np.hstack((train_features_encoded, train_features_sclaed))           # combine processed categorical and numerical training set columns
    test_features_processed = np.hstack((test_features_encoded, test_features_scaled))              # combine processed categorical and numerical testing set columns

    # convert to tensors
    train_features_tensor = torch.tensor(train_features_processed, dtype=torch.float32)             # convert training set to tensor
    test_features_tensor = torch.tensor(test_features_processed, dtype=torch.float32)               # convert testing seet to tensor
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.float32)                           
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.float32)  

    return train_features_tensor, test_features_tensor, train_labels_tensor, test_labels_tensor

# function to limit samples per race to that of the one with the smallest class
def balance_dataset(train_features, train_labels):

    data = train_features.copy()            # copy training features data for manipulation
    data['label'] = train_labels            # copy training labels for manipulation
    
    unique_races = data['race'].unique()    # list of all unique races
    
    balanced_data = []                      # initialize list for new balanced dataset

    for race in unique_races:               # iterate through each race
        race_data = data[data['race'] == race]                              # create boolean filter array for samples belonging to current race
        class_0_samples = race_data[race_data['label'] == 0]                # class 0 samples for the respective race
        class_1_samples = race_data[race_data['label'] == 1]                # class 1 samples for the respective race
        
        min_class_size = min(len(class_0_samples), len(class_1_samples))    # number of samples belonging to the smallest class
        
        class_0_balanced = class_0_samples.sample(n=min_class_size)         # reduce calss 0 to number of samples
        class_1_balanced = class_1_samples.sample(n=min_class_size)         # reduce calss 1 to number of samples
        
        balanced_data.append(pd.concat([class_0_balanced, class_1_balanced]))   # join class 0 and class 1 samples and add to new balanced dataset               
    
    balanced_data = pd.concat(balanced_data)                        # join all races together into 1 pandas dataset
    
    balanced_features = balanced_data.drop(columns=['label'])       # seperate features
    balanced_labels = balanced_data['label']                        # seperate labels
    
    return balanced_features, balanced_labels


# Logistic Regression Model ------------------------------------------------------------------------------------------------------------

# logistic regression class for target predictions
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        
        super(LogisticRegression, self).__init__()      # initialize parent torch.nn.Module class
        
        self.linear = nn.Linear(n_input_features, 1)    # linear layer

    # forward pass of linear regression
    def forward(self, x):
        return torch.sigmoid(self.linear(x))            # apply sigmoid function to linear layer

# Train and Evaluate ------------------------------------------------------------------------------------------------------------

# function to train the model using epochs, gradient descent, and binary cross entropy loss
def train_model_lr(model, train_features_tensor, train_labels_tensor, before_after, epochs):
    
    bce_loss = nn.BCELoss()                             # initialize binary cross entropy loss function
    sgd = torch.optim.SGD(model.parameters(), lr=0.1)   # initialize stochastic gradient descent method
    
    print(f"Training model {before_after} sampling strategy:")
    
    for epoch in range(1, epochs+1):                            # iterate through all epochs     
        model.train()                                           # set torch.nn model to training mode

        sgd.zero_grad()                                         # initialize gradients to zero
        predictions = model(train_features_tensor).squeeze()    # get predictions from logits - forward pass
        loss = bce_loss(predictions, train_labels_tensor)       # compute loss using binary cross entropy loss
        loss.backward()                                         # back propagation
        sgd.step()                                              # stochastic gradient descent to calculate weights

        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

# function to evaluate trained model using a test set
def evaluate_model_lr(model, test_features_tensor, test_labels_tensor, race, before_after):
    
    model.eval()                                                            # set model to evaluation mode
    predictions = model(test_features_tensor).squeeze()                     # forward pass for model predictions
    predictions = (predictions >= 0.5)                                      # convert probabilities < 0.5 to class 0, and >=0.5 to class 1
    accuracy = (predictions == test_labels_tensor).float().mean() * 100     # calculate accuracy of predictions
    print(f"Model test accuracy {before_after} sampling strategy: {accuracy:.2f}%")


    print(f"\nEqualized odds bias {before_after} sampling strategy:")
    race_values = race.values                           # race values in each sample of dataset
    unique_races = np.unique(race_values)               # list of all unique race values

    for race in unique_races:                                   # iterate through all unique races

        race_filter = race_values == race                       # create boolean filter array for samples belonging to current race
        race_predictions = predictions[race_filter]             # get all predictions for the current race
        race_true_labels = test_labels_tensor[race_filter]      # get all true labels for current race

        tp = ((race_true_labels == 1) & (race_predictions == 1)).sum()      # true positives
        fp = ((race_true_labels == 0) & (race_predictions == 1)).sum()      # false positives
        tn = ((race_true_labels == 0) & (race_predictions == 0)).sum()      # true negatives
        fn = ((race_true_labels == 1) & (race_predictions == 0)).sum()      # false negatives

        if (tp + fn) > 0:                   # check division by 0
            tpr = tp / (tp + fn) * 100      # tpr formula
        else:
            tpr = 0
        if (fp + tn) > 0:                   # check division by 0
            fpr = fp / (fp + tn) * 100      # fpr formula
        else:
            fpr = 0

        print(f"Race: {race}, tpr: {tpr:.2f}%, fpr: {fpr:.2f}%")
    print("\n")

# -----------------------------------------------------------------------------------------------------------------------
# Assignment ------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# function executes part 1 of assignment
def part_1():
    
    print("\nStart of Part 1 ----------------------------------------------------------\n")
    
    train_images_path = "./data/fashion/train-images-idx3-ubyte.gz"          # path for fashion-MNIST training images
    train_labels_path = "./data/fashion/train-labels-idx1-ubyte.gz"          # path for fashion-MNIST training labels
    test_images_path = "./data/fashion/t10k-images-idx3-ubyte.gz"            # path for fashion-MNIST test images
    test_labels_path = "./data/fashion/t10k-labels-idx1-ubyte.gz"            # path for fashion-MNIST test labels

    # load dataset, preprocess data, create batches, and training/val splits
    train_batches, val_batches, test_batches = load_FASHION_MNIST(train_images_path, train_labels_path, test_images_path, test_labels_path, batch_size=50)

    model = CNN()                                       # initialization cnn model
    cross_entropy_loss = nn.CrossEntropyLoss()          # initialize cross entropy loss function 
    sdg = torch.optim.SGD(model.parameters(), lr=0.1)   # initialize stochastic gradient descent method

    # train model and collect evaluation data
    training_losses, training_accuracies, validation_losses, validation_accuracies = train_model_cnn(model, train_batches, val_batches, cross_entropy_loss, sdg, epochs=15)
    
    # plot training vs validatoin scores
    plot_loss_accuracy(training_losses, training_accuracies, validation_losses, validation_accuracies)

    # evaluate trained model on test set
    test_loss, test_accuracy = evaluate_model_cnn(model, test_batches, cross_entropy_loss)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    
# function executes part 2 of assignment
def part_2():
    
    print("\nStart of Part 2 ----------------------------------------------------------\n")
    
    compas_path = "./compas-scores.csv"         # compas-scores.csv file path

    data = load_compas(compas_path)             # load compas dataset
    data = assign_labels(data)                  # assign recidivism labels
    data = drop_columns(data)                   # drop irrelevant columns

    features = data.drop(columns=['label'])     # select feature columns
    target = data['label']                      # select target column
    # split dataset into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, target, test_size=0.2, random_state=42)

    # ---------------------------------
    # Before balanced sampling
    # ---------------------------------

    # preprocess features, create tensors
    train_features_tensor, test_features_tensor, train_labels_tensor, test_labels_tensor = preprocess_features_lr(train_features, test_features, train_labels, test_labels)
    
    # train the model
    num_features = train_features_tensor.shape[1]           # number of feature columns
    model_unbalanced = LogisticRegression(num_features)     # initialize unbalanced linear regression model
    train_model_lr(model_unbalanced, train_features_tensor, train_labels_tensor, 'before', epochs=50)  # train model with parameters
    
    # evaluate balanced linear regression model accuracy and bias scores 
    evaluate_model_lr(model_unbalanced, test_features_tensor, test_labels_tensor, test_features['race'], 'before')
    
    # ---------------------------------
    # After balanced sampling
    # ---------------------------------
    
    # balance the dataset
    balanced_train_features, balanced_train_labels = balance_dataset(train_features, train_labels)

    # preprocess balanced features, create balanced tensors
    balanced_train_features_tensor, test_features_tensor, balanced_train_labels_tensor, test_labels_tensor = preprocess_features_lr(balanced_train_features, test_features, balanced_train_labels, test_labels)

    # retrain the model with the balanced dataset
    model_balanced = LogisticRegression(num_features)   # initialize balanced linear regression model
    train_model_lr(model_balanced, balanced_train_features_tensor, balanced_train_labels_tensor, 'after', epochs=50)   # train model with parameters
    
    # evaluate balanced linear regression model accuracy and bias scores 
    evaluate_model_lr(model_balanced, test_features_tensor, test_labels_tensor, test_features['race'], 'after')

# Main ------------------------------------------------------------------------------------------------------------

np.random.seed(42) # initialize random seed

part_1() # call part1

part_2() # call part2
