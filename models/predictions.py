from shiftscope.helperfunctions import Colors
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.nn.functional import softmax


# function to make prediction with a model and a dataloader
def predict(model, dataloader):
    print(f"Predicting with the provided trained model on the cells from the provided dataloader \n {Colors.BLUE}will return array of predicted percentages in form [lym%,mon%,eos%,neu%]{Colors.RESET}")
    device = "cpu"
    model.eval()
    # Store predictions and actual labels
    predictions = []
    actual_labels = []

    with torch.no_grad():  # Disable gradient computation
        # Store predictions and actual labels
        for data in dataloader:
            if len(data) == 2:
                # DataLoader returns inputs and labels
                inputs, labels = data
                labels = labels.to(device)
                actual_labels.extend(labels.cpu().numpy())  # Store actual labels
                inputs = inputs.to(device)  # Move the data to the CPU
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability as the prediction
                predictions.extend(predicted.cpu().numpy())  # Store predictions
            else:
                # DataLoader returns only inputs
                inputs = data.to(device)  # Move the data to the CPU
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability as the prediction
                predictions.extend(predicted.cpu().numpy())  # Store predictions

    if actual_labels:
        # Calculate true percentages from actual labels
        label_counts_true = Counter(actual_labels)
        total_labels = len(actual_labels)
        true_percentages = [label_counts_true[i] / total_labels * 100 for i in range(4)]
        print(f"True percentages are: {true_percentages}")
        # Calculate accuracy
        accuracy = accuracy_score(actual_labels, predictions)

        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average=None)

        # Generate a confusion matrix
        conf_matrix = confusion_matrix(actual_labels, predictions)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
    label_counts = Counter(predictions)

    # Calculate the percentage of each cell type
    total_predictions = len(predictions)
    predicted_percentages = [label_counts[i] / total_predictions * 100 for i in range(4)]
    if actual_labels:
        return accuracy
    return predicted_percentages


# function to make predictions... including DEBRIS class
def predictDEBRIS(model, dataloader):
    print(f"Predicting with the provided trained model on the cells from the provided dataloader \n {Colors.BLUE}will return array of predicted percentages in form [lym%,mon%,eos%,neu%,debris%]{Colors.RESET}")
    device = "cpu"
    model.eval()
    # Store predictions and actual labels
    predictions = []
    actual_labels = []
    with torch.no_grad():  # Disable gradient computation
        # Store predictions and actual labels
        for data in dataloader:
            if len(data) == 2:
                # DataLoader returns inputs and labels
                inputs, labels = data
                labels = labels.to(device)
                actual_labels.extend(labels.cpu().numpy())  # Store actual labels
                inputs = inputs.to(device)  # Move the data to the CPU
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability as the prediction
                predictions.extend(predicted.cpu().numpy())  # Store predictions
            else:
                # DataLoader returns only inputs
                inputs = data.to(device)  # Move the data to the CPU
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability as the prediction
                predictions.extend(predicted.cpu().numpy())  # Store predictions

    if actual_labels:
        # Calculate true percentages from actual labels
        label_counts_true = Counter(actual_labels)
        total_labels = len(actual_labels)
        true_percentages = [label_counts_true[i] / total_labels * 100 for i in range(5)]
        print(f"True percentages are: {true_percentages}")
        # Calculate accuracy
        accuracy = accuracy_score(actual_labels, predictions)

        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average=None)

        # Generate a confusion matrix
        conf_matrix = confusion_matrix(actual_labels, predictions)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
    # Count the occurrences of each label
    label_counts = Counter(predictions)
    lym_percentage = np.round(((label_counts[0] / (label_counts[0]+label_counts[1]+label_counts[2]+label_counts[3]))*100),2)
    mon_percentage = np.round(((label_counts[1] / (label_counts[0]+label_counts[1]+label_counts[2]+label_counts[3]))*100),2)
    eos_percentage = np.round(((label_counts[2] / (label_counts[0]+label_counts[1]+label_counts[2]+label_counts[3]))*100),2)
    neu_percentage = np.round(((label_counts[3] / (label_counts[0]+label_counts[1]+label_counts[2]+label_counts[3]))*100),2)
    debris_percentage = np.round(((label_counts[4] / (label_counts[0]+label_counts[1]+label_counts[2]+label_counts[3]+label_counts[4]))*100),2)
    predicted_percentages = [lym_percentage, mon_percentage, eos_percentage, neu_percentage, debris_percentage]
    predicted_labels = predictions
    return predicted_labels, predicted_percentages
