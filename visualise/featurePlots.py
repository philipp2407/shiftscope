import matplotlib.pyplot as plt
import seaborn as sns
import os
from shiftscope.helperfunctions import Colors
import numpy as np

# compare datasets with kde plots for a feature ...
def plot_KDE_to_compare_datasets(datasets, feature, labels, colors=None, savePlotPath="", savePlotFileName="", xAxisStart=0, xAxisEnd=1, legendLocation="upper right"):
    if not colors:  # If no colors are provided, generate them dynamically
        palette = sns.color_palette("hsv", len(datasets))
        colors = palette.as_hex()

    if savePlotPath=="":
        print("no path defined to save plot to - so not saving plot!")

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dynamically plot the distributions for each dataset
    for dataset, label, color in zip(datasets, labels, colors):
        kde_values = dataset[feature]
        kde_values_1D = kde_values.ravel()
        sns.kdeplot(kde_values_1D, ax=ax, fill=True, color=color, label=label)

    ax.set_xlabel(feature, fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    ax.set_xlim(xAxisStart, xAxisEnd)
    ax.legend(fontsize=20, loc=legendLocation)
    ax.tick_params(axis='x', labelsize=18)  # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=18)  # Adjust y-axis tick label font size
    plt.tight_layout()
    if savePlotPath!="":
        file_path = os.path.join(savePlotPath, f'{savePlotFileName}.pdf')
        plt.savefig(file_path)
        print(f"Saved {savePlotFileName} at path {savePlotPath}!")
    plt.show()

def plot_boxplot_to_compare_datasets(datasets, feature, labels, colors=None, savePlotPath="", savePlotFileName=""):
    # Prepare the data to plot
    data_to_plot = [dataset[feature].ravel() for dataset in datasets]
    
    plt.figure(figsize=(10, 6))
    
    # Creating the boxplot
    if colors is None:  # Generate colors dynamically if not provided
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
    
    bplot = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, widths=0.5)  # You can adjust widths as needed
    
    # Setting colors for each box, if specific colors were provided or generated
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel(f"{feature.capitalize()}", fontsize=27)
    plt.xticks(fontsize=27)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=27) 
    plt.grid()
    plt.tight_layout()
    
    if savePlotPath != "":
        file_path = os.path.join(savePlotPath, f'boxplot_{savePlotFileName}.pdf')
        plt.savefig(file_path)
    
    plt.show()
# Example usage:
# plot_boxplot_to_compare_datasets([all_features_dataset1, all_features_dataset2, all_features_dataset3], 'feature_name', ['Dataset 1', 'Dataset 2', 'Dataset 3'])


# OUTSOURCED
"""
# function to plot a KDE plot for a feature to compare two datasets
def plot_KDE_to_compare_two_datasets(all_features_source_data, all_features_target_data, feature, savePlotPath="", savePlotFileName="", xAxisStart=0, xAxisEnd=1, label_source_dataset="Source Dataset", label_target_dataset="Target Dataset", legendLocation=0):
    legLoc = ""
    if legendLocation == 0:
        legLoc = "upper right"
    else:
        legLoc = "upper left"
    if savePlotPath=="":
        print("no path defined to save plot to - so not saving plot!")
    kde_values_labeled = all_features_source_data[feature]
    kde_values_unlabeled = all_features_target_data[feature]
    kde_values_labeled_1D = kde_values_labeled.ravel()
    kde_values_unlabeled_1D = kde_values_unlabeled.ravel()

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the distributions
    sns.kdeplot(kde_values_labeled_1D, ax=ax, fill=True, color='skyblue', label=label_source_dataset)
    sns.kdeplot(kde_values_unlabeled_1D, ax=ax, fill=True, color='lightcoral', label=label_target_dataset)
    ax.set_xlabel("Diameter", fontsize=24)
    #ax.set_xlabel(f"{feature.capitalize()}", fontsize=24)
    ax.set_ylabel("Density", fontsize=24)
    ax.set_xlim(xAxisStart, xAxisEnd)
    ax.legend(fontsize=18,loc=legLoc)
    ax.tick_params(axis='x', labelsize=16)  # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=16)  # Adjust y-axis tick label font size
    plt.tight_layout()
    
    if savePlotPath!="":
        file_path = os.path.join(savePlotPath, f'{savePlotFileName}.pdf')
        plt.savefig(file_path)
        print(f"{Colors.GREEN}Saved {savePlotFileName} at path {savePlotPath}!{Colors.RESET}")
    plt.show()
    
# function to plot a KDE plot for a feature to compare three datasets
def plot_KDE_to_compare_three_datasets(all_features_1, all_features_2, all_features_3, feature, savePlotPath="", savePlotFileName="", xAxisStart=0, xAxisEnd=1, label_dataset_1="Dataset 1", label_dataset_2="Dataset 2",label_dataset_3="Dataset 3", legendLocation=0):
    legLoc = ""
    if legendLocation == 0:
        legLoc = "upper right"
    else:
        legLoc = "upper left"
    if savePlotPath=="":
        print("no path defined to save plot to - so not saving plot!")
    kde_values_1 = all_features_1[feature]
    kde_values_1_1D = kde_values_1.ravel()
    
    kde_values_2 = all_features_2[feature]
    kde_values_2_1D = kde_values_2.ravel()
    
    kde_values_3 = all_features_3[feature]
    kde_values_3_1D = kde_values_3.ravel()

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the distributions
    sns.kdeplot(kde_values_1_1D, ax=ax, fill=True, color='skyblue', label=label_dataset_1)
    sns.kdeplot(kde_values_2_1D, ax=ax, fill=True, color='lightcoral', label=label_dataset_2)
    sns.kdeplot(kde_values_3_1D, ax=ax, fill=True, color='salmon', label=label_dataset_3)
    ax.set_xlabel(feature, fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    ax.set_xlim(xAxisStart, xAxisEnd)
    ax.legend(fontsize=20,loc=legLoc)
    ax.tick_params(axis='x', labelsize=18)  # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=18)  # Adjust y-axis tick label font size
    plt.tight_layout()
    
    if savePlotPath!="":
        file_path = os.path.join(savePlotPath, f'{savePlotFileName}.pdf')
        plt.savefig(file_path)
        print(f"{Colors.GREEN}Saved {savePlotFileName} at path {savePlotPath}!{Colors.RESET}")
    plt.show()

# function to plot a KDE plot for a feature to compare three datasets
def plot_KDE_to_compare_four_datasets(all_features_1, all_features_2, all_features_3, all_features_4, feature, savePlotPath="", savePlotFileName="", xAxisStart=0, xAxisEnd=1, label_dataset_1="Dataset 1", label_dataset_2="Dataset 2",label_dataset_3="Dataset 3",label_dataset_4="Dataset 4", legendLocation=0):
    legLoc = ""
    if legendLocation == 0:
        legLoc = "upper right"
    else:
        legLoc = "upper left"
    if savePlotPath=="":
        print("no path defined to save plot to - so not saving plot!")
    kde_values_1 = all_features_1[feature]
    kde_values_1_1D = kde_values_1.ravel()
    
    kde_values_2 = all_features_2[feature]
    kde_values_2_1D = kde_values_2.ravel()
    
    kde_values_3 = all_features_3[feature]
    kde_values_3_1D = kde_values_3.ravel()

    kde_values_4 = all_features_4[feature]
    kde_values_4_1D = kde_values_4.ravel()

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the distributions
    sns.kdeplot(kde_values_1_1D, ax=ax, fill=True, color='skyblue', label=label_dataset_1)
    sns.kdeplot(kde_values_2_1D, ax=ax, fill=True, color='purple', label=label_dataset_2)
    sns.kdeplot(kde_values_3_1D, ax=ax, fill=True, color='salmon', label=label_dataset_3)
    sns.kdeplot(kde_values_4_1D, ax=ax, fill=True, color='green', label=label_dataset_4)
    ax.set_xlabel("Diameter", fontsize=24)
    #ax.set_xlabel(feature, fontsize=20)
    ax.set_ylabel("Density", fontsize=24)
    ax.set_xlim(xAxisStart, xAxisEnd)
    ax.legend(fontsize=16,loc=legLoc)
    ax.tick_params(axis='x', labelsize=18)  # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=18)  # Adjust y-axis tick label font size
    plt.tight_layout()
    
    if savePlotPath!="":
        file_path = os.path.join(savePlotPath, f'{savePlotFileName}.pdf')
        plt.savefig(file_path)
        print(f"{Colors.GREEN}Saved {savePlotFileName} at path {savePlotPath}!{Colors.RESET}")
    plt.show()
    

# function to plot a boxplot for a feature to compare two datasets
def plot_boxplot_to_compare_two_datasets(all_features_source_data, all_features_target_data, feature, label_source_dataset="Source Dataset", label_target_dataset="Target Dataset", savePlotPath="", savePlotFileName=""):
    data_to_plot = [
        all_features_source_data[feature].ravel(),
        all_features_target_data[feature].ravel(),
    ]
    plt.figure(figsize=(10, 6))
    
    # Creating boxplot with specific colors and box widths
    bplot = plt.boxplot(data_to_plot, patch_artist=True, labels=[label_source_dataset, label_target_dataset], widths=0.5)  # You can adjust widths as needed
    
    # Setting colors for each box
    colors = ['#3274A1', '#E1812C']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel(f"{feature.capitalize()}", fontsize=27)
    plt.xticks(fontsize=27)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=27) 
    plt.grid()
    plt.tight_layout()
    
    if savePlotPath != "":
        file_path = os.path.join(savePlotPath, f'boxplot_{savePlotFileName}.pdf')
        plt.savefig(file_path)
    
    plt.show()


    
# function to plot a boxplot for a feature to compare three datasets
def plot_boxplot_to_compare_three_datasets(all_features_1, all_features_2, all_features_3, feature, label_dataset_1="Dataset 3", label_dataset_2="Dataset 2", label_dataset_3="Dataset 3", savePlotPath="", savePlotFileName=""):
    data_to_plot = [
        all_features_1[feature].ravel(),
        all_features_2[feature].ravel(),
        all_features_3[feature].ravel(),
    ]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, patch_artist=True, labels=[label_dataset_1, label_dataset_2, label_dataset_3])
    plt.ylabel(feature, fontsize=20)
    plt.xticks(fontsize=18)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=18) 
    plt.tight_layout()
    if savePlotPath!= "":
        file_path = os.path.join(savePlotPath, f'boxplot_{savePlotFileName}.pdf')
        plt.savefig(file_path)
    plt.show()

"""