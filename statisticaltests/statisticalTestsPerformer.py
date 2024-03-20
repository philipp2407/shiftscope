import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import norm
import matplotlib.pyplot as plt
from shiftscope.helperfunctions import Colors
from prettytable import PrettyTable
from scipy.stats import f_oneway


def print_eco_profile_table(eco_profile_vector):
    table = PrettyTable()
    table.field_names = ["Feature", "Score Difference"]
    for feature, score in eco_profile_vector.items():
        # Round the score for nicer display
        rounded_score = round(score, 2)
        table.add_row([feature, rounded_score])
    print(table)


# Helper function to compute statistics
def calculate_stats_and_return_average_max_min_p25_p975_variance_median(values):
    values = np.round(values, 2)
    P2_5 = np.percentile(values, 2.5)
    P97_5 = np.percentile(values, 97.5)
    average = np.mean(values)
    maximum = np.max(values)
    minimum = np.min(values)
    variance = np.var(values)  # Added variance
    median = np.median(values)  # Added median
    return average, maximum, minimum, P2_5, P97_5, variance, median  


# Calculate ECO Vector
def calculate_ECO_vector(p_values_comparison, p_values_reference):
    print("Calculating ECO Profile Vector")
    #print("Your p-values have to be in this form:\n p_values_comparison = {'equivalent_diameter':1.3686126532481916e-16,'circularity': 9.338267411102963e-306,'optical_height_max': 0.0,}")
    # Calculate Z-scores for two-sided test
    z_scores_comparison = {feature: norm.ppf(1 - p_value / 2) for feature, p_value in p_values_comparison.items()}
    z_scores_reference = {feature: norm.ppf(1 - p_value / 2) for feature, p_value in p_values_reference.items()}

    # Calculate score differences
    score_differences = {
        feature: 100 * (norm.cdf(z_scores_comparison[feature]) - norm.cdf(z_scores_reference[feature]))
        for feature in p_values_comparison
    }
    #print(f"ECO-Profile Vector: {score_differences}")
    #print_eco_profile_table(score_differences)
    return score_differences
    

# runs the Mann Whitney U-Test comparing two datasets (needs morphological features as input) for a list of defined features and returns the p-values as a dictionary with features as key and p-value as values
def run_U_Test(all_features_X, all_features_Y, features = ['equivalent_diameter', 'aspect_ratio', 'circularity', 'solidity','contrast', 'dissimilarity', 'homogeneity', 'correlation', 'energy', 'entropy','mass_center_shift', 'optical_height_max', 'optical_height_min', 'optical_height_mean', 'optical_height_var', 'volume','radius_mean', 'radius_var','biconcavity', 'sphericity']):
    print(f"{Colors.BLUE}Running Mann-Whitney-U-Test to compare the two datasets using the provided morphological features and will return p_values as a dictionary with features as key and p-value as values{Colors.RESET}")
    if len(features) == 20:
        print("Using all features as you didn't provide a specific list of features")
    p_values = {}
    for feature in features:
        data_X = all_features_X[feature]
        data_Y = all_features_Y[feature]
        u_stat, p_val = mannwhitneyu(data_X, data_Y, alternative='two-sided')
        p_values[feature] = float(p_val)  # Store p-value in dictionary with feature as key
    return p_values


# run ANOVA comparing 3 datasets from their provided morphological features for the features contained in the provided features list
def run_ANOVA_Test(all_features_1, all_features_2, all_features_3, features):
    f_statistics = []
    p_values = []
    for feature in features:
        # Flatten the input arrays
        data_target_1 = all_features_target_data_1[feature].ravel()
        data_target_2 = all_features_target_data_2[feature].ravel()
        data_target_3 = all_features_target_data_3[feature].ravel()

        f_stat, p_val = f_oneway(data_target_1, data_target_2, data_target_3)

        # Append the scalar values of f_stat and p_val
        f_statistics.append(f_stat.item())  # Convert numpy scalar to Python scalar
        p_values.append(p_val.item())
    return f_statistics, p_values