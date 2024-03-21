from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container
from shiftscope.cellfaceMultistreamSupport.cellface.processing.feature_extraction import MorphologicalFeatureExtraction, GlcmFeatureExtraction
from shiftscope.helperfunctions import Colors
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
#from tqdm import tqdm
import os
import pickle


"""
    Filtering Cells. Has been updated to work with the new feature_extraction module.
"""

# pre-defined filtering options for the specific cell types
filter_options_lymphocytes = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [5.5, 15]}
filter_options_monocytes = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [7.5, 15]}
# filter options for eosinophils, neutrophils and basophils
filter_options_granulocytes = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [8.5, 15]}
filter_options_wbc_healthy = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [5.5, 15]}


# the morphological and glcm features
morph = ['area', 'aspect_ratio', 'biconcavity', 'center_x', 'center_y', 'circularity','density','discocyte_error',
              'equivalent_diameter','height','mass_center_shift','mass_center_x','mass_center_y','optical_height_max','optical_height_min', 
              'optical_height_mean','optical_height_std','perimeter','radius_max','radius_min','radius_mean','radius_std','solidity','steepness','volume','width']
glcm = ['contrast', 'correlation', 'dissimilarity', 'energy', 'entropy', 'homogeneity']

# function to check whether a value is wihin a provided interval
def check_within_intervals(value, intervals):
    # Ensure intervals is a list
    if not isinstance(intervals[0], (list, tuple)):
        intervals = [intervals]
    # Check if the value lies within any of the provided intervals
    return any(interval[0] <= value <= interval[1] for interval in intervals)

# function to filter a container (with the option to use pre defined filter options and saving the filter indices to a pickle file & slicing the indices)
def filter_container(cellPath, use_predefined_filter_options, filter_options=[], saveIndices = False, path_to_save_filter_options_to="",  slicer=0):
    if not filter_options:
        if use_predefined_filter_options == "lym":
            filter_options = filter_options_lymphocytes
        elif use_predefined_filter_options == "mon":
            filter_options = filter_options_monocytes
        elif use_predefined_filter_options == "eos" or use_predefined_filter_options == "neu" or use_predefined_filter_options == "bas":
            filter_options = filter_options_granulocytes
        elif use_predefined_filter_options == "wbc_healthy":
            filter_options = filter_options_wbc_healthy
    filter_options_morph = {}
    filter_options_glcm = {}
    for key,val in filter_options.items():
        if key in morph:
            filter_options_morph[key] = val
        elif key in glcm:
            filter_options_glcm[key] = val
    delayed_features = []  # Initialize an empty list to store delayed objects
    morph_feature_extractor = MorphologicalFeatureExtraction()
    delayed_features_glcm = []
    glcm_feature_extractor = GlcmFeatureExtraction()
    with Container(cellPath, 'r') as seg:
        phase_images = np.array(seg.content.phase.images, dtype=np.float32)
        masks = np.array(seg.content.mask.images, dtype=np.uint8)
        for img, msk in zip(phase_images, masks):
            delayed_feature = delayed(morph_feature_extractor.calculate_features_single_cell)(img, msk)
            delayed_features.append(delayed_feature)
        with ProgressBar():
            results = dask.compute(*delayed_features)
        # store results in numpy array
        morphological_features = np.concatenate(results)
        for img, msk in zip(phase_images, masks):
            delayed_feature_glcm = delayed(glcm_feature_extractor.calculate_features_single_cell)(img, msk)
            delayed_features_glcm.append(delayed_feature_glcm)
        with ProgressBar():
            results_glcm = dask.compute(*delayed_features_glcm)
        glcm_features = np.concatenate(results_glcm)
        filtered_indices = []
        for i, feature in enumerate(morphological_features): 
            if all(check_within_intervals(feature[key], filter_options[key]) for key in filter_options_morph):
                filtered_indices.append(i)
        filtered_indices_glcm = []
        for i, feature in enumerate(glcm_features): 
            if all(check_within_intervals(feature[key], filter_options[key]) for key in filter_options_glcm):
                filtered_indices_glcm.append(i)
        filter_indices = [i for i in filtered_indices if i in filtered_indices_glcm]
        if slicer != 0:
            filter_indices = filter_indices[:slicer]
        if saveIndices == True:
            with open(path_to_save_filter_options_to, 'wb') as f:
                pickle.dump(filter_indices, f)
            print(f"""----> Path {Colors.RED}{cellPath}{Colors.RESET}\n was filtered with filter options \n----> {Colors.BLUE}{filter_options}{Colors.RESET}\n----> the filter indices have been saved to {Colors.GREEN}{path_to_save_filter_options_to}{Colors.RESET}""")
            return filter_indices, path_to_save_filter_options_to
        else:
            print(f"""----> Path {Colors.RED}{cellPath}{Colors.RESET}\n was filtered with filter options \n----> {Colors.BLUE}{filter_options}{Colors.RESET}""")
            return filter_indices