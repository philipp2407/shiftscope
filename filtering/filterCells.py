from shiftscope.helperfunctions import Colors
from shiftscope.morphs import get_mf
from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
from tqdm import tqdm
import os
import pickle
from shiftscope.cellfaceMultistreamSupport.cellface.processing.feature_extraction import MorphologicalFeatureExtraction, GlcmFeatureExtraction

"""
    contains functions to filter cells.
    Functions: 
    - 
"""

filter_options_lymphocytes = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [5.5, 15]}
filter_options_monocytes = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [7.5, 15]}
# filter options for eosinophils, neutrophils and basophils
filter_options_granulocytes = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [8.5, 15]}
filter_options_wbc_healthy = {'optical_height_max': [0.2, np.inf], 'circularity': [0.85, np.inf], 'equivalent_diameter': [5.5, 15]}
    
def check_within_intervals(value, intervals):
    # Ensure intervals is a list
    if not isinstance(intervals[0], (list, tuple)):
        intervals = [intervals]
    # Check if the value lies within any of the provided intervals
    return any(interval[0] <= value <= interval[1] for interval in intervals)

def filter_container_and_return_indices(cellPath, use_predefined_filter_options=None, filter_options=[]):
    morph = ['area', 'aspect_ratio', 'biconcavity', 'center_x', 'center_y', 'circularity','density','discocyte_error',
                  'equivalent_diameter','height','mass_center_shift','mass_center_x','mass_center_y','optical_height_max','optical_height_min', 
                  'optical_height_mean','optical_height_std','perimeter','radius_max','radius_min','radius_mean','radius_std','solidity','steepness','volume','width']
    glcm = ['contrast', 'correlation', 'dissimilarity', 'energy', 'entropy', 'homogeneity']
    filter_options_morph = {}
    filter_options_glcm = {}
    if filter_options != []:
        for key,val in filter_options.items():
            if key in morph:
                filter_options_morph[key] = val
            elif key in glcm:
                filter_options_glcm[key] = val
    print(filter_options_morph)
    print(filter_options_glcm)
    
    delayed_features = []  # Initialize an empty list to store delayed objects
    morph_feature_extractor = MorphologicalFeatureExtraction()
    delayed_features_glcm = []
    glcm_feature_extractor = GlcmFeatureExtraction()
    if filter_options == []:
        print(f"{Colors.RED}Using predefined filter options for {use_predefined_filter_options}{Colors.RESET}")
        if use_predefined_filter_options == "lym":
            filter_options = filter_options_lymphocytes
        elif use_predefined_filter_options == "mon":
            filter_options = filter_options_monocytes
        elif use_predefined_filter_options == "eos" or use_predefined_filter_options == "neu" or use_predefined_filter_options == "bas":
            filter_options = filter_options_granulocytes
        elif use_predefined_filter_options == "wbc_healthy":
            filter_options = filter_options_wbc_healthy
    print(f"Filtering container in path {Colors.BLUE}{cellPath}{Colors.RESET} \n with filter options {Colors.RED}{filter_options}{Colors.RESET}")
    with Container(cellPath, 'r') as seg:
        phase_images = np.array(seg.content.phase.images, dtype=np.float32)
        masks = np.array(seg.content.mask.images, dtype=np.uint8)
        # Filter only valid image-mask pairs
        # valid_pairs = [(img, msk) for img, msk in zip(phase_images, masks) if img.shape == (48, 48) and msk.shape == (48, 48)]
        # delayed_results = [delayed(get_mf)(image, mask) for image, mask in valid_pairs]
        for img, msk in zip(phase_images, masks):
            delayed_feature = delayed(morph_feature_extractor.calculate_features_single_cell)(img, msk)
            delayed_features.append(delayed_feature)
        with ProgressBar():
            results = dask.compute(*delayed_features)
        # store results in numpy array
        morphological_features = np.concatenate(results)
        # glcm 
        for img, msk in zip(phase_images, masks):
            delayed_feature_glcm = delayed(glcm_feature_extractor.calculate_features_single_cell)(img, msk)
            delayed_features_glcm.append(delayed_feature_glcm)
        with ProgressBar():
            results_glcm = dask.compute(*delayed_features_glcm)
        glcm_features = np.concatenate(results_glcm)
        # end glcm
        
        filtered_indices = []

        for i, feature in enumerate(tqdm(morphological_features, desc=f"Filtering dataset")): 
            if all(check_within_intervals(feature[key], filter_options[key]) for key in filter_options_morph):
                filtered_indices.append(i)
        filtered_indices_glcm = []
        
        for i, feature in enumerate(tqdm(glcm_features, desc=f"Filtering dataset glcm")): 
            if all(check_within_intervals(feature[key], filter_options[key]) for key in filter_options_glcm):
                filtered_indices_glcm.append(i)
        
        filter_indices = [i for i in filtered_indices if i in filtered_indices_glcm]
        return filter_indices
    
def filter_container_and_save_indices_to_file_and_return_indices_and_path(cellPath, use_predefined_filter_options, filter_options=[], path_to_save_filter_options_to="", slicer=0): #file_name_for_filter_indices=""
    delayed_features = []  # Initialize an empty list to store delayed objects
    morph_feature_extractor = MorphologicalFeatureExtraction()
    if filter_options == []:
        print(f"{Colors.RED}Using predefined filter options for {use_predefined_filter_options}{Colors.RESET}")
        if use_predefined_filter_options == "lym":
            filter_options = filter_options_lymphocytes
        elif use_predefined_filter_options == "mon":
            filter_options = filter_options_monocytes
        elif use_predefined_filter_options == "eos" or use_predefined_filter_options == "neu" or use_predefined_filter_options == "bas":
            filter_options = filter_options_granulocytes
        elif use_predefined_filter_options == "wbc_healthy":
            filter_options = filter_options_wbc_healthy
            print("YAY")
    if path_to_save_filter_options_to == "":
        print(f"{Colors.RED}You didn't specify a path to save the filter indices to - please specify one now: (Please make sure its a valid path!){Colors.RESET}")
        userInput = input()
        path_to_save_filter_options_to = userInput
    #if file_name_for_filter_indices =="":
     #   print(f"{Colors.RED}You didnt't specify a file name for the filter indices. Please specify one now:{Colors.RESET}")
      #  userInput = input()
       # file_name_for_filter_indices = userInput
    print(f"Filtering container in path {Colors.BLUE}{cellPath}{Colors.RESET} with filter options {Colors.RED}{filter_options}{Colors.RESET}")
    #print(f"{Colors.GREEN}Will save filter indices to path {path_to_save_filter_options_to} with filename {file_name_for_filter_indices} (.pkl will be appended automatically){Colors.RESET}")
    with Container(cellPath, 'r') as seg:
        phase_images = np.array(seg.content.phase.images, dtype=np.float32)
        masks = np.array(seg.content.mask.images, dtype=np.uint8)
        # Filter only valid image-mask pairs
        for img, msk in zip(phase_images, masks):
            delayed_feature = delayed(morph_feature_extractor.calculate_features_single_cell)(img, msk)
            delayed_features.append(delayed_feature)
        with ProgressBar():
            results = dask.compute(*delayed_features)
        # store results in numpy array
        morphological_features = np.concatenate(results)
        filtered_indices = []

        for i, feature in enumerate(tqdm(morphological_features, desc=f"Filtering dataset")): 
            if all(check_within_intervals(feature[key], filter_options[key]) for key in filter_options):
                filtered_indices.append(i)
        print(f"length of filter_indices before slicing is {len(filtered_indices)}")
        if slicer != 0:
            print(f"Additionally slicing the indices to : :{slicer}")
            filtered_indices = filtered_indices[:slicer]
        # Save the filtered_indices list to a file
        file_path = path_to_save_filter_options_to #os.path.join(path_to_save_filter_options_to, f'{file_name_for_filter_indices}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(filtered_indices, f)
        return filtered_indices, file_path
    
    
    
