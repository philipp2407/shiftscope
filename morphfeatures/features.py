from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container
from dask.diagnostics import ProgressBar
import pickle
from skimage.measure import label, regionprops
from skimage.transform import resize
from dask import compute, delayed
import dask
import numpy as np
from typing import List, Optional
from shiftscope.cellfaceMultistreamSupport.cellface.processing.feature_extraction import MorphologicalFeatureExtraction, GlcmFeatureExtraction

# function to calculate glcm features (updated version) ['contrast', 'correlation', 'dissimilarity', 'energy', 'entropy', 'homogeneity']
def calculate_glcm_features(paths: List[str], indices_paths: List[str] = None, concrete_indices: List[List[int]] = None):
    all_features = []
    total_cells = 0
    glcm_feature_extractor = GlcmFeatureExtraction(bin_width=0.1)
    delayed_results = []
    for index, path in enumerate(paths):
        with Container(path, 'r') as seg:
            if indices_paths is not None:
                print("Using the indices_paths from the path to the indices_paths file provided")
                with open(indices_paths[index], "rb") as f:
                    myindices = pickle.load(f)
                print(f"length of filtered indices is: {len(myindices)}")
                # Filter phase images and masks based on provided indices
                phase_images = np.array([seg.content.phase.images[idx] for idx in myindices], dtype=np.float32)
                masks = np.array([seg.content.mask.images[idx] for idx in myindices], dtype=np.uint8)    
            else:
                if concrete_indices != None:
                    print("Using the concrete indices provided")
                    myindices = concrete_indices[index]
                    phase_images = np.array([seg.content.phase.images[idx] for idx in myindices], dtype=np.float32)
                    masks = np.array([seg.content.mask.images[idx] for idx in myindices], dtype=np.uint8)
                else:
                    print("no indices used - using all images without filter indices")
                    # Use all images if no indices provided
                    phase_images = np.array(seg.content.phase.images[:], dtype=np.float32)
                    masks = np.array(seg.content.mask.images[:], dtype=np.uint8)

            num_cells = len(phase_images)
            for img, msk in zip(phase_images, masks):
                delayed_feature = delayed(glcm_feature_extractor.calculate_features_single_cell)(img, msk)
                delayed_results.append(delayed_feature)
            with ProgressBar():
                results = dask.compute(*delayed_results)
            total_cells += num_cells
            all_features.extend(results)
    # Concatenate all features into a single numpy array
    all_features = np.concatenate(all_features)
    features_dict = {}
    # Iterate over the feature names in the dtype of the numpy array
    for feature_name in all_features.dtype.names:
        # Extract the array for the current feature across all tuples
        features_dict[feature_name] = all_features[feature_name]
    return features_dict, total_cells


# function to calculate other morph features (updated version)

# function to calculate morph features ['area', 'aspect_ratio', 'biconcavity', 'center_x', 'center_y', 'circularity', 'density', 'discocyte_error',
# 'equivalent_diameter', 'height', 'mass_center_shift', 'mass_center_x', 'mass_center_y', 'optical_height_max', 'optical_height_min', 'optical_height_mean', 
# 'optical_height_std', 'perimeter', 'radius_max', 'radius_min', 'radius_mean', 'radius_std', 'solidity', 'steepness', 'volume', 'width']
def calculate_morph_features(paths: List[str], indices_paths: List[str] = [], concrete_indices: List[List[int]]=[]):
    all_features = []
    total_cells = 0
    morph_feature_extractor = MorphologicalFeatureExtraction()
    delayed_results = []
    for index, path in enumerate(paths):
        with Container(path, 'r') as seg:
            if indices_paths:
                print("Using the indices_paths from the path to the indices_paths file provided")
                with open(indices_paths[index], "rb") as f:
                    myindices = pickle.load(f)
                print(f"length of filtered indices is: {len(myindices)}")
                # Filter phase images and masks based on provided indices
                phase_images = np.array([seg.content.phase.images[idx] for idx in myindices], dtype=np.float32)
                masks = np.array([seg.content.mask.images[idx] for idx in myindices], dtype=np.uint8)    
            elif concrete_indices:
                print("Using the concrete indices provided")
                myindices = concrete_indices[index]
                phase_images = np.array([seg.content.phase.images[idx] for idx in myindices], dtype=np.float32)
                masks = np.array([seg.content.mask.images[idx] for idx in myindices], dtype=np.uint8)
            else:
                print("no indices used - using all images without filter indices")
                # Use all images if no indices provided
                phase_images = np.array(seg.content.phase.images[:], dtype=np.float32)
                masks = np.array(seg.content.mask.images[:], dtype=np.uint8)

            num_cells = len(phase_images)
            for img, msk in zip(phase_images, masks):
                delayed_feature = delayed(morph_feature_extractor.calculate_features_single_cell)(img, msk)
                delayed_results.append(delayed_feature)
            with ProgressBar():
                results = dask.compute(*delayed_results)
            total_cells += num_cells
            all_features.extend(results)
    # Concatenate all features into a single numpy array
    all_features = np.concatenate(all_features)
    features_dict = {}
    # Iterate over the feature names in the dtype of the numpy array
    for feature_name in all_features.dtype.names:
        # Extract the array for the current feature across all tuples
        features_dict[feature_name] = all_features[feature_name]
    #print(list(all_features.dtype.names))
    return features_dict, total_cells
