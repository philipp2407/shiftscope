from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container, Seg
from shiftscope.helperfunctions import Colors



# function to merge a bunch of .seg containers with labels (for training data) together!! paths to filter indices have to be provided!!!
def merge_seg_containers_with_labels(output_path, input_paths, cell_types, filter_indices_paths = [], filter_indices_raw=[], channels_to_use=['phase','hologram','amplitude']):
    print(f"{Colors.RED}Please make sure that your input paths, filter indices paths and cell_types fit together!! \n cell_types has to be in this form: cell_types = (['eos'] * 3 + ['lym'] * 1 +  ['mon'] * 1 + ['neu'] * 2) \n {Colors.RESET}")
    print(f"{Colors.BLUE}Now starting to merge containers to one big .seg file using {channels_to_use}{Colors.RESET}")
    label_dtype = np.dtype([('lym', np.float32), ('mon', np.float32) , ('eos', np.float32), ('neu', np.float32)])
    numberOfPatches = 0
    # create the new merged container and merge all input containers together
    with Container(output_path, mode="w") as merged_container:
        merged_container.create_structure(seg_structure)
        
    for index, path in enumerate(input_paths):
        
        if filter_indices_paths == [] and filter_indices_raw == []:
            print("no filter indices used")
            with Seg(input_paths[index], 'r') as segOld:
                with Seg(output_path, mode="r+") as segNew:
                    name = f"{index:03d}_{cell_types[index]}_dataset{index}"
                    # Create a manual progress bar
                    with tqdm(total=2, desc=f"Added dataset {name} to output", ncols=100) as pbar:

                        # load masks
                        masks = segOld.content.mask.images[:, :, :]
                        masks = np.array(masks, dtype=np.uint8)
                        segNew.content.mask.source.create_dataset(name, data=masks)
                        mask_list = []
                        mask_list.append(segNew.content.mask.source[name]) 
                        segNew.content.mask.images.append(mask_list)

                        # load hologram images
                        if 'hologram' in channels_to_use:
                            hologramImages = segOld.content.hologram.images[:, :, :]
                            hologramImages = np.array(hologramImages, dtype=np.float16)
                            segNew.content.hologram.source.create_dataset(name, data=hologramImages)
                            hologram_list = []
                            hologram_list.append(segNew.content.hologram.source[name])
                            segNew.content.hologram.images.append(hologram_list)

                        # load phase images
                        if 'phase' in channels_to_use:
                            phaseImages = segOld.content.phase.images[:, :, :]
                            phaseImages = np.array(phaseImages, dtype=np.float16)
                            segNew.content.phase.source.create_dataset(name, data=phaseImages)
                            phase_list = []
                            phase_list.append(segNew.content.phase.source[name])
                            segNew.content.phase.images.append(phase_list)

                        # load amplitude images 
                        if 'amplitude' in channels_to_use:
                            amplitudeImages = segOld.content.amplitude.images[:, :, :]
                            amplitudeImages = np.array(amplitudeImages, dtype=np.float16)
                            segNew.content.amplitude.source.create_dataset(name, data=amplitudeImages)
                            amplitude_list = []
                            amplitude_list.append(segNew.content.amplitude.source[name])
                            segNew.content.amplitude.images.append(amplitude_list)

                        pbar.update(1)  # Update progress
                        numberOfPatches += len(masks)
                        # add the labels
                        # Assign labels dynamically based on the cell type for each path
                        labels = np.zeros((len(masks),), dtype=label_dtype)
                        cell_type = cell_types[index]
                        labels[cell_type] = 1
                        label_ds = segNew.content.label.celltype
                        current_label_size = label_ds.shape[0]  # Get the current size of the label dataset
                        new_label_size = current_label_size + len(masks)
                        label_ds.resize((new_label_size,))  # Resize to accommodate new labels
                        label_ds[current_label_size:] = labels  # Append new labels
                        pbar.update(1)  # Update progress
        
        else:
            if filter_indices_paths == []:
                filter_indices = filter_indices_raw[index]
            else:
                with open(filter_indices_paths[index], "rb") as f:
                    filter_indices = pickle.load(f)

            with Seg(input_paths[index], 'r') as segOld:
                with Seg(output_path, mode="r+") as segNew:
                    name = f"{index:03d}_{cell_types[index]}_dataset{index}"
                    # Create a manual progress bar
                    with tqdm(total=2, desc=f"Added dataset {name} to output", ncols=100) as pbar:

                        # load masks
                        masks = segOld.content.mask.images[filter_indices, :, :]
                        masks = np.array(masks, dtype=np.uint8)
                        segNew.content.mask.source.create_dataset(name, data=masks)
                        mask_list = []
                        mask_list.append(segNew.content.mask.source[name]) 
                        segNew.content.mask.images.append(mask_list)

                        # load hologram images
                        if 'hologram' in channels_to_use:
                            hologramImages = segOld.content.hologram.images[filter_indices, :, :]
                            hologramImages = np.array(hologramImages, dtype=np.float16)
                            segNew.content.hologram.source.create_dataset(name, data=hologramImages)
                            hologram_list = []
                            hologram_list.append(segNew.content.hologram.source[name])
                            segNew.content.hologram.images.append(hologram_list)

                        # load phase images
                        if 'phase' in channels_to_use:
                            phaseImages = segOld.content.phase.images[filter_indices, :, :]
                            phaseImages = np.array(phaseImages, dtype=np.float16)
                            segNew.content.phase.source.create_dataset(name, data=phaseImages)
                            phase_list = []
                            phase_list.append(segNew.content.phase.source[name])
                            segNew.content.phase.images.append(phase_list)

                        # load amplitude images 
                        if 'amplitude' in channels_to_use:
                            amplitudeImages = segOld.content.amplitude.images[filter_indices, :, :]
                            amplitudeImages = np.array(amplitudeImages, dtype=np.float16)
                            segNew.content.amplitude.source.create_dataset(name, data=amplitudeImages)
                            amplitude_list = []
                            amplitude_list.append(segNew.content.amplitude.source[name])
                            segNew.content.amplitude.images.append(amplitude_list)

                        pbar.update(1)  # Update progress
                        numberOfPatches += len(masks)
                        # add the labels
                        # Assign labels dynamically based on the cell type for each path
                        labels = np.zeros((len(filter_indices),), dtype=label_dtype)
                        cell_type = cell_types[index]
                        labels[cell_type] = 1
                        label_ds = segNew.content.label.celltype
                        current_label_size = label_ds.shape[0]  # Get the current size of the label dataset
                        new_label_size = current_label_size + len(filter_indices)
                        label_ds.resize((new_label_size,))  # Resize to accommodate new labels
                        label_ds[current_label_size:] = labels  # Append new labels
                        pbar.update(1)  # Update progress
    print("DONE")
    
    
def merge_seg_containers_without_labels(output_path, input_paths, filter_indices_paths=[], filter_indices= [], channels_to_use=['phase','hologram','amplitude']):
    print(f"{Colors.BLUE}Now starting to merge containers without labels to one big .seg file using {channels_to_use}{Colors.RESET}")
    numberOfPatches = 0
    # create the new merged container and merge all input containers together
    with Container(output_path, mode="w") as merged_container:
        merged_container.create_structure(seg_structure)
        #merged_container.tree(details=True)
        
    for index, path in enumerate(input_paths):
        #print(f"filter indices are: {filter_indices[index]}")
        if filter_indices_paths == []:
            current_indices = filter_indices[index]
            #print(f"{Colors.RED}The current indices are {current_indices}{Colors.RESET}")
            #filter_indices = filter_indices[index]
        else:
            with open(filter_indices_paths[index], "rb") as f:
                current_indices = pickle.load(f)
        with Seg(input_paths[index], 'r') as segOld:
            with Seg(output_path, mode="r+") as segNew:
                name = f"{index:03d}_dataset{index}"
                # Create a manual progress bar
                with tqdm(total=4, desc=f"Adding dataset {name} to output", ncols=100) as pbar:
                    # load masks
                    masks = segOld.content.mask.images[current_indices, :, :]
                    print("Appending mask of shape:", masks.shape)
                    masks = np.array(masks, dtype=np.uint8)
                    segNew.content.mask.source.create_dataset(name, data=masks)
                    mask_list = []
                    mask_list.append(segNew.content.mask.source[name]) 
                    segNew.content.mask.images.append(mask_list)
                    pbar.update(1)  # Update progress
                    # load hologram images
                    if 'hologram' in channels_to_use:
                        hologramImages = segOld.content.hologram.images[current_indices, :, :]
                        print("Appending hologram of shape:", hologramImages.shape)
                        hologramImages = np.array(hologramImages, dtype=np.float16)
                        segNew.content.hologram.source.create_dataset(name, data=hologramImages)
                        hologram_list = []
                        hologram_list.append(segNew.content.hologram.source[name])
                        segNew.content.hologram.images.append(hologram_list)
                    pbar.update(1)  # Update progress
                    # load phase images
                    if 'phase' in channels_to_use:
                        phaseImages = segOld.content.phase.images[current_indices, :, :]
                        print("Appending phase of shape:", phaseImages.shape)
                        phaseImages = np.array(phaseImages, dtype=np.float16)
                        segNew.content.phase.source.create_dataset(name, data=phaseImages)
                        phase_list = []
                        phase_list.append(segNew.content.phase.source[name])
                        segNew.content.phase.images.append(phase_list)
                    pbar.update(1)  # Update progress 
                    # load amplitude images 
                    if 'amplitude' in channels_to_use:
                        amplitudeImages = segOld.content.amplitude.images[current_indices, :, :]
                        print("Appending amplitude of shape:", amplitudeImages.shape)
                        amplitudeImages = np.array(amplitudeImages, dtype=np.float16)
                        segNew.content.amplitude.source.create_dataset(name, data=amplitudeImages)
                        amplitude_list = []
                        amplitude_list.append(segNew.content.amplitude.source[name])
                        segNew.content.amplitude.images.append(amplitude_list)
                    pbar.update(1)  # Update progress
                    numberOfPatches += len(masks)
    print("DONE")


# the Segmentation Container structure
seg_structure = {        
    "children": {
        "hologram": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "virtual": True,
                    "elementShape": "(192, 192)",
                    "baseType": "float16",
                    "statistics": {
                        "min": {},
                        "max": {},
                        "mean": {},
                        "std": {},
                        "median": {},
                        "background": {"slice": "[0:1000,:,:]"}
                    }
                },
                "statistics": {
                    "nodeType": "group",
                    "metaType": "statistics"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "phase": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "virtual": True,
                    "elementShape": "(48, 48)",
                    "baseType": "float16",   
                    "statistics": {
                        "min": {},
                        "max": {},
                        "mean": {},
                        "std": {},
                        "median": {},
                        "background": {"slice": "[0:1000,:,:]"}
                    }
                },
                "statistics": {
                    "nodeType": "group",
                    "metaType": "statistics"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "amplitude": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "virtual": True,
                    "elementShape": "(48, 48)",
                    "baseType": "float16",
                    "statistics": {
                        "min": {},
                        "max": {},
                        "mean": {},
                        "std": {},
                        "median": {},
                        "background": {"slice": "[0:1000,:,:]"}
                    }
                },
                "statistics": {
                    "nodeType": "group",
                    "metaType": "statistics"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "mask": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "baseType": "uint8",
                    "virtual": True,
                    "elementShape": "(48, 48)"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "label": {
            "nodeType": "group",
            "children": {
                "celltype": {
                    "nodeType": "dataset",
                    "elementShape": "(0,)",
                    "metaType": "table",
                    "columns": {
                        "lym": {
                            "baseType": "float32"
                        },
                        "mon": {
                            "baseType": "float32"
                        },
                        "eos": {
                            "baseType": "float32"
                        },
                        "neu": {
                            "baseType": "float32"
                        }
                    }
                }
            }
        },
        "features": {
            "nodeType": "group",
            "children": {
                "morphological_features": {
                    "nodeType": "dataset",
                    "metaType": "morphological_features"
                }
            }
        }
    }
}



def create_label_structure(labels):
    label_children = {}
    for label in labels:
        label_children[label] = {"baseType": "float32"}

    label_structure = {
        "nodeType": "group",
        "children": {
            "celltype": {
                "nodeType": "dataset",
                "elementShape": "(0,)",
                "metaType": "table",
                "columns": label_children
            }
        }
    }
    return label_structure


### function to create a custom seg container with a specific number of cells per dataset and specific labels per dataset
def create_custom_seg_container_with_labels(output_path, input_files_paths, labels, numCellsPerPath, channels_to_use, filter_indices_paths=[]):
    # create custom seg container structure
    label_structure = create_label_structure(labels)
    seg_structure = {
        "children": {
            "hologram": {
                "nodeType": "group",
                "children": {
                    "images": {
                        "nodeType": "dataset",
                        "virtual": True,
                        "elementShape": "(192, 192)",
                        "baseType": "float16",
                        "statistics": {
                            "min": {},
                            "max": {},
                            "mean": {},
                            "std": {},
                            "median": {},
                            "background": {"slice": "[0:1000,:,:]"}
                        }
                    },
                    "statistics": {
                        "nodeType": "group",
                        "metaType": "statistics"
                    },
                    "source": {
                        "nodeType": "group"
                    }
                }
            },
            "phase": {
                "nodeType": "group",
                "children": {
                    "images": {
                        "nodeType": "dataset",
                        "virtual": True,
                        "elementShape": "(48, 48)",
                        "baseType": "float16",   
                        "statistics": {
                            "min": {},
                            "max": {},
                            "mean": {},
                            "std": {},
                            "median": {},
                            "background": {"slice": "[0:1000,:,:]"}
                        }
                    },
                    "statistics": {
                        "nodeType": "group",
                        "metaType": "statistics"
                    },
                    "source": {
                        "nodeType": "group"
                    }
                }
            },
            "amplitude": {
                "nodeType": "group",
                "children": {
                    "images": {
                        "nodeType": "dataset",
                        "virtual": True,
                        "elementShape": "(48, 48)",
                        "baseType": "float16",
                        "statistics": {
                            "min": {},
                            "max": {},
                            "mean": {},
                            "std": {},
                            "median": {},
                            "background": {"slice": "[0:1000,:,:]"}
                        }
                    },
                    "statistics": {
                        "nodeType": "group",
                        "metaType": "statistics"
                    },
                    "source": {
                        "nodeType": "group"
                    }
                }
            },
            "mask": {
                "nodeType": "group",
                "children": {
                    "images": {
                        "nodeType": "dataset",
                        "baseType": "uint8",
                        "virtual": True,
                        "elementShape": "(48, 48)"
                    },
                    "source": {
                        "nodeType": "group"
                    }
                }
            },
            "label": label_structure,
            "features": {
                "nodeType": "group",
                "children": {
                    "morphological_features": {
                        "nodeType": "dataset",
                        "metaType": "morphological_features"
                    }
                }
            }
        }
    }
    
    # create the container structure
    with Container(output_path, mode="w") as seg:
        seg.create_structure(seg_structure)  
    numberOfPatches = 0
    
    # Map each dataset to a label
    dataset_label_mapping = {index: label for index, label in enumerate(labels)}
    
    for index, path in enumerate(input_files_paths):
        if filter_indices_paths == []:
            with Seg(input_files_paths[index], 'r') as segOld:
                with Seg(output_path, mode="r+") as segNew:
                    name = f"Dataset{index}"
                    # Create a manual progress bar
                    with tqdm(total=2, desc="Processing datasets", ncols=100) as pbar:
                        # load masks
                        masks = segOld.content.mask.images[:numCellsPerPath, :, :]
                        masks = np.array(masks, dtype=np.uint8)
                        segNew.content.mask.source.create_dataset(name, data=masks)
                        mask_list = []
                        mask_list.append(segNew.content.mask.source[name]) 
                        segNew.content.mask.images.append(mask_list)
                        pbar.update(1)  # Update progress
                        # load hologram images
                        if 'hologram' in channels_to_use:
                            hologramImages = segOld.content.hologram.images[:numCellsPerPath, :, :]
                            hologramImages = np.array(hologramImages, dtype=np.float16)
                            segNew.content.hologram.source.create_dataset(name, data=hologramImages)
                            hologram_list = []
                            hologram_list.append(segNew.content.hologram.source[name])
                            segNew.content.hologram.images.append(hologram_list)

                        # load phase images
                        if 'phase' in channels_to_use:
                            phaseImages = segOld.content.phase.images[:numCellsPerPath, :, :]
                            phaseImages = np.array(phaseImages, dtype=np.float16)
                            segNew.content.phase.source.create_dataset(name, data=phaseImages)
                            phase_list = []
                            phase_list.append(segNew.content.phase.source[name])
                            segNew.content.phase.images.append(phase_list)

                        # load amplitude images 
                        if 'amplitude' in channels_to_use:
                            amplitudeImages = segOld.content.amplitude.images[:numCellsPerPath, :, :]
                            amplitudeImages = np.array(amplitudeImages, dtype=np.float16)
                            segNew.content.amplitude.source.create_dataset(name, data=amplitudeImages)
                            amplitude_list = []
                            amplitude_list.append(segNew.content.amplitude.source[name])
                            segNew.content.amplitude.images.append(amplitude_list)

                        numberOfPatches += len(phaseImages)
                        pbar.update(1)  # Update progress
        else:
            with open(filter_indices_paths[index], "rb") as f:
                filter_indices = pickle.load(f)
                filter_indices = filter_indices[:numCellsPerPath]
            with Seg(input_files_paths[index], 'r') as segOld:
                with Seg(output_path, mode="r+") as segNew:
                    name = f"Dataset{index}"
                    # Create a manual progress bar
                    with tqdm(total=2, desc="Processing datasets", ncols=100) as pbar:
                        # load masks
                        masks = segOld.content.mask.images[filter_indices, :, :]
                        masks = np.array(masks, dtype=np.uint8)
                        segNew.content.mask.source.create_dataset(name, data=masks)
                        mask_list = []
                        mask_list.append(segNew.content.mask.source[name]) 
                        segNew.content.mask.images.append(mask_list)
                        pbar.update(1)  # Update progress
                        # load hologram images
                        if 'hologram' in channels_to_use:
                            hologramImages = segOld.content.hologram.images[filter_indices, :, :]
                            hologramImages = np.array(hologramImages, dtype=np.float16)
                            segNew.content.hologram.source.create_dataset(name, data=hologramImages)
                            hologram_list = []
                            hologram_list.append(segNew.content.hologram.source[name])
                            segNew.content.hologram.images.append(hologram_list)

                        # load phase images
                        if 'phase' in channels_to_use:
                            phaseImages = segOld.content.phase.images[filter_indices, :, :]
                            phaseImages = np.array(phaseImages, dtype=np.float16)
                            segNew.content.phase.source.create_dataset(name, data=phaseImages)
                            phase_list = []
                            phase_list.append(segNew.content.phase.source[name])
                            segNew.content.phase.images.append(phase_list)

                        # load amplitude images 
                        if 'amplitude' in channels_to_use:
                            amplitudeImages = segOld.content.amplitude.images[filter_indices, :, :]
                            amplitudeImages = np.array(amplitudeImages, dtype=np.float16)
                            segNew.content.amplitude.source.create_dataset(name, data=amplitudeImages)
                            amplitude_list = []
                            amplitude_list.append(segNew.content.amplitude.source[name])
                            segNew.content.amplitude.images.append(amplitude_list)

                        numberOfPatches += len(phaseImages)
                        pbar.update(1)  # Update progress



    # Define the data type for the structured array
    label_dtype_fields = [(label, np.float32) for label in labels]
    label_dtype = np.dtype(label_dtype_fields)

    # Initialize the labels array
    labels_array = np.zeros((numberOfPatches,), dtype=label_dtype)
    current_patch_index = 0

    for index, path in enumerate(input_files_paths):
        # Assign labels
        for i in range(numCellsPerPath):
            for label in labels:
                # Assign 1 to the current label and 0 to others
                labels_array[label][current_patch_index + i] = 1 if label == labels[index] else 0
        current_patch_index += numCellsPerPath

    # Add the labels to the container
    with Container(output_path, mode="r+") as seg:
        label_ds = seg.content.label.celltype
        label_ds.resize((numberOfPatches,))
        label_ds[...] = labels_array

        
        

# function to merge a bunch of .seg containers with labels (for training data) together!! paths to filter indices have to be provided!!!
def merge_seg_containers_with_labels_with_DEBRIS(output_path, input_paths, cell_types, filter_indices_paths = [], filter_indices_raw=[], channels_to_use=['phase','hologram','amplitude']):
    print(f"{Colors.RED}Please make sure that your input paths, filter indices paths and cell_types fit together!! \n cell_types has to be in this form: cell_types = (['eos'] * 3 + ['lym'] * 1 +  ['mon'] * 1 + ['neu'] * 2 + ['debris'] * 1) \n {Colors.RESET}")
    print(f"{Colors.BLUE}Now starting to merge containers to one big .seg file using {channels_to_use}{Colors.RESET}")
    label_dtype = np.dtype([('lym', np.float32), ('mon', np.float32) , ('eos', np.float32), ('neu', np.float32), ('debris', np.float32)])
    numberOfPatches = 0
    # create the new merged container and merge all input containers together
    with Container(output_path, mode="w") as merged_container:
        merged_container.create_structure(seg_structure_with_DEBRIS)
    for index, path in enumerate(input_paths):
        if filter_indices_paths == []:
            filter_indices = filter_indices_raw[index]
        else:
            with open(filter_indices_paths[index], "rb") as f:
                filter_indices = pickle.load(f)

        with Seg(input_paths[index], 'r') as segOld:
            with Seg(output_path, mode="r+") as segNew:
                name = f"{index:03d}_{cell_types[index]}_dataset{index}"
                # Create a manual progress bar
                with tqdm(total=2, desc=f"Added dataset {name} to output", ncols=100) as pbar:

                    # load masks
                    masks = segOld.content.mask.images[filter_indices, :, :]
                    masks = np.array(masks, dtype=np.uint8)
                    segNew.content.mask.source.create_dataset(name, data=masks)
                    mask_list = []
                    mask_list.append(segNew.content.mask.source[name]) 
                    segNew.content.mask.images.append(mask_list)

                    # load hologram images
                    if 'hologram' in channels_to_use:
                        hologramImages = segOld.content.hologram.images[filter_indices, :, :]
                        hologramImages = np.array(hologramImages, dtype=np.float16)
                        segNew.content.hologram.source.create_dataset(name, data=hologramImages)
                        hologram_list = []
                        hologram_list.append(segNew.content.hologram.source[name])
                        segNew.content.hologram.images.append(hologram_list)

                    # load phase images
                    if 'phase' in channels_to_use:
                        phaseImages = segOld.content.phase.images[filter_indices, :, :]
                        phaseImages = np.array(phaseImages, dtype=np.float16)
                        segNew.content.phase.source.create_dataset(name, data=phaseImages)
                        phase_list = []
                        phase_list.append(segNew.content.phase.source[name])
                        segNew.content.phase.images.append(phase_list)

                    # load amplitude images 
                    if 'amplitude' in channels_to_use:
                        amplitudeImages = segOld.content.amplitude.images[filter_indices, :, :]
                        amplitudeImages = np.array(amplitudeImages, dtype=np.float16)
                        segNew.content.amplitude.source.create_dataset(name, data=amplitudeImages)
                        amplitude_list = []
                        amplitude_list.append(segNew.content.amplitude.source[name])
                        segNew.content.amplitude.images.append(amplitude_list)

                    pbar.update(1)  # Update progress
                    numberOfPatches += len(masks)
                    # add the labels
                    # Assign labels dynamically based on the cell type for each path
                    labels = np.zeros((len(filter_indices),), dtype=label_dtype)
                    cell_type = cell_types[index]
                    labels[cell_type] = 1
                    label_ds = segNew.content.label.celltype
                    current_label_size = label_ds.shape[0]  # Get the current size of the label dataset
                    new_label_size = current_label_size + len(filter_indices)
                    label_ds.resize((new_label_size,))  # Resize to accommodate new labels
                    label_ds[current_label_size:] = labels  # Append new labels
                    pbar.update(1)  # Update progress
    print("DONE")
    
    
# the Segmentation Container structure with DEBRIS
seg_structure_with_DEBRIS = {        
    "children": {
        "hologram": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "virtual": True,
                    "elementShape": "(192, 192)",
                    "baseType": "float16",
                    "statistics": {
                        "min": {},
                        "max": {},
                        "mean": {},
                        "std": {},
                        "median": {},
                        "background": {"slice": "[0:1000,:,:]"}
                    }
                },
                "statistics": {
                    "nodeType": "group",
                    "metaType": "statistics"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "phase": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "virtual": True,
                    "elementShape": "(48, 48)",
                    "baseType": "float16",   
                    "statistics": {
                        "min": {},
                        "max": {},
                        "mean": {},
                        "std": {},
                        "median": {},
                        "background": {"slice": "[0:1000,:,:]"}
                    }
                },
                "statistics": {
                    "nodeType": "group",
                    "metaType": "statistics"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "amplitude": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "virtual": True,
                    "elementShape": "(48, 48)",
                    "baseType": "float16",
                    "statistics": {
                        "min": {},
                        "max": {},
                        "mean": {},
                        "std": {},
                        "median": {},
                        "background": {"slice": "[0:1000,:,:]"}
                    }
                },
                "statistics": {
                    "nodeType": "group",
                    "metaType": "statistics"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "mask": {
            "nodeType": "group",
            "children": {
                "images": {
                    "nodeType": "dataset",
                    "baseType": "uint8",
                    "virtual": True,
                    "elementShape": "(48, 48)"
                },
                "source": {
                    "nodeType": "group"
                }
            }
        },
        "label": {
            "nodeType": "group",
            "children": {
                "celltype": {
                    "nodeType": "dataset",
                    "elementShape": "(0,)",
                    "metaType": "table",
                    "columns": {
                        "lym": {
                            "baseType": "float32"
                        },
                        "mon": {
                            "baseType": "float32"
                        },
                        "eos": {
                            "baseType": "float32"
                        },
                        "neu": {
                            "baseType": "float32"
                        },
                        "debris": {
                            "baseType": "float32"
                        }
                    }
                }
            }
        },
        "features": {
            "nodeType": "group",
            "children": {
                "morphological_features": {
                    "nodeType": "dataset",
                    "metaType": "morphological_features"
                }
            }
        }
    }
}