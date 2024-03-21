from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container, Seg
from shiftscope.helperfunctions import Colors
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# merge a bunch of seg containers together to one container which will be stored at "output_path" --> that path can then be used as training data for a 4-part-diff classifier
# filter_indices can be provided either directly (use "filter_indices_raw") or indirectly as paths (use "filter_indices_paths") (or not at all to use no filtering)
# the labels have to be provided in the list "cell_types" - where each entry corresponds to the same index in "input_paths" (or not if dont need labels)
# ---> so input_paths, cell_types and filter_indices have to fit together!!
# NOTE: this function only works for the cell types eos,lym,mon,neu 
def merge_seg_containers(output_path, input_paths, cell_types=[], filter_indices_paths = [], filter_indices_raw=[], channels_to_use=['phase','hologram','amplitude']):
    label_dtype = np.dtype([('lym', np.float32), ('mon', np.float32) , ('eos', np.float32), ('neu', np.float32)])
    numberOfPatches = 0
    print(f"{Colors.BLUE}Now starting to merge containers to one big .seg file using {channels_to_use}{Colors.RESET}")
    with Container(output_path, mode="w") as merged_container:
        merged_container.create_structure(seg_structure) # create the new merged container and merge all input containers together
    for index, path in enumerate(input_paths):
        filter_indices = []
        if filter_indices_paths and not filter_indices_raw:
            # using filter_indices from the paths provided
            with open(filter_indices_paths[index], "rb") as f:
                filter_indices = pickle.load(f)
        elif filter_indices_raw and not filter_indices_paths:
            # using the filter indices from filter_indices_raw
            filter_indices = filter_indices_raw
        if not filter_indices:
            print("Using no filter indices!")
        else: print("using filter indices")
        with Seg(input_paths[index], 'r') as segOld:
            with Seg(output_path, mode="r+") as segNew:
                if cell_types: name = f"{index:03d}_{cell_types[index]}_dataset{index}"
                else: name = f"{index:03d}_dataset{index}"
                # Create a manual progress bar
                with tqdm(total=2, desc=f"Added dataset {name} to output", ncols=100) as pbar:
                    # load masks
                    masks = segOld.content.mask.images[:, :, :]
                    if filter_indices:
                        masks = segOld.content.mask.images[filter_indices, :, :]
                    masks = np.array(masks, dtype=np.uint8)
                    segNew.content.mask.source.create_dataset(name, data=masks)
                    mask_list = []
                    mask_list.append(segNew.content.mask.source[name]) 
                    segNew.content.mask.images.append(mask_list)
                    numberOfPatches += len(masks)
                    for channel in channels_to_use:
                        channel_images = getattr(segOld.content, channel).images[:,:,:]
                        if filter_indices:
                            # if filter_indices is not empty
                            channel_images = getattr(segOld.content, channel).images[filter_indices,:,:]
                        channel_images = np.array(channel_images, dtype=np.float16)
                        getattr(segNew.content, channel).source.create_dataset(name, data=channel_images)
                        channel_list = []
                        channel_list.append(getattr(segNew.content, channel).source[name])
                        getattr(segNew.content, channel).images.append(channel_list)
                    pbar.update(1)  # Update progress
                    
                    if cell_types: # if cell_types is provided --> add labels
                        print("ADDING LABELS")
                        # add the labels
                        if filter_indices:
                            labels = np.zeros((len(filter_indices),), dtype=label_dtype)
                            cell_type = cell_types[index]
                            labels[cell_type] = 1
                            label_ds = segNew.content.label.celltype
                            current_label_size = label_ds.shape[0]  # Get the current size of the label dataset
                            new_label_size = current_label_size + len(filter_indices)
                            label_ds.resize((new_label_size,))  # Resize to accommodate new labels
                            label_ds[current_label_size:] = labels  # Append new labels
                            pbar.update(1)  # Update progress
                        else: 
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
                    else: print("not adding labels as no cell_types list is not provided")
    print("DONE")


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



### function to create a custom seg container with a specific number of cells per dataset and custom labels per dataset
def create_custom_seg_container_with_labels(output_path, input_files_paths, labels, numCellsPerPath, channels_to_use=['phase','hologram','amplitude'], filter_indices_paths=[]):
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
        filter_indices = []
        if filter_indices_paths:
            # using filter_indices from the paths provided
            with open(filter_indices_paths[index], "rb") as f:
                filter_indices = pickle.load(f)
                filter_indices = filter_indices[:numCellsPerPath]
        if not filter_indices:
            print("Using no filter indices!")
        else: print("using filter indices")
        with Seg(input_files_paths[index], 'r') as segOld:
            with Seg(output_path, mode="r+") as segNew:
                name = f"Dataset{index}"
                # Create a manual progress bar
                with tqdm(total=2, desc="Processing datasets", ncols=100) as pbar:
                    # load masks
                    masks = segOld.content.mask.images[:, :, :]
                    if filter_indices:
                        masks = segOld.content.mask.images[filter_indices, :, :]
                    else:
                        masks = segOld.content.mask.images[:numCellsPerPath, :, :]
                    masks = segOld.content.mask.images[:numCellsPerPath, :, :]
                    masks = np.array(masks, dtype=np.uint8)
                    segNew.content.mask.source.create_dataset(name, data=masks)
                    mask_list = []
                    mask_list.append(segNew.content.mask.source[name]) 
                    segNew.content.mask.images.append(mask_list)
                    pbar.update(1)  # Update progress
                    numberOfPatches += len(masks)
                    for channel in channels_to_use:
                        if filter_indices:
                            # if filter_indices is not empty
                            channel_images = getattr(segOld.content, channel).images[filter_indices,:,:]
                        else: 
                            channel_images = getattr(segOld.content, channel).images[:numCellsPerPath,:,:]
                        channel_images = np.array(channel_images, dtype=np.float16)
                        getattr(segNew.content, channel).source.create_dataset(name, data=channel_images)
                        channel_list = []
                        channel_list.append(getattr(segNew.content, channel).source[name])
                        getattr(segNew.content, channel).images.append(channel_list)
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
        
        
        
# function to merge a bunch of .seg containers with labels (for training data) together (including a debris class here) !!
def merge_seg_containers_with_DEBRIS(output_path, input_paths, cell_types, filter_indices_paths = [], filter_indices_raw=[], channels_to_use=['phase','hologram','amplitude']):
    print(f"{Colors.RED}Please make sure that your input paths, filter indices paths and cell_types fit together!! \n cell_types has to be in this form: cell_types = (['eos'] * 3 + ['lym'] * 1 +  ['mon'] * 1 + ['neu'] * 2 + ['debris'] * 1) \n {Colors.RESET}")
    
    print(f"{Colors.BLUE}Now starting to merge containers to one big .seg file using {channels_to_use}{Colors.RESET}")
    label_dtype = np.dtype([('lym', np.float32), ('mon', np.float32) , ('eos', np.float32), ('neu', np.float32), ('debris', np.float32)])
    numberOfPatches = 0
    # create the new merged container and merge all input containers together
    with Container(output_path, mode="w") as merged_container:
        merged_container.create_structure(seg_structure_with_DEBRIS)
    for index, path in enumerate(input_paths):
        filter_indices = []
        if filter_indices_paths and not filter_indices_raw:
            # using filter_indices from the paths provided
            with open(filter_indices_paths[index], "rb") as f:
                filter_indices = pickle.load(f)
        elif filter_indices_raw and not filter_indices_paths:
            # using the filter indices from filter_indices_raw
            filter_indices = filter_indices_raw
        if not filter_indices:
            print("Using no filter indices!")
        else: print("using filter indices")
        
        with Seg(input_paths[index], 'r') as segOld:
            with Seg(output_path, mode="r+") as segNew:
                name = f"{index:03d}_{cell_types[index]}_dataset{index}"
                # Create a manual progress bar
                with tqdm(total=2, desc=f"Added dataset {name} to output", ncols=100) as pbar:

                    # load masks
                    masks = segOld.content.mask.images[:, :, :]
                    if filter_indices:
                        masks = segOld.content.mask.images[filter_indices, :, :]
                    masks = np.array(masks, dtype=np.uint8)
                    segNew.content.mask.source.create_dataset(name, data=masks)
                    mask_list = []
                    mask_list.append(segNew.content.mask.source[name]) 
                    segNew.content.mask.images.append(mask_list)
                    
                    numberOfPatches += len(masks)
                    for channel in channels_to_use:
                        channel_images = getattr(segOld.content, channel).images[:,:,:]
                        if filter_indices:
                            # if filter_indices is not empty
                            channel_images = getattr(segOld.content, channel).images[filter_indices,:,:]
                        channel_images = np.array(channel_images, dtype=np.float16)
                        getattr(segNew.content, channel).source.create_dataset(name, data=channel_images)
                        channel_list = []
                        channel_list.append(getattr(segNew.content, channel).source[name])
                        getattr(segNew.content, channel).images.append(channel_list)
                    pbar.update(1)  # Update progress
                    if filter_indices:
                        labels = np.zeros((len(filter_indices),), dtype=label_dtype)
                        cell_type = cell_types[index]
                        labels[cell_type] = 1
                        label_ds = segNew.content.label.celltype
                        current_label_size = label_ds.shape[0]  # Get the current size of the label dataset
                        new_label_size = current_label_size + len(filter_indices)
                        label_ds.resize((new_label_size,))  # Resize to accommodate new labels
                        label_ds[current_label_size:] = labels  # Append new labels
                        pbar.update(1)  # Update progress
                    else: 
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
