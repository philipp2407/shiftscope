from shiftscope.helperfunctions import Colors
import warnings
warnings.filterwarnings("ignore")
from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1: INFO messages are not printed; 2: INFO and WARNING messages are not printed)
import shutil
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from numpy.lib.recfunctions import structured_to_unstructured
import gc
from tqdm import tqdm
import torch

"""
    Dataset Classes, creating tensors etc...
    UPDATES: 
    (1): integrated create_tensors_no_labels_no_filering() into create_tensors_no_labels() and changed sequence of input arguments (filter_indices can be None)
    (2): create_tensor(): removed cellNumber input parameter
    (3): create_tensor_forDDM(): removed cellNumber as parameter
    (4): create_tensor_DEBRIS(): removed cellNumber as parameter
"""

# helper function to delete files in a directory/folder
def delete_files_in_folder(folderPath):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path) 
    print(f"deleted all files in {folderPath}")

# Cellface Dataset Class - for labeled samples
class CellfaceDatasetWithLabels:
    def __init__(self, folder_path, num_files=4, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.num_files = num_files
        self.file_indices = []
        self.labels = []
        for i in range(0, self.num_files):
            labels_file = np.load(os.path.join(folder_path, f'labels_{i}.npy'))
            num_labels = len(labels_file)
            self.file_indices.extend([(i, j) for j in range(num_labels)])
            self.labels.extend(labels_file)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_idx, tensor_idx = self.file_indices[idx]
        # Load the specific tensor file when it's needed
        tensor_path = os.path.join(self.folder_path, f'tensor_{file_idx}.npy')
        tensor_file = np.load(tensor_path, mmap_mode='r')
        tensor = tensor_file[tensor_idx]
        # Check if the array is writable, and if not, make a copy
        if not tensor.flags.writeable:
            tensor = tensor.copy()
        # Rearrange dimensions from [H, W, C] to [C, H, W] for PyTorch
        assert tensor.ndim == 3 and tensor.shape[2] in [1, 3], f"Tensor shape mismatch: {tensor.shape}" ### added line
        tensor = np.transpose(tensor, (2, 0, 1))
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(tensor).float()
        label = self.labels[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

# Cellface Dataset Class - for not labeled samples
class CellfaceDatasetNoLabels:
    def __init__(self, folder_path, num_files=4, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.num_files = num_files
        self.file_indices = []
        # Instead of loading labels, just count the number of tensors in each file
        for i in range(0, self.num_files):
            tensor_file_path = os.path.join(folder_path, f'tensor_{i}.npy')
            num_tensors = np.load(tensor_file_path, mmap_mode='r').shape[0]
            self.file_indices.extend([(i, j) for j in range(num_tensors)])
        
    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx, tensor_idx = self.file_indices[idx]
        # Load the specific tensor file when it's needed
        tensor_path = os.path.join(self.folder_path, f'tensor_{file_idx}.npy')
        tensor_file = np.load(tensor_path, mmap_mode='r')
        tensor = tensor_file[tensor_idx]
        # Check if the array is writable, and if not, make a copy
        if not tensor.flags.writeable:
            tensor = tensor.copy()
        # Rearrange dimensions from [H, W, C] to [C, H, W] for PyTorch
        tensor = np.transpose(tensor, (2, 0, 1))
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(tensor).float()
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


# function to create tensors but no label files - e.g. for whc_healthy files!
# changed sequence of arguments and included old "create_tensors_no_labels_no_filering()"
def create_tensors_no_labels(path, channels, path_to_save_tensors_to, size_of_each_tensor, filtered_indices=None, shape=(48,48)):
    delete_files_in_folder(path_to_save_tensors_to)
    with Container(path, 'r') as seg:
        channel = []
        for i, c in enumerate(channels):
            images = getattr(seg.content, c).images
            if filtered_indices is not None:
                selected_images = images[filtered_indices]
                print(f"Creating tensors for {path} with the provided filtered indices of length {len(filtered_indices)}")
            else:
                print("No filtering used as no filter indices are provided")
                selected_images = images[:]
            channel.append(selected_images)
        nameIndex = 0
        for i in range(0, len(filtered_indices), size_of_each_tensor):
            tensor_list = []
            for c in channel:
                batch_c = c[i:i+size_of_each_tensor]
                batch_c = np.tile(batch_c[..., np.newaxis], 1)
                batch_c = tf.image.resize(batch_c, size=shape)
                batch_c = tf.squeeze(batch_c)
                tensor_list.append(batch_c)
            batch_tensor = np.stack(tensor_list, axis=-1)                                                                              
            tensor_filename = f'tensor_{nameIndex}.npy'
            np.save(os.path.join(path_to_save_tensors_to, tensor_filename), batch_tensor)
            # Clear memory
            del batch_tensor
            gc.collect()
            nameIndex = nameIndex + 1
    
    
# function to create a tensor from a path using a start and end id
# UPDATE: removed cellNumber input parameter!
def create_tensor(path, channels, shape, start_idx, end_idx):
    with Container(path, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        # Get indices of each cell type
        lym_idx = np.where(celltype == 'lym')[0] 
        mon_idx = np.where(celltype == 'mon')[0]
        eos_idx = np.where(celltype == 'eos')[0]
        neu_idx = np.where(celltype == 'neu')[0]
        # Select a subset of cells from each class
        selected_idx = np.concatenate([lym_idx[start_idx:end_idx],
                                       mon_idx[start_idx:end_idx],
                                       eos_idx[start_idx:end_idx],
                                       neu_idx[start_idx:end_idx]])
        # Extract the specified image channels for the selected cells
        channel = []
        for i, c in enumerate(channels):
            channel.append(getattr(seg.content, c).images[selected_idx])
        # Resize and stack the channels to create the tensor
        tensor_list = []
        for c in channel:
            c = np.tile(c[..., np.newaxis], 1)
            c = tf.image.resize(c, size=shape)
            c = tf.squeeze(c)
            tensor_list.append(c)
        # Map cell type labels to integer values
        cell_label_mapping = {'lym': 0, 'mon': 1, 'eos': 2, 'neu': 3}
        int_labels = [cell_label_mapping[label[0]] for label in celltype[selected_idx]]
        # Stack the channels and labels to create the final tensor and label arrays
        return np.stack(tensor_list, axis=-1), int_labels
    

# this function creates train and test tensors from one .seg file (with 80/20 split)
# no filtering is done in this class!
def create_train80_test20_tensors(path_to_seg_file, channels, training_tensor_path, testset_tensor_path, shape=(48,48)):
    print(f"{Colors.BLUE}Creating tensors from the provided seg file including those channels {channels}{Colors.BLUE}")
    print(f"{Colors.RED}Will resize all images to shape {shape} - Please adapt the shape parameter if you want to change this{Colors.BLUE}")
    delete_files_in_folder(training_tensor_path)
    delete_files_in_folder(testset_tensor_path)
    # first need to find out how many cells of each celltype we have!
    with Container(path_to_seg_file, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        # Get indices of each cell type
        lym_idx = np.where(celltype == 'lym')[0] 
        mon_idx = np.where(celltype == 'mon')[0]
        eos_idx = np.where(celltype == 'eos')[0]
        neu_idx = np.where(celltype == 'neu')[0]
        print(f"We have {len(lym_idx)} lymphocytes, {len(mon_idx)} monocytes, {len(eos_idx)} eosinophils and {len(neu_idx)} neutrophils in this dataset")
        # calculate of which cells we have the fewest
        minimum = min(len(lym_idx), len(mon_idx), len(eos_idx), len(neu_idx))
        numberOfCellsWeCanUse = minimum*4 
        print(f"We can maximally use {numberOfCellsWeCanUse} cells of this dataset if we want it to be equally balanced")
        numberOfCellsPerTensorWeCanUse = numberOfCellsWeCanUse / 16
        num_cells_per_test_tensor = round(numberOfCellsPerTensorWeCanUse * 0.2)
        num_cells_per_training_tensor = numberOfCellsPerTensorWeCanUse - num_cells_per_test_tensor 
        print(f"{Colors.BLUE}We use a 80/20 Train/Test split: Using {num_cells_per_training_tensor * 16} for training, {num_cells_per_test_tensor * 16} for testing{Colors.RESET}")
        for i in tqdm(range(16), desc=f"{Colors.GREEN}Creating tensors{Colors.RESET}"):

            # Training set
            start_idx_train = int(i * num_cells_per_training_tensor) // 4
            end_idx_train = int((i + 1) * num_cells_per_training_tensor) // 4
            tensorTrain, labelsTrain = create_tensor(path_to_seg_file, channels, shape, start_idx_train, end_idx_train)
            np.save(os.path.join(training_tensor_path, f'tensor_{i}.npy'), tensorTrain) #  saves the tensor to disk as a .npy file
            np.save(os.path.join(training_tensor_path, f'labels_{i}.npy'), labelsTrain) # saves the corresponding labels to disk as a .npy file
            del tensorTrain 
            del labelsTrain 
            gc.collect() 

            # Test set
            start_idx_val = int(16 * num_cells_per_training_tensor) // 4 + int(i * num_cells_per_test_tensor) // 4
            end_idx_val = int(16 * num_cells_per_training_tensor) // 4 + int((i + 1) * num_cells_per_test_tensor) // 4
            tensorVal, labelsVal = create_tensor(path_to_seg_file, channels, shape, start_idx_val, end_idx_val)
            np.save(os.path.join(testset_tensor_path, f'tensor_{i}.npy'), tensorVal)
            np.save(os.path.join(testset_tensor_path, f'labels_{i}.npy'), labelsVal)
            del tensorVal 
            del labelsVal 
            gc.collect() 



# this function creates train and val tensors from one .seg file (with 80/20 split)
# for a DDM (Domain Discriminative model) with custom labels!!!
# no filtering is done in this class!
def create_train80_val20_tensors_forDDM(path_to_seg_file, labels, channels, training_tensor_path, val_tensor_path, shape=(48,48)):
    print(f"{Colors.BLUE}Creating tensors from the provided seg file including those channels {channels}{Colors.BLUE}")
    print(f"{Colors.RED}Will resize all images to shape {shape} - Please adapt the shape parameter if you want to change this{Colors.BLUE}")
    delete_files_in_folder(training_tensor_path)
    delete_files_in_folder(val_tensor_path)
    # first need to find out how many cells of each celltype we have!
    with Container(path_to_seg_file, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        print(categories)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        # Get indices of each cell type
        label1_idx = np.where(celltype == labels[0])[0] 
        label2_idx = np.where(celltype == labels[1])[0]
        print(f"We have {len(label1_idx)} cells of label {labels[0]} and {len(label2_idx)} of label {labels[1]}")
        # calculate of which cells we have the fewest
        minimum = min(len(label1_idx), len(label1_idx))
        numberOfCellsWeCanUse = minimum*2 
        print(f"We can maximally use {numberOfCellsWeCanUse} cells of this dataset if we want it to be equally balanced")
        numberOfCellsPerTensorWeCanUse = numberOfCellsWeCanUse / 16
        
        num_cells_per_validation_tensor = round(numberOfCellsPerTensorWeCanUse * 0.2)
        num_cells_per_training_tensor = numberOfCellsPerTensorWeCanUse - num_cells_per_validation_tensor 
        print(f"{Colors.BLUE}We use a 80/20 Train/Val split: Using {num_cells_per_training_tensor * 16} for training, {num_cells_per_validation_tensor * 16} for validation{Colors.RESET}")
        for i in tqdm(range(16), desc=f"{Colors.GREEN}Creating tensors{Colors.RESET}"):
            # Training set
            start_idx_train = int(i * num_cells_per_training_tensor) // 4
            end_idx_train = int((i + 1) * num_cells_per_training_tensor) // 4
            tensorTrain, labelsTrain = create_tensor_forDDM(path_to_seg_file, channels, shape, start_idx_train, end_idx_train, labels)
            np.save(os.path.join(training_tensor_path, f'tensor_{i}.npy'), tensorTrain) #  saves the tensor to disk as a .npy file
            np.save(os.path.join(training_tensor_path, f'labels_{i}.npy'), labelsTrain) # saves the corresponding labels to disk as a .npy file
            del tensorTrain 
            del labelsTrain 
            gc.collect() 

            # Test set
            start_idx_val = int(16 * num_cells_per_training_tensor) // 4 + int(i * num_cells_per_validation_tensor) // 4
            end_idx_val = int(16 * num_cells_per_training_tensor) // 4 + int((i + 1) * num_cells_per_validation_tensor) // 4
            tensorVal, labelsVal = create_tensor_forDDM(path_to_seg_file, channels, shape, start_idx_val, end_idx_val, labels)
            np.save(os.path.join(val_tensor_path, f'tensor_{i}.npy'), tensorVal)
            np.save(os.path.join(val_tensor_path, f'labels_{i}.npy'), labelsVal)
            del tensorVal 
            del labelsVal 
            gc.collect() 
    
    
    

# function to create a tensor using a start and end id
# UPDATE: removed cellNumber as parameter
def create_tensor_forDDM(path, channels, shape, start_idx, end_idx, labels):
    #print("Running create_tensor_forDDM")
    with Container(path, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        # Get indices of each cell type
        label1_idx = np.where(celltype == labels[0])[0] 
        label2_idx = np.where(celltype == labels[1])[0]
        # Select a subset of cells from each class
        selected_idx = np.concatenate([label1_idx[start_idx:end_idx],
                                       label2_idx[start_idx:end_idx]
                                      ])
        # Extract the specified image channels for the selected cells
        channel = []
        for i, c in enumerate(channels):
            channel.append(getattr(seg.content, c).images[selected_idx])
        # Resize and stack the channels to create the tensor
        tensor_list = []
        for c in channel:
            c = np.tile(c[..., np.newaxis], 1)
            c = tf.image.resize(c, size=shape)
            c = tf.squeeze(c)
            tensor_list.append(c)
        # Map cell type labels to integer values
        cell_label_mapping = { labels[0] : 0, labels[1]: 1}
        int_labels = [cell_label_mapping[label[0]] for label in celltype[selected_idx]]
        # Stack the channels and labels to create the final tensor and label arrays
        return np.stack(tensor_list, axis=-1), int_labels
    

# no filtering is done in this class! -- including debris here!!
def create_train80_test20_tensors_DEBRIS(path_to_seg_file, channels, training_tensor_path, testset_tensor_path, shape=(48,48)):
    print(f"{Colors.BLUE}Creating tensors from the provided seg file including those channels {channels}{Colors.BLUE}")
    print(f"{Colors.RED}Will resize all images to shape {shape} - Please adapt the shape parameter if you want to change this{Colors.BLUE}")
    delete_files_in_folder(training_tensor_path)
    delete_files_in_folder(testset_tensor_path)
    # first need to find out how many cells of each celltype we have!
    with Container(path_to_seg_file, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        # Get indices of each cell type
        lym_idx = np.where(celltype == 'lym')[0] 
        mon_idx = np.where(celltype == 'mon')[0]
        eos_idx = np.where(celltype == 'eos')[0]
        neu_idx = np.where(celltype == 'neu')[0]
        debris_idx = np.where(celltype == 'debris')[0]
        print(f"We have {len(lym_idx)} lymphocytes, {len(mon_idx)} monocytes, {len(eos_idx)} eosinophils, {len(neu_idx)} neutrophils and {len(debris_idx)} Debris images in this dataset")
        # calculate of which cells we have the fewest
        minimum = min(len(lym_idx), len(mon_idx), len(eos_idx), len(neu_idx), len(debris_idx))
        numberOfCellsWeCanUse = minimum*5
        print(f"We can maximally use {numberOfCellsWeCanUse} cells of this dataset if we want it to be equally balanced")
        numberOfCellsPerTensorWeCanUse = numberOfCellsWeCanUse / 16
        num_cells_per_test_tensor = round(numberOfCellsPerTensorWeCanUse * 0.2)
        num_cells_per_training_tensor = numberOfCellsPerTensorWeCanUse - num_cells_per_test_tensor 
        print(f"{Colors.BLUE}We use a 80/20 Train/Test split: Using {num_cells_per_training_tensor * 16} for training, {num_cells_per_test_tensor * 16} for testing{Colors.RESET}")
        for i in tqdm(range(16), desc=f"{Colors.GREEN}Creating tensors{Colors.RESET}"):

            # Training set
            start_idx_train = int(i * num_cells_per_training_tensor) // 5
            end_idx_train = int((i + 1) * num_cells_per_training_tensor) // 5
            tensorTrain, labelsTrain = create_tensor_DEBRIS(path_to_seg_file, channels, shape, start_idx_train, end_idx_train)
            np.save(os.path.join(training_tensor_path, f'tensor_{i}.npy'), tensorTrain) 
            np.save(os.path.join(training_tensor_path, f'labels_{i}.npy'), labelsTrain)
            del tensorTrain 
            del labelsTrain 
            gc.collect() 

            # Test set
            start_idx_val = int(16 * num_cells_per_training_tensor) // 5 + int(i * num_cells_per_test_tensor) // 5
            end_idx_val = int(16 * num_cells_per_training_tensor) // 5 + int((i + 1) * num_cells_per_test_tensor) // 5
            tensorVal, labelsVal = create_tensor_DEBRIS(path_to_seg_file, channels, shape, start_idx_val, end_idx_val)
            np.save(os.path.join(testset_tensor_path, f'tensor_{i}.npy'), tensorVal)
            np.save(os.path.join(testset_tensor_path, f'labels_{i}.npy'), labelsVal)
            del tensorVal 
            del labelsVal 
            gc.collect() 
    
# function to create a tensor using a start and end id DEBRIS
# UPDATE: removed cellNumber from the input parameters
def create_tensor_DEBRIS(path, channels, shape, start_idx, end_idx):
    with Container(path, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        # Get indices of each cell type
        lym_idx = np.where(celltype == 'lym')[0] 
        mon_idx = np.where(celltype == 'mon')[0]
        eos_idx = np.where(celltype == 'eos')[0]
        neu_idx = np.where(celltype == 'neu')[0]
        debris_idx = np.where(celltype == 'debris')[0]
        # Select a subset of cells from each class
        print(f"Selected indices per class: {len(lym_idx[start_idx:end_idx])},{len(mon_idx[start_idx:end_idx])},{len(eos_idx[start_idx:end_idx])},{len(neu_idx[start_idx:end_idx])},{len(debris_idx[start_idx:end_idx])} ...")

        selected_idx = np.concatenate([lym_idx[start_idx:end_idx],
                                       mon_idx[start_idx:end_idx],
                                       eos_idx[start_idx:end_idx],
                                       neu_idx[start_idx:end_idx],
                                       debris_idx[start_idx:end_idx]
                                      ])
        # Extract the specified image channels for the selected cells
        channel = []
        for i, c in enumerate(channels):
            channel.append(getattr(seg.content, c).images[selected_idx])
        # Resize and stack the channels to create the tensor
        tensor_list = []
        for c in channel:
            c = np.tile(c[..., np.newaxis], 1)
            c = tf.image.resize(c, size=shape)
            c = tf.squeeze(c)
            tensor_list.append(c)
        # Map cell type labels to integer values
        cell_label_mapping = {'lym': 0, 'mon': 1, 'eos': 2, 'neu': 3, 'debris':4}
        int_labels = [cell_label_mapping[label[0]] for label in celltype[selected_idx]]
        # Stack the channels and labels to create the final tensor and label arrays
        final_tensor = np.stack(tensor_list, axis=-1)
        assert final_tensor.ndim == 4 and final_tensor.shape[1:3] == shape, f"Final tensor shape mismatch: {final_tensor.shape}"
        return np.stack(tensor_list, axis=-1), int_labels
    
# for representation shift experiment - create 250 cells per celltype batches (...)
def create_tensors_250_per_celltype(path_to_seg_file, channels, tensor_path, shape=(48,48)):
    delete_files_in_folder(tensor_path)
    # first need to find out how many cells of each celltype we have!
    with Container(path_to_seg_file, 'r') as seg:  
        categories = list(seg.content.label.celltype.names)
        # Create one-hot encoding for the labels
        ohc = OneHotEncoder(categories=[categories])
        ohc.fit([[c] for c in categories])
        # Convert cell type labels to one-hot encoded vectors
        celltype_ohc = structured_to_unstructured(seg.content.label.celltype.reference[:]) 
        celltype = ohc.inverse_transform(celltype_ohc)
        for i in tqdm(range(10), desc=f"{Colors.GREEN}Creating tensors{Colors.RESET}"):
            # Training set
            start_idx_train = i*250
            end_idx_train = start_idx_train + 250
            print(f"{Colors.BLUE}start idx train is: {start_idx_train}{Colors.RESET}")
            print(f"{Colors.BLUE}end idx train is: {end_idx_train}{Colors.RESET}")
            tensorTrain, labelsTrain = create_tensor(path_to_seg_file, channels, shape, 1000, start_idx_train, end_idx_train)
            np.save(os.path.join(tensor_path, f'tensor_{i}.npy'), tensorTrain) #  saves the tensor to disk as a .npy file
            np.save(os.path.join(tensor_path, f'labels_{i}.npy'), labelsTrain) # saves the corresponding labels to disk as a .npy file
            del tensorTrain 
            del labelsTrain 
            gc.collect() 
    
# OUTSOURCED STUFF
"""
# function to create tensors but no label files - e.g. for whc_healthy files!
def create_tensors_no_labels_no_filering(path, channels, path_to_save_tensors_to, size_of_each_tensor,shape=(48,48)):
    delete_files_in_folder(path_to_save_tensors_to)
    print(f"Creating tensors for {path} ")
    with Container(path, 'r') as seg:
        channel = []
        for i, c in enumerate(channels):
            images = getattr(seg.content, c).images
            selected_images = images[:]
            channel.append(selected_images)
        nameIndex = 0
        for i in range(0, len(selected_images), size_of_each_tensor):
            tensor_list = []
            for c in channel:
                batch_c = c[i:i+size_of_each_tensor]
                batch_c = np.tile(batch_c[..., np.newaxis], 1)
                batch_c = tf.image.resize(batch_c, size=shape)
                batch_c = tf.squeeze(batch_c)
                tensor_list.append(batch_c)
            batch_tensor = np.stack(tensor_list, axis=-1)                                                                              
            tensor_filename = f'tensor_{nameIndex}.npy'
            np.save(os.path.join(path_to_save_tensors_to, tensor_filename), batch_tensor)
            # Clear memory
            del batch_tensor
            gc.collect()
            nameIndex = nameIndex + 1
            
"""