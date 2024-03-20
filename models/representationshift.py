from shiftscope.helperfunctions import Colors
from tqdm import tqdm
import torch
import numpy as np
from scipy.stats import entropy


# calculate representation shift between two datasets within a specific layer of a trained CNN
def calculate_representation_shift(model, num_bins, dataset_S_loader, dataset_T_loader, include_labels_dataset_S, include_labels_dataset_T, layer_name, device):
    print(f"Calculating the representation shift between the two datasets in layer {layer_name} of the trained CNN.")
    activations = []  # Initialize the activations list here

    def get_activation(name):
        def hook(model, input, output):
            activations.append(output.detach())
        return hook

    hook = getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
    #hook = model.layer3.register_forward_hook(get_activation(layer_name))
    activations_S = get_all_activations(dataset_S_loader, include_labels_dataset_S, device, model, activations)
    # And like this for dataset_T_loader, which does not include labels:
    activations_T = get_all_activations(dataset_T_loader, include_labels_dataset_T, device, model, activations)
    # Unregister hook
    hook.remove()
    # Assuming activations are passed in as a list of numpy arrays with shape (num_samples, num_filters)
    # You may need to adjust this to fit your data structure
    # Find the global minimum and maximum to define the range of bins
    min_val = min(activations_S.min(), activations_T.min())
    max_val = max(activations_S.max(), activations_T.max())
    # Compute the representation shift
    representation_shifts = []
    for k in range(activations_S.shape[1]):  # Assuming second dimension is the filter dimension
        # Discretize the activations for this filter
        p, _ = np.histogram(activations_S[:, k], bins=num_bins, range=(min_val, max_val), density=True)
        q, _ = np.histogram(activations_T[:, k], bins=num_bins, range=(min_val, max_val), density=True)

        # Calculate the KL divergence
        kl_div = kl_divergence(p, q)
        representation_shifts.append(kl_div)

    # Calculate the mean KL divergence across all filters
    mean_representation_shift = np.mean(representation_shifts)
    print(f"{Colors.BLUE}The Representation Shift is: {mean_representation_shift}{Colors.RESET}")
    return mean_representation_shift


def kl_divergence(p, q):
    # Add a small constant to p and q to avoid division by zero
    p = np.asarray(p, dtype=np.float64) + 1e-3
    q = np.asarray(q, dtype=np.float64) + 1e-3

    # Normalize p and q to sum to 1
    p /= np.sum(p)
    q /= np.sum(q)

    return entropy(p, q)

# Function to get all activations from a dataset
def get_all_activations(loader, include_labels, device, model, activations):
    all_activations = []
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch[0].to(device) if include_labels else batch.to(device)
            _ = model(inputs)
            all_activations.append(activations[-1].cpu().numpy())  # Get the last stored activation
    return np.concatenate(all_activations, axis=0)