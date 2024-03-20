import numpy as np
from shiftscope.helperfunctions import Colors
class RepresentationShift:
    """
        This class can calculate the representation shift between two datasets for a trained model - giving insights into how trustworthy the predictions are.
    """
    def __init__(self, model, num_bins, dataset_S_loader, dataset_T_loader, include_labels_dataset_S, include_labels_dataset_T, layer_name, device):
        self.model = model
        self.num_bins = num_bins
        self.dataset_S_loader = dataset_S_loader
        self.dataset_T_loader = dataset_T_loader
        self.include_labels_dataset_S = include_labels_dataset_S
        self.include_labels_dataset_T = include_labels_dataset_T
        self.layer_name = layer_name
        self.device = device
        self.activations = []
        
        
    def calculate_representation_shift(self):
        print(f"Calculating the representation shift between the two datasets in layer {self.layer_name} of the trained CNN.")
        hook = self.model.layer3.register_forward_hook(self.get_activation(self.layer_name))
        activations_S = self.get_all_activations(self.dataset_S_loader, self.include_labels_dataset_S)
        # And like this for dataset_T_loader, which does not include labels:
        activations_T = self.get_all_activations(self.dataset_T_loader, self.include_labels_dataset_T)
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
            p, _ = np.histogram(activations_S[:, k], bins=self.num_bins, range=(min_val, max_val), density=True)
            q, _ = np.histogram(activations_T[:, k], bins=self.num_bins, range=(min_val, max_val), density=True)

            # Calculate the KL divergence
            kl_div = self.kl_divergence(p, q)
            representation_shifts.append(kl_div)
        print(representation_shifts)
        # Calculate the mean KL divergence across all filters
        mean_representation_shift = np.mean(representation_shifts)
        print(f"{Colors.BLUE}The Representation Shift is: {mean_representation_shift}{Colors.RESET}")
        return mean_representation_shift
    
    
    def kl_divergence(self, p, q):
        # Add a small constant to p and q to avoid division by zero
        p = np.asarray(p, dtype=np.float64) + 1e-3
        q = np.asarray(q, dtype=np.float64) + 1e-3

        # Normalize p and q to sum to 1
        p /= np.sum(p)
        q /= np.sum(q)

        return entropy(p, q)
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations.append(output.detach())
        return hook
    
    # Function to get all activations from a dataset
    def get_all_activations(self, loader, include_labels):
        all_activations = []
        with torch.no_grad():
            for batch in tqdm(loader):
                inputs = batch[0].to(self.device) if include_labels else batch.to(self.device)
                _ = self.model(inputs)
                all_activations.append(self.activations[-1].cpu().numpy())  # Get the last stored activation
        return np.concatenate(all_activations, axis=0)