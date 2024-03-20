from shiftscope.colormaps import CellfaceStdCMap,CellfaceStdCMapLeukocytes,CellfaceStdNorm
import matplotlib.pyplot as plt
from shiftscope.cellfaceMultistreamSupport.cellface.storage.container import Container
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as clrs
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
import numpy as np
import math


def draw_cell(image, ax, fig):
    im1 = ax.imshow(image, cmap=CellfaceStdCMap, norm=CellfaceStdNorm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbr = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbr.set_ticks([-3,0,3,6,9,12])
    cbr.ax.tick_params(labelsize=20)
    ax.axis('off')
    fontprops = FontProperties(size=20)  # Adjust the size as needed, here it's 14

    scalebar = AnchoredSizeBar(ax.transData, 29, '10 $\mu m$', 'lower left',  
                           pad=0.1, color='black', frameon=False, size_vertical=1, 
                           fontproperties=fontprops)
    ax.add_artist(scalebar)
    
    
# function to visualise some cells...
def plot_images(path, filter_mode, channel='phase', filter_indices=[], labels= [], batch_size=100, batch_number=0, filter_indices_path="", color_mode = ""):
    print(f"Plotting batch number {batch_number} {channel} images (batch size is {batch_size})") 
    with Container(path, 'r') as seg:
        total_images = seg.content.phase.images.shape[0]
    if filter_mode == "use_indices":
        print("Using the provided filter indices")
        filter_indices = filter_indices
        print(f"Length of filter indices is {len(filter_indices)}")
    elif filter_mode == "use_filter_indices_path":
        print("Using the path to the filter indices provided to get the filter indices")
        with open(filter_indices_path, "rb") as f:
            filter_indices = pickle.load(f)
            print(f"Length of filter indices is {len(filter_indices)}")
    elif filter_mode == "no_filtering":
        print("No filtering used")
        filter_indices = list(range(total_images))
        print(f"Length of filter indices is {len(filter_indices)} - as no filtering is used")
    # show the images for visual inspection!!
    # Calculate the number of rows, rounding up if not divisible by 10
    num_rows = math.ceil(batch_size / 10)
    # Use integer values for the number of rows and columns
    fig, axs = plt.subplots(num_rows, 10, figsize=(40, num_rows * 4))  # Adjust the height factor as needed
    # Calculate the starting and ending indices for the batch
    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    # Access the container and display the images
    with Container(path, 'r') as seg:
        for i in range(start_index, min(end_index, len(filter_indices))):
            if channel == "phase":
                img = np.float32(seg.content.phase.images[filter_indices[i]])  # Get the ith image
            elif channel == "hologram":
                img = np.float32(seg.content.hologram.images[filter_indices[i]])  # Get the ith image
            elif channel == "amplitude":
                img = np.float32(seg.content.amplitude.images[filter_indices[i]])  # Get the ith image
            elif channel == "mask":
                img = np.float32(seg.content.mask.images[filter_indices[i]])  # Get the ith image
            
            # Calculate the row and column index for the current image
            adjusted_index = i - start_index
            row = adjusted_index // 10
            col = adjusted_index % 10
            if color_mode == "color":
                draw_cell(img, axs[row, col], fig)  # Draw the cell image on the appropriate subplot
            else:
                axs[row, col].imshow(img,cmap='gray')
            if labels == []:
                axs[row, col].set_title(f"Index: {filter_indices[i]}",size=30) 
            else:
                axs[row, col].set_title(f"Index: {labels[i]}",size=30) 


    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the figure with the images