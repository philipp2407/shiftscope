## Domain Shift Investigation in Holographic Leukocyte Images

This repository contains modules developed to investigate domain shifts in holographic images of leukocytes. These modules were created during my time as a research assistant in Singapore. Each module is designed to function independently or in conjunction with others, providing a flexible and comprehensive toolkit for cell image analysis and machine learning. 

__Note:__ Be aware that the imports from __*'cellfaceMultistreamSupport'*__ won't work as I am not able to share this module publicly.

### Modules Overview

__1. Filtering Submodule__
Description: Provides functions to filter cell images based on morphological characteristics.
Key Functions:
    - Filter by morphological features
    - Filter by glcm features (Gray Level Cooccurrence Matrix)

__2. MergeSegContainers Submodule__
Description: Contains functions to merge multiple segmentation containers into a single container, facilitating seamless integration with data loaders.
Key Functions:
    - Merge containers
    - Validate container integrity

__3. Dataloader Submodule__
Description: Offers functions to create custom tensors and cell datasets ready for machine learning model training.
Key Functions:
    - Generate custom tensors
    - Prepare cell datasets
    - Integrate with ML models

__4. Models Submodule__
Description: Provides tools for training machine learning models and using them for predictions on cell datasets.
Key Functions:
    - Train models
    - Validate models
    - Predict on new datasets

__5. Morphfeatures Submodule__
Description: Calculates morphological features of cell datasets to aid in analysis and classification.
Key Functions:
    - Extract morphological features
    - Feature normalization
    - Feature selection

__6. StatisticalTests Submodule__
Description: Includes functions to perform statistical tests on the morphological features of cells to assess domain shifts.
Key Functions:
    - Perform t-tests
    - Conduct ANOVA
    - Calculate correlation coefficients

__7. Visualise Submodule__
Description: Contains functions to visualize cells and plot various types of graphs, including KDE plots, for comprehensive data analysis.
Key Functions:
    - Visualize cell images
    - Generate KDE plots
    - Create scatter plots
