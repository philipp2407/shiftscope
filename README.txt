Module by Philipp Schupeck to investigate the Domain Shift:


shiftscope module structure:

__________
shiftscope |
__________ |
    |
    |
    |
    |
    |---------(cellfaceMultistreamSupport): a modified copy of the cellface library with added support for multistream models
    |
    |
    |
    |---------(dataloader)---------> celldataloader.py: 
    |                                    - class: CellfaceDatasetWithLabels 
    |                                    - class: CellfaceDatasetNoLabels
    |                                    - delete_files_in_folder()
    |                                    - delete_files_in_folder()
    |                                    - create_tensors_no_labels()
    |                                    - create_train80_val10_test10_tensors()
    |                                    - create_tensor()
    |                                    - create_train80_val20_tensors()
    |
    |
    |
    |---------(filtering)---------> filterCells.py:
    |                                      - filter_container_and_return_indices()
    |                                      - filter_container_and_save_indices_to_file_and_return_indices_and_path()
    |
    |
    |
    |---------(models)---------> modeltraining.py:
    |                                 -  trainResnet()
    |                            predictions.py:
    |                                 -  predict()
    |                            representationshift.py:
    |                                 - calculate_representation_shift()
    |
    |
    |
    |---------(morphfeatures)---------> featurs.py:
    |                                       - calculate_morphological_features_paths_withIndices()
    |
    |
    |
    |---------(statisticaltests)---------> statisticalTestsPerformer.py:
    |                                             - calculate_stats_and_return_average_max_min_p25_p975_variance_median()
    |                                             - run_U_Test()
    |                                             - run_ANOVA_Test()
    |                                             - calculate_ECO_vector()
    |                                             - print_eco_profile_table()
    |                                             
    |
    |
    |
    |---------(visualise)---------> cellVisualiser.py:
    |                                      - draw_cell()
    |                                      - plot_images()
    |
    |
    |
    |---------(high_level_experiments): some more high level stuff such as making complete experiments combining stuff from all library parts
    |
    |
____