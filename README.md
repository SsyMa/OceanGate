# OceanGate
This project is made for the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection). The objective is to find ships in satellite images using semantic segmentation. The challange also provides a training and test dataset with satellite images and RLE encoded masks as labels. The dataset can be found here: https://www.kaggle.com/competitions/airbus-ship-detection/data

## Members
| Name                  |NEPTUN|
|:----------------------|:-----|
|Szénássy Márton        |HIYXIQ|
|Nagy Dániel            |UU5SCQ|
|Varga-Labóczki Vazul   |H947XW|

## Milestone 1
- **data_loader.py**
    - Loads the dataset from the data folder.
    - Creates a single mask for each image by decoding the RLE encoded csv.
    - Handles data balancing inside batches so there is roughly the same amount of image with and without ships. (The original dataset is quite imbalanced, meaning the majority of images doesn't feature ships at all)
    - Handles memory-efficient loading using tensorflow.
    - Handles train/validation splitting.
    - Contains a demo function to show usage.

*Since the offered dataset is too large for version control, and downloading is only possible through a logged-in account, the project does not handle the downloading of the data automatically. If you wish to run the project, you must download the dataset separately and place it in the "data" folder.*

- **prepocessor.py**
    - Applies data augmentation during training (while keeping the augmented mask consistent).
    - Implements test-time augmentation for boosting inference accuracy (8 image variants).
- **exploration.ipynb** (Jupyter notebook to visualize and explore the dataset)
    - Contains various metrics and charts about the dataset.
    - Illustrates the training ready images and masks.
    - Illustrates data augmentations.
    - Contains a demo function to show usage.

## Git use
1. Pull develop
2. Create new branch for task
3. Commit as you like
4. git push --set-upstream <branch_name>
5. Create pull request to develop
    - resolve merge conflicts if needed
6. Merge branches
7. After the project has been developped we merge back to main.


