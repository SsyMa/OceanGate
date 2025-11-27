# OceanGate
This project is made for the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection). The objective is to find ships in satellite images using semantic segmentation. The challange also provides a training and test dataset with satellite images and RLE encoded masks as labels. The dataset can be found here: https://www.kaggle.com/competitions/airbus-ship-detection/data

## Members
| Name                  |NEPTUN|
|:----------------------|:-----|
|Szénássy Márton        |HIYXIQ|
|Nagy Dániel            |UU5SCQ|
|Varga-Labóczki Vazul   |H947XW|

## Milestone 1
- **download_dataset.py**
    - Downloads and extracts the database to the data folder.

*Since the competition database could only be downloaded by authenticated kaggle users, a few extra steps should be made before running this script:*

*Make sure, you have an account on kaggle and accepted the terms of the challange. After that you should aquire the kaggle.json (account setting -> create new token) and place it in the C:/user/username/.kaggle folder. The script will aquire the necessary tokens for authorization from this json.*

*Be aware that the datbase zip alone is around 30Gb, so only run this script with a stable internet connection and expect it to take a good while.*
- **data_loader.py**
    - Loads the dataset from the data folder.
    - Creates a single mask for each image by decoding the RLE encoded csv.
    - Handles data balancing inside batches so there is roughly the same amount of image with and without ships. (The original dataset is quite imbalanced, meaning the majority of images doesn't feature ships at all)
    - Handles memory-efficient loading using tensorflow.
    - Handles train/validation splitting.
    - Contains a demo function to show usage.
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

## Starting training

To get started with training in this project on your local computer (using WSL), follow the steps below.

1. Create a Python virtual environment and activate it (inside WSL):

```bash
# create venv
python -m venv venv

# activate the virtual environment
source venv/bin/activate
```

2. Install required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*If you want to use your GPU for the training, you need to install the appropriate tensorflow version:*

```bash
pip install tensorflow[and-cuda]
```

*Run src/tf_test.py to check if tensorflow can see the GPU on your machine.*

3. Download and place the Airbus Ship Detection dataset into the `data/` folder (see Kaggle competition page).

4. Start training using the trainer script:

```bash
python src/trainer.py

```
## Evaluation
You can find the evaluation of the best model at the time in the notebooks/evaluation.py notebook. The notebook saves the outputs of the previous run, so you can instantly see the results. (Metrics, Tensorboard, Predicted mask visualization).

If you wish to run the evaluation by yourself, make sure to follow the steps listed above in the "Starting training" section.

*If the first block throws an error related to "bad backend", run the following command at the root of the project, then restart the notebook kernel:*
```bash
echo "backend : Agg" > ~/.config/matplotlib/matplotlibrc

```

