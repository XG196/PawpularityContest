## What is this competition about?
#### Prediction of Pawpularity is kind of like a cuteness meter, the goal is to 
#### rank pet photos, even suggest improvement to help pets get shelters.


## Evaluation
#### RMSE between groundtruth(GT) and predicted Pawpularity. 

## Timeline
#### Sept 23th 2021 - Start Date
#### Jan 6th 2022 - Entry Deadline
#### Jan 13th 2022 - Final Submission Deadline 


## Software and Hardware
#### Use Pytorch and Tesla P100 16GB GPU acceleration

##    Installations
####  conda create -n pawpularity python==3.6
####  conda activate pawpularity

####  pip install opencv-python   
####  pip install timm   Pytorch Image Models
####  pip install --upgrade wandb
####  conda install -c anaconda pandas 
####  conda install -c conda-forge tqdm
####  conda install -c conda-forge joblib
####  conda install -c anaconda scikit-learn
####  conda install -c conda-forge albumentations

## Model Development
## Use SWIN transformer with different FC layers, dropout maybe added to prevent overfitting.
## The hyperparameters such as learning rate needed to be specified for each model architecture.

## Bugs
#### RuntimeError: Only one file(not dir) is allowed in the zipfile: upgrade Pytorch to 1.8



