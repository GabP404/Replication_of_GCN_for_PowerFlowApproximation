# Replication of GCN for Power Flow Approximation
This project aims to replicate the models used in the following [paper](https://ieeexplore.ieee.org/document/8999165). We have implemented the models using PyTorch and DGL as the main frameworks.

## What's in this repository?
1. Datasets and trained models: You can find the datasets and the models trained for this project using the following [link](https://mega.nz/folder/VbdzTIpI#cE4o12YsoQzeztSVPdM7bg)
2. Model architecture and hyperparameters: Detailed information about the models' architecture and the hyperparameters used can be found in this repository.
3. Jupyter notebook: We provide a Jupyter notebook that facilitates data conversion and includes the code for training, testing, and validation of the models.

## How to train the models
1. Download the dataset using the provided link.
2. Create a folder named "dataset" within the project directory and place the downloaded dataset inside it.
3. Run the Jupyter Notebook section titled "Creation of the Custom Dataset" only once. Note that reading the dataset from disk may require some time.
4. Proceed to run the remaining sections of the notebook, which define the routines for training, testing, and validation, and train the model.
The scripts will create a folder containing the trained model and two log files with information about the training, validation, and testing processes. At the end of the notebook, there is a script that plots the results.

Feel free to explore this repository and reproduce the experiments conducted in the paper. If you have any questions or encounter any issues, please don't hesitate to reach out.

