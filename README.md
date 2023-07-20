# Replication of Graph Convolutional Networks (GCN) for Power Flow Approximation
This project aims to replicate the models used in the following [paper](https://ieeexplore.ieee.org/document/8999165). We have implemented the models using PyTorch and DGL as the main frameworks.

## What's in this repository?
1. Datasets and trained models: You can find the datasets and the models trained for this project using the following [link](https://polimi365-my.sharepoint.com/:f:/g/personal/10797982_polimi_it/Eg0Lk8JCB3FCsqdDuSdTtQ4BevD6vghS6YhOwpy8byQXzA?e=0u76Bf)
2. Model architecture and hyperparameters: Detailed information about the models' architecture and the hyperparameters used can be found in this repository.
3. Jupyter notebook: We provide a Jupyter notebook that facilitates data conversion and includes the code for training, testing, and validation of the models.
 you CUDA version

## Environment Setup
To use this notebook, you need to install the required Python libraries listed in the requirements.txt file located in the project.

### CUDA Drivers
If you want to train models using CUDA, you should check the compatibility of your GPU on the NVIDIA website. Depending on your GPU's compatibility, you may need to install a specific version of PyTorch and DGL (Deep Graph Library) that supports your GPU. You can refer to the PyTorch and DGL documentation for more information.

### PyTorch and DGL Installation
To train the networks, you need to install two libraries: PyTorch and DGL. You can download and install them using the following links:

#### PyTorch: 
Visit the PyTorch website (https://pytorch.org/get-started/locally/) to download the appropriate version for your operating system. If you have CUDA drivers installed, select the PyTorch version that matches your CUDA version.
#### DGL: 
Visit the DGL website (https://www.dgl.ai/pages/start.html) for installation instructions. Follow the provided instructions to install DGL on your system.
Make sure to follow the installation instructions provided by PyTorch and DGL to ensure a successful installation.

## How to train the models
1. Download the dataset using the provided link and .
2. Create a folder named "datasets" within the project directory and place the downloaded dataset inside it.
3. Run the Jupyter Notebook section titled "Creation of the Custom Dataset" only once. Note that reading the dataset from disk may require some time.
4. Proceed to run the remaining sections of the notebook, which define the routines for training, testing, and validation, and train the model.
The scripts will create a folder containing the trained model and two log files with information about the training, validation, and testing processes. At the end of the notebook, there is a script that plots the results.

Feel free to explore this repository and reproduce the experiments conducted in the paper. If you have any questions or encounter any issues, please don't hesitate to reach out.
