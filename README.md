## CNN with Attention for Demographic Inference

This repository contains a deep learning pipeline for inferring demographic parameters from simulated genomic SNP data.

The approach combines:
- a convolutional neural network (CNN) applied to SNP matrices
- an attention-based model to aggregate predictions across simulation replicates.

The pipeline is implemented in TensorFlow (CNN) and PyTorch (attention module).

## Repository structure

- `main_CNNAtt.ipynb`  
  Main notebook used to run the training and evaluation pipeline.

- `code/`  
  Python scripts implementing the CNN model, data loading utilities, and attention module.

- `parameters/`  
  JSON configuration files containing training and model hyperparameters.

  This repository provides a modular framework for demographic parameter inference using simulation-based deep learning.

