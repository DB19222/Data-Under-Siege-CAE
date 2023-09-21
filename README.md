# Side-Channel Attack Experiment README

This repository contains the necessary files and instructions for conducting side-channel attack experiments using Convolutional Autoencoders (CAE) as portrayed in the paper "Data Under Siege: The Quest for the Optimal Convolutional Autoencoder in Side-Channel Attacks." [1]. The experiment flow is detailed below, along with references to the relevant code and contributors.

## Data
The ASCAD database was used, specifically the ATM_AES_v1_fixed_key dataset, to conduct these experiments. Two files out of this dataset were used: 'ASCAD.h5' containing 'raw' traces already trimmed to the points of interest where encryption occurs, and ASCAD_desync50.h5 with again 'raw' traces already trimmed to the points of interest but desynchronized with up to 50 points.

## Files Included

1. **optuna_hybrid.py**: This Python script facilitates the search for the optimal Convolutional Autoencoder (CAE) architecture.
2. **helper.py**: This file contains helper functions, most of which were made by [2] and [3] to ensure a consistent experimental setup.
3. **ta.py**: This script is used for the template attack and is based on code provided by Tom Slooff.
4. **run.py**: this Python script is used for testing the CAE in combination with the attack models MLP and CNN.

## Experiment Flow

Follow these steps to conduct the experiment:

1. **Choose Countermeasure**: Decide which countermeasure you want to test, options include 'gauss', 'rdi', and 'desync'. Set the chosen countermeasure in both `run.py` and `optuna_hybrid.py` in the 'Set Parameters' section. Note that instead of setting the parameters in the file, one can also choose to run the files with the following command: "python optuna_hybrid.py 'countermeasure' 'storage' 'study_name'" and "python run.py 'countermeasure'". Also note that for desynchronization, the noise is readily provided by the ASCAD database and wil not be generated in the code.

2. **Set Optuna Parameters**: In `optuna_hybrid.py`, configure the storage location and search name in the 'Set Parameters' section. The storage determines where Optuna stores its results (if you want to save this in memory set this to 'None'). Please head to the [optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html) for additional help regarding the storage parameter.

3. **Run Optuna Optimization**: Execute the Python script `optuna_hybrid.py`. This script initiates the search for the optimal convolutional autoencoder using the Optuna framework.

4. **Set Hyperparameters**: Take note of the hyperparameters discovered by Optuna and set them in the 'Set Parameters' section of the `run.py` file.

5. **Train CAE and Perform Attacks**: Run the Python script `run.py`. This will train the Convolutional Autoencoder (CAE) and subsequently perform the attacks using Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN). Data on the experiments, including model weights, statistics on training and guessing entropy will be stored in a newly created folder in the root folder of the code.

6. **Template Attack**: To run the template attack, execute the `ta.py` script. Note that this script is multithreaded for a maximum of 100 threads; if your computer has limited threads, it may lead to complications. The Guessing entropy of the template attack too will be stored in the newly created folder.

For additional help regarding the codes in this repository please contact the author(s) of this paper.

## References
[1]   van den Berg, D., Slooff, T., Brohet, M., Papagiannopoulos, K., & Regazzoni, F. (2023, June). Data Under Siege: The Quest for the Optimal Convolutional Autoencoder in Side-Channel Attacks. In 2023 International Joint Conference on Neural Networks (IJCNN) (pp. 01-09). IEEE.

[2]	  L. Wu and S. Picek, “Remove Some Noise: On Pre-processing of Side-channel 	Measurements with Autoencoders,” IACR Transactions on Cryptographic Hardware and 	Embedded Systems, pp. 389–415, Aug. 2020.

https://github.com/AISyLab/Denoising-autoencoder

[3]	  R. Benadjila, E. Prouff, R. Strullu, E. Cagli, and C. Dumas, “Deep learning for side-channel 	analysis and introduction to ASCAD database,” Journal of Cryptographic Engineering, vol. 	10, pp. 163–188, Nov. 2019.

https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/Readme.md
