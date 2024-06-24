# Characterizing out-of-distribution generalization of neural networks: application to the disordered Su-Schrieffer-Heeger model

[![arXiv](https://img.shields.io/badge/arXiv-2406.10012-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2406.10012)  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12518289.svg)](https://doi.org/10.5281/zenodo.12518289)   

The aim of this repository is to accompany paper ["Characterizing out-of-distribution generalization of neural networks: application to the disordered Su-Schrieffer-Heeger model"](https://arxiv.org/abs/2406.10012) by K. Cybiński, M. Płodzień, M. Tomza, M. Lewenstein, A. Dauphin and A. Dawid.   

In this repository we provide Jupyter notebooks with a detailed walkthrough of the whole research process—from data generation through network training to CAM and clustering analysis.   

The helper functions for demonstrations are in folder `src`, and their naming convention is chosen to align with the respective demonstration notebooks they accompany.   

We also share with the reader the four networks highlighted in the paper and a directory containing 100 trained networks demonstrating a whole array of generalization performances. 

Using the notebooks we prepared you can recreate the figures from the paper, and explore for yourself the data on which we base our work.

Repository contents:
* [/src/](./src/) - folder containing all the helper functions for Jupyter Notebooks. Its contents are discussed along the notebooks they accompany.
* [/00_demonstrations_data/](./00_demonstrations_data/) - folder with all the assets needed for running the notebooks. Its full contents can be downloaded from Zenodo.
* [01_data_generation.ipynb](./01_data_generation.ipynb) - Notebook providing a step-by-step walkthrough of the data generation for calculation of observables from the SSH model. This noteboook allows you to recreate `Fig.3` from the paper.
    
    * [data_generation.py](./src/data_generation.py) - Helper file containing Numba's JIT-compilable implementations of second-quantized Hamiltonian generation, and plotting utilities.

* [02_architecture_training.ipynb](./02_architecture_training.ipynb) - Notebook which allows the reader to train from scratch a CNN for the topological phases classification task and recreate 
    * [data_loaders.py](./src/data_loaders.py) - helper functions for loading generated datasets into PyTorch-compatible format.
    * [training_utils.py](./src/training_utils.py) - helper functions implementing the training routine in PyTorch
* [03_CAM.ipynb](./03_CAM.ipynb) - This notebook provides a pipeline for generation of CAM heatmaps, and allows for recreation of `Fig.2` from the paper.
    * [cam_utils.py](src/cam_utils.py) - helper functions implementing CAM and grad-CAM.
* [04_clustering.ipynb](./04_clustering.ipynb) - A detailed walkthrough by the process of gathering activations from a network during forward pass, and later their PCA. This notebook also allows for recreation of `Fig.4`, `Fig.12` and `Fig.13` from the paper.
* [04a_figure_models_analysis.py](./04a_figure_models_analysis.py) - Python script for an automated clustering or CAM analysis of the networks highlighted in `Fig.2` of the paper. The trained models can be downloaded from Zenodo in folder [00_demonstrations_data](./00_demonstrations_data/).
* [05_explain_windnum.ipynb](./05_explain_windnum.ipynb) - Detailed walkthrough of the calculation of the winding number. This notebook accompanies `Appendix B` of the paper.
* [06_100_cnns_analysis.ipynb](./06_100_cnns_analysis.ipynb) - Interactive ipywidgets for analysis of the 100 trained CNNs provided in [/00_demonstrations_data/100_CNNs/](./00_demonstrations_data/100_CNNs/).
* [07_thermometer_encoding.ipynb](./07_thermometer_encoding.ipynb) - Interactive ipywidgets outlining the thermometer encoding from the appendinx of the paper.
    * [thermometer_encoding.py](./src/thermometer_encoding.py) - helper functions extending the loader and training functions to models with thermometer encoding, described in the appendix of the paper


Code was written by Kacper Cybiński (University of Warsaw) with help of Marcin Płodzień (ICFO), Alexandre Dauphin (PASQAL) and Anna Dawid (The Flatiron Institute).
