# VTrans: A VAE-based Pre-trained Transformer Method for Microbiome Data Analysis

## Overview
In this paper, we introduce a VAE-based pre-trained Transformer method for microbiome data analysis deep learning model called VTrans, which is built upon a pre-trained VAE and Transformer encoder. The main goal of VTrans is to enhance the feature representation of individual datasets using a large-scale microbiome dataset of cancer patients, and to assess patient risk. Additionally, inspired by the concept of computer vision saliency maps, we calculate the saliency of microbiota to determine their impact on different cancer diseases, thus aiding in medical analysis. Initially, the microbial data of a cancer data is encoded by VAE to obtain a reconstructed representation containing latent feature distributions, and then added to the original data to obtain an enhanced representation of the data. Then, this enhanced representation, along with the reconstructed data, is fed into the cross co-attention Transformer module for feature reconstruction. Finally, the classification task is completed through a linear layer. It is worth noting that multi-modal data serves as an optional input. The multi-modal data is first passed through a feedforward neural network for dimensional transformation before being fed into the Transformer. After training the model, microbiome saliency computation is achieved using a method similar to saliency map calculation of computer vision. ``It is worth noting that the multimodal data is included as an optional module.``

![image](overview.jpg)

## Installations
* Windows
* NVIDIA GPU (both pretrained VAE and trained VTrans on a single Nvidia GeForce RTX 3090)
* ``pip install -r requiremnts.txt``

## Data
All the datasets used in this paper can be downloaded from url：https://doi.org/10.5281/zenodo.14166580.

## Data preprocessing
Before running VTrans, you need to perform data preprocessing：
* ``data_process/VAE_data_preparation.ipynb``: you can obtain microbiome data compatible with the pre-trained VAE model through this script.
* ``data_process/VTrans_data_preparation.ipynb``: you can obtain training and testing data compatible with the VTrans model, including normalized microbiome data and the classification labels corresponding to each sample through this script.

## Pre-train VAE
``python pretrain_vae/vae_main.py``：you can perform the pre-training process for the VAE model through this script.

## Training VTrans
``python VTrans/main.py``：you can run the VTrans model through this script.

## Contact details
If you have any questions, please contact shixinyuan217@aliyun.com.
