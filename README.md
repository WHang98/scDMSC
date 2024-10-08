# scDMSC: Deep multi-view subspace clustering for single-cell multi-omics data

We propose scDMSC, an unsupervised clustering algorithm based on deep multi-view subspace learning for single-cell multi-omics data, including single-cell transcriptomics, proteomics, and chromatin accessibility data. It coordinates omics heterogeneity through weighted reconstruction and  identifies shared latent features, elucidating correlations among omics. The algorithm outperforms most existing tools, reveals biological mechanisms in downstream analyses, and highlights the advantages of integrating multi-omics data.

## Requirements:

Python --- 3.8.13

pytorch -- 1.11.0

Scanpy --- 1.0.4

### Usage

Step 1: Prepare pytorch environment. See [Pytorch](https://pytorch.org/get-started/locally/).

Step 2: Prepare data. Data are available in the supplementary material.
The public datasets in the paper are stored as .h5 files.

Step 3: Data preprocess.
```
preprocess.py #Load and process the data
```
Step 4: Run on the Single-cell multi-omics datasets
```
scVSC.py #implementation of scDMSC algorithm
```
