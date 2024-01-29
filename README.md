# Semi-Supervised Learning for Tabular Data

This repository contains all data, code, and results we gathered, created, and obtained during the works on our machine-learning project.

The contents of this repository are organized into the following directories:

- `data`
  - in this directory, you can find the prepared training, validation, and test sets
  - we do not provide the entire original dataset as it can be easily downloaded from [UCI ML Repository](https://doi.org/10.24432/C50K5N)

- `code`
  - all code for data preparation and experiments is located here
  - there is a separate file for each tested type of model, including all hyper-parameter configurations
  - you should be able to replicate our experiments with the same results simply by running the respective scripts

- `results`
  - this directory contains the results from all the conducted experiments, both in a raw text form and in the the form of confusion matrices
  - the results from training on the dataset `data/covtype-train.csv` are stored in the subdirectory `original`, while the subdirectory `balanced` holds the results from training on `data/covtype-train-balanced.csv`
  - for each set of experiments with one model, there are two pdf files with the above-mentioned confusion matrices, normalized either by the predicted classes (with the suffix `prec.pdf`) or by the true classes (with the suffix `rec.pdf`)

# Requirements

The code in this directory was run under the following setup:

- `python 3.10.12`
- `pandas 2.1.4`
- `matplotlib 3.6.0`
- `numpy 1.26.3`
- `scipy 1.11.4`
- `scikit-learn 1.3.2`
