# Fine-scale spatiotemporal predator-prey interactions in an Antarctic fur seal colony

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18141471.svg)](https://doi.org/10.5281/zenodo.18141471)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
This repository contains the code and analysis workflow for the paper **"Fine-scale spatiotemporal predator-prey interactions in an Antarctic fur seal colony"**. It includes the neural network training pipeline (YOLO-based) and the downstream ecological analysis of predator-prey spatial dynamics.

## Abstract
Density critically shapes population dynamics, with high densities exacerbating intraspecific competition and disease transmission, while low densities increase predation risk. To investigate spatiotemporal density patterns and predator-prey interactions in an Antarctic fur seal (*Arctocephalus gazella*) colony, we deployed an autonomous camera that captured minute-by-minute high-resolution images throughout a breeding season. Using a YOLO-based neural network, we identified adult males, females and pups, and avian predator-scavenger species: giant petrels (*Macronectes* spp.), brown skuas (*Stercorarius antarcticus*) and snowy sheathbills (*Chionis alba*). Analysing a dataset of 4.1 million automated detections from over 10,000 high-quality images, we found spatiotemporal abundance patterns corresponding with the known foraging and breeding behaviours of these species. Strong temporal associations also emerged between the abundance of pups and two of the avian species. Fine-scale spatial analyses further revealed that pups typically remained near other pups and adult females but avoided avian predators and territorial males. Notably, the proximity of adult fur seals of both sexes reduced pup predation risk, defined as the distance between the pup and the nearest bird, whereas proximity to other pups did not. This study provides a framework for studying density-dependent interactions in wild populations and highlights the value of remote observation in ecological research.

## Installation

### Prerequisites
* Python: 3.8.10
* python packages listed in requirements.txt
* Hardware: NVIDIA GeForce RTX 3070 (8GB VRAM)
* CUDA Toolkit: 11.2 (Required for reproducibility)
* NVIDIA Driver: 575.51 (Supports CUDA 11.2+)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

*Note: The data annotation was performed using [ClickPoints](https://clickpoints.readthedocs.io/en/latest/).*

## Data & Model Setup

To reproduce the results, you must first download the dataset and pre-trained weights from Zenodo.

1.  **Download Data:**
    *  Go to [10.5281/zenodo.18141471](https://doi.org/10.5281/zenodo.18141471)
    * Download `data.zip` and `model.h5`.
2.  **Unpack:**
    * Unzip `data.zip` into the repository root.
    * Place `model.h5` in the repository root.
3.  **Directory Structure:**
    Ensure your folder looks like this:
    ```text
    .
    ├── data/                  # Unzipped data folder
    ├── model.h5               # Network weights
    ├── train.ipynb
    ├── analyze.ipynb
    └── README.md
    ```

## Usage

### 1. Training (`train.ipynb`)
Run this notebook to retrain the network or finetune the model. 
* Follow the comments provided within the notebook.
* Ensure the data directory is correctly linked.

### 2. Analysis (`analyze.ipynb`)
This notebook contains the entire ecological data analysis presented in the paper.
* Generates the spatiotemporal plots and predator-prey interaction statistics.
* Input: The detection data (CSV/dataframe) produced by the network.

