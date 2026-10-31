# GeoFair-DFM: Data–Feature–Model Attribution Framework

This repository contains the official code and supporting materials for the paper: 
**Diagnosing Geographic Bias in Remote Sensing Building Extraction: A Data–Feature–Model Attribution Framework**

## 📖 Overview

This project proposes the Data–Feature–Model (DFM) attribution framework to diagnose geographic bias in deep learning-based building extraction. The framework decomposes bias into data, feature, and model layers, and uses causal inference to separate intrinsic geographic difficulty from algorithmic bias. 

The benchmark dataset (**GeoFair-Building v1.0**) used in this paper is available at: 
[https://github.com/lideliang2015/geofairbuilding](https://github.com/lideliang2015/geofairbuilding).

## 📂 Repository Structure

- `dfm/data_layer.py`: Correlation analysis and spatial-lag correlation.
- `dfm/feature_layer.py`: SHAP values, t-SNE/UMAP, and spatial error mapping.
- `dfm/model_layer.py`: Loss-surface analysis and decision-boundary shifts.
- `dfm/fusion.py`: Three-layer variance decomposition.
- `dfm/causal.py`: ATE/CATE estimation and counterfactual testing.
- `scripts/run_inference.py`: Generates prediction masks and baseline metrics.

## ⚙️ Environment & Installation

We recommend using Conda to manage the environment. 

```bash
# 1. Create and activate a new environment
conda create -n geofair-dfm python=3.9
conda activate geofair-dfm

# 2. Install dependencies
pip install -r requirements.txt


# 1. Run inference to generate prediction masks, baseline metrics, and deep features
python scripts/run_inference.py

# 2. Run data-layer attribution
python -m dfm.data_layer

# 3. Run feature-layer attribution
python -m dfm.feature_layer

# 4. Run model-layer attribution
python -m dfm.model_layer

# 5. Run three-layer fusion
python -m dfm.fusion

# 6. Run causal separation
python -m dfm.causal
