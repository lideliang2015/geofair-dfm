# GeoFair-DFM: Data–Feature–Model Attribution Framework

This repository contains the official code for the paper:
**Diagnosing Geographic Bias in Remote Sensing Building Extraction: A Data–Feature–Model Attribution Framework**

## 📖 Overview
This project proposes the Data–Feature–Model (DFM) attribution framework to diagnose geographic bias. 
The benchmark dataset (GeoFair-Building v1.0) used in this paper is available at: 
[https://github.com/lideliang2015/geofairbuilding](https://github.com/lideliang2015/geofairbuilding).

## 📂 Repository Structure
- `dfm/data_layer/`: Correlation analysis, stratified sampling, and label quality.
- `dfm/feature_layer/`: SHAP, t-SNE/UMAP, Prediction Error Mapping.
- `dfm/model_layer/`: Cross-architecture comparison, loss-surface, and decision-boundary.
- `dfm/fusion/`: Three-layer variance decomposition and MGWR.
- `dfm/causal/`: ATE/CATE estimation and counterfactual testing.

## ⚙️ Installation
```bash
conda create -n geofair-dfm python=3.9
conda activate geofair-dfm
pip install -r requirements.txt

# 1. Run inference to generate prediction masks and baseline metrics
python scripts/run_inference.py

# 2. Run attribution analyses
python scripts/run_data_attribution.py
python scripts/run_feature_attribution.py
python scripts/run_model_attribution.py
python scripts/run_causal_attribution.py



---

