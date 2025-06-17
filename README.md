# WSISA: Weakly Supervised Survival Analysis on Whole Slide Histopathology Images

> 2025 Spring Project | Optimizing weakly supervised survival prediction on gigapixel pathology slides

## Table of Contents
- [Introduction](#introduction)
- [Research Background](#research-background)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Optimization Highlights](#optimization-highlights)
- [Installation and Usage](#installation-and-usage)
- [Results and Evaluation](#results-and-evaluation)
- [References](#references)
- [Contributors](#contributors)

## Introduction

WSISA is an optimized framework for survival prediction on Whole Slide Images (WSI) without manual region annotation. This implementation achieves state-of-the-art performance on cancer survival prediction through:

- Adaptive patch sampling from gigapixel pathology slides
- Deep survival modeling with phenotype clustering
- Optimization for medical data constraints (small samples, high heterogeneity)
- Clinical-grade evaluation with C-index and hazard ratio analysis

**Key result**: Achieved **0.7021 C-index** on TCGA-GBM dataset (15% improvement over baseline)

## Research Background

| Challenge                  | Traditional Limitations       | Our Solution              |
|----------------------------|-------------------------------|---------------------------|
| Ultra-high resolution      | Manual ROI annotation         | Weakly supervised analysis|
| Tumor heterogeneity        | Handcrafted features          | Deep phenotype clusters   |
| Small medical datasets     | Linear Cox limitations        | CNN-Cox hybrid model      |

## Methodology

The WSISA framework implements a multi-stage approach for survival analysis:

1. **Adaptive Patch Sampling**: Extract informative regions from gigapixel images
2. **Phenotype Clustering**: Group similar histological patterns using unsupervised learning
3. **Cluster-specific CNNs**: Train specialized models for each phenotype cluster
4. **Risk Feature Aggregation**: Combine cluster-specific risk predictions
5. **Cox Survival Prediction**: Generate final survival risk scores

## Repository Structure
```
WSISA_Spring25/
├── config.py               # Global configuration parameters
├── main.py                 # Training pipeline entry point
├── survival_model.py       # Core survival model implementation
├── clinical_utils/        # Clinical data processing
│   ├── align_svs_clinical.py
│   └── split_patients.py
├── feature_cluster/        # Image feature clustering
│   ├── cluster_generator.py
│   └── kmeans_pca.py
├── model/                  # Survival prediction models
│   ├── deepconvsurv.py     # CNN feature extractor
│   └── cox_aggregator.py   # Cox risk predictor
├── utils/                  # Training utilities
│   ├── early_stopping.py
│   └── visualization.py
└── data/                   # Sample data (excluded from Git)
    ├── clinical.tsv
    └── follow_up.tsv
```

## Optimization Highlights

| Component                  | Problem                       |  Solution              |
|----------------------------|-------------------------------|---------------------------|
| Architecture               | Monolithic codebase           | Modular component design  |
| Feature Selection          | Noisy cluster features        | Wald-test filtering (p<0.1)|
|Training Process            | Overfitting risks             | EarlyStopping + loss monitoring      |
|Resource Usage              |	Memory constraints	          |Patch caching + lazy loading|

## Installation and Usage
**Prerequisites**
- Python 3.8+
- Pytorch 1.12+
- Whole Slide Image(.svs format)
- Clinical data with survival annotations

### Installation Steps
```bash
# Create conda environment
conda create -n wsisa python=3.8

# Install PyTorch with CUDA support
conda install pytorch==1.12 torchvision cudatoolkit=11.3 -c pytorch

# Install additional dependencies
pip install -r requirements.txt
```
## Configuration
### Core Parameters
```python
# config.py
CONFIG = {
    "patch_size": 512,           # Patch sampling size in pixels
    "n_clusters": 10,            # Number of phenotype clusters
    "p_threshold": 0.1,          # Significance threshold for feature filtering
    "min_cluster_size": 20,      # Minimum patches per cluster
    "batch_size": 32,            # Training batch size
    "learning_rate": 5e-5,       # Initial learning rate
    "early_stop_patience": 10    # Epochs to wait before early stopping
}
```
## Execution Pipeline
### Training Process
``` bash
python main.py \
  --data_dir ./data/svs_images \        # Path to WSI files
  --clinical_path ./data/clinical.tsv \ # Clinical data file
  --output_dir ./results                # Output directory
```
### Model Evaluation
```bash
python load_and_evaluate.py \
  --model checkpoints/cox_agg.pkl \  # Trained model path
  --data test_dataset.pkl            # Test dataset
```
## Results and Evaluation
### Performance Metrics

| Optimization Phase                  | Key Improvements                      |  Test C-index             |
|----------------------------|-------------------------------|---------------------------|
| Baseline             | Initial implementation           | 	0.5971 |
| Architecture Refactoring         | Modular design        | 	0.6889|
|Feature Filtering           | p-value thresholding (p<0.1)            | 0.7021     |

### Significant Phenotype Clusters
Cluster	|Hazard Ratio|	P-value	|Clinical Significance|
|----------------------------|-----------------|--------------|---------------------------|
#3	|2.64|	p < 0.001	|High-risk phenotype
#8	|1.95	|p = 0.003	|Moderate-risk phenotype
#11	|1.21	|p = 0.038	|Borderline significant

## References

- Zhu et al. "WSISA: Making Survival Prediction from Whole Slide Histopathological Images." CVPR 2017
- Katzman et al. "DeepSurv: Personalized Treatment Recommender System." BMC Bioinformatics 2018

## Contirbutors
- Bobby Jin
- Lexuan Jiang
- Yutong Du
