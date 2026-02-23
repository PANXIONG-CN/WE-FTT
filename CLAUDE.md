# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based machine learning research project for earthquake precursor detection using Microwave Brightness Temperature (MBT) data. The project implements a Weight-Enhanced Feature-based Transformer (WE-FTT) that combines association rule mining with deep learning to identify environment-specific precursors to major earthquakes (M≥7.0).

## Core Architecture

### Main Models
- **WE_FT_Transformer.py**: Primary Weight-Enhanced Feature-based Transformer implementation with knowledge-guided feature weighting
- **FT_Transformer.py**: Baseline Feature-based Transformer without knowledge enhancement  
- **FT_Transformer+RL.py**: Transformer with reinforcement learning components

### Baseline Models
- **RandomForest.py**: Random Forest baseline
- **catboost.py**: CatBoost gradient boosting
- **tabnet.py**: TabNet neural network
- **xg_light.py**: XGBoost and LightGBM implementations

### Data Mining Components
- **apriori.py**: Apriori association rule mining
- **eclat.py**: Eclat frequent itemset mining  
- **fp-growth.py**: FP-Growth algorithm implementation

## Key Data Structures

### Input Features
All models work with AMSR-2 brightness temperature data across multiple frequencies and polarizations:
- BT_06_H/V, BT_10_H/V, BT_23_H/V, BT_36_H/V, BT_89_H/V (10 features total)

### Knowledge Integration
The WE-FTT model incorporates mined association rules as feature weights:
- Weight columns follow pattern: `{feature}_cluster_labels_weight`
- Weights are derived from frequent itemset mining on discretized MBT data

### Data Organization
- **data/**: Contains processed parquet files (downsampled_f{0,1}t{0-4}.csv/parquet)
- **Apriori_Results/**: Association rule mining outputs and cluster information
- **ablation_results/**: Ablation study experimental results

## Common Development Commands

### Running Main Models
```bash
# Run Weight-Enhanced FT-Transformer
python WE_FT_Transformer.py

# Run baseline FT-Transformer  
python FT_Transformer.py

# Run FT-Transformer with RL
python FT_Transformer+RL.py

# Run baseline models
python RandomForest.py
python catboost.py
python tabnet.py
python xg_light.py
```

### Running Ablation Studies
```bash
# Run comprehensive ablation study
python ablation.py
```

### Data Mining and Analysis
```bash
# Run association rule mining
python apriori.py
python eclat.py
python fp-growth.py

# Generate visualizations
python picture/plt_result.py
python picture/plt_abla.py
python freqItemsets/plt.py
```

## Configuration

Models use class-based configuration systems:
- Hyperparameters defined in `Config` classes within each model file
- Key settings: batch size, learning rate, epochs, file paths
- Data paths typically point to `/home/panxiong/MBT/` directory structure

## Environment

- **Python**: 3.11.7 (conda environment: gpytorch)
- **Key Dependencies**: PyTorch, scikit-learn, pandas, numpy, optuna, matplotlib
- **Hardware**: Designed for GPU training with CUDA support
- **Data Format**: Parquet files for efficient large-scale data handling

## Results and Outputs

- Model training outputs saved to `{model_name}_results.txt`
- Ablation study results in `ablation_results/` directory
- Visualizations and plots in `picture/` directory
- Association mining results in `freqItemsets/` and `Apriori_Results/`

## Key Implementation Notes

- Models support distributed training with DistributedDataParallel
- Implements early stopping and learning rate scheduling
- Uses mixed precision training (GradScaler) for memory efficiency
- Extensive evaluation metrics including ROC-AUC, precision/recall, Cohen's kappa
- Knowledge-guided weighting is the core innovation in WE_FT_Transformer.py