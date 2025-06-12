# Changelog

All notable changes to the WE-FTT project refactoring.

## [2.0.0] - 2024-06-12

### Added
- **Baseline Model Implementations**: Complete implementations for all baseline models
  - `RandomForestModel`: Random Forest classifier with feature importance
  - `CatBoostModel`: CatBoost gradient boosting model
  - `TabNetModel`: TabNet neural network with validation support
  - `XGBoostModel`: XGBoost classifier with early stopping
  - `LightGBMModel`: LightGBM classifier with optimization
  - `BaselineTrainer`: Unified trainer for all baseline models
  
- **Enhanced Training Pipeline**: Unified training script supporting all models
  - Consistent evaluation metrics across all models
  - Automatic model selection based on model name
  - Support for hyperparameter configuration
  - Comprehensive logging and result tracking

- **Project Structure**: Professional directory organization
  - Separated data into `raw/` and `processed/` directories
  - Created `sample_data/` for quick experimentation
  - Organized results by model type
  - Clear separation of source code and scripts

### Changed
- **Language**: Converted all code comments and documentation from Chinese to English
- **Data Organization**: 
  - Renamed `13-23EQ.csv` to `earthquake_catalog.csv` for clarity
  - Renamed `trimmed/` directory to `sample_data/` for better naming
- **Repository**: Updated all GitHub URLs to `https://github.com/PANXIONG-CN/WE-FTT`
- **Configuration**: Enhanced configuration management with baseline model parameters

### Removed
- **Author Information**: Removed author and email information from package metadata
- **Redundant Code**: Eliminated duplicate implementations across model files
- **Result Files**: Excluded all training output files from version control

### Fixed
- **Import Issues**: Fixed import paths for shared components
- **Model Integration**: Proper integration of baseline models with training pipeline
- **Configuration Management**: Centralized all model configurations

### Security
- **Data Privacy**: Enhanced .gitignore to prevent accidental commit of result files
- **Clean Repository**: Excluded all training outputs and temporary files

## Features Overview

### Core Models
1. **WE-FTT (Weight-Enhanced Feature-based Transformer)**
   - Knowledge-guided feature weighting
   - Environment-specific processing
   - Dynamic focal loss optimization

2. **Baseline Models**
   - Random Forest
   - CatBoost  
   - TabNet
   - XGBoost
   - LightGBM

### Training Pipeline
- Unified command-line interface for all models
- Consistent evaluation metrics
- Automated hyperparameter management
- Comprehensive logging and result saving

### Data Processing
- Complete preprocessing pipeline
- Association rule mining integration
- Knowledge extraction and weight calculation
- Environment-specific data handling

### Evaluation Framework
- Multiple performance metrics (Accuracy, F1, MCC, Cohen's κ, ROC-AUC)
- Cross-validation support
- Feature importance analysis
- Confusion matrix generation

## Usage Examples

### Train WE-FTT Model
```bash
python scripts/train.py --model_name we_ftt --epochs 50 --batch_size 32
```

### Train Baseline Models
```bash
python scripts/train.py --model_name random_forest
python scripts/train.py --model_name catboost
python scripts/train.py --model_name xgboost
```

### Run Complete Pipeline
```bash
# Data preprocessing
python scripts/run_preprocessing.py --input_data data/raw/earthquake_catalog.csv

# Model training
python scripts/train.py --model_name we_ftt

# Ablation study
python scripts/run_ablation.py
```

## Technical Improvements

### Code Quality
- Eliminated code duplication across files
- Implemented consistent error handling
- Added comprehensive type hints
- Enhanced docstring documentation

### Performance
- Optimized data loading with PyTorch DataLoaders
- Implemented early stopping for all applicable models
- Added memory-efficient batch processing
- Integrated GPU acceleration support

### Maintainability
- Modular architecture with clear separation of concerns
- Centralized configuration management
- Consistent coding standards throughout
- Comprehensive test framework foundation

---

This refactoring transforms the WE-FTT project from research code to production-ready, open-source software following industry best practices.