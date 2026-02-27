# WE-FTT: Weight-Enhanced Feature-based Transformer for Earthquake Precursor Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of **Weight-Enhanced Feature-based Transformer (WE-FTT)**, a knowledge-guided deep learning model that analyzes Microwave Brightness Temperature (MBT) data to reveal environment-specific precursors to major earthquakes.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproducibility](#-reproducibility)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Experiments](#-experiments)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

## 🌟 Overview

The WE-FTT model addresses the challenge of identifying reliable satellite-based earthquake precursors by:

1. **Environment-specific Analysis**: Classifying global regions into distinct environmental zones
2. **Knowledge Mining**: Using association rule mining to discover zone-specific microwave frequency–polarization signatures
3. **Knowledge-guided Learning**: Integrating mined signatures as prior weights into a Feature-based Transformer architecture

Applied to AMSR-2 brightness temperature data (2013–2023), our approach reveals distinct precursor signatures across different environments and achieves significantly improved detection performance (~84% accuracy, MCC ~0.84).

### 🔬 Methodology Overview

The WE-FTT framework follows a four-stage process:

1. **Data Preparation & Preprocessing**: Surface type classification and spatio-temporal sampling
2. **Knowledge Mining**: K-means clustering and association rule mining to extract frequency-polarization signatures
3. **Knowledge-Guided Deep Learning**: Integration of mined weights into the WE-FTT model
4. **Result Analysis**: Performance comparison, ablation studies, and physical interpretation

## ✨ Key Features

- **🧠 Knowledge-Guided Architecture**: Integrates association rule mining with deep learning
- **🌍 Environment-Specific Processing**: Adapts to different geographical and environmental conditions
- **⚡ High Performance**: Achieves ~84% accuracy with robust evaluation metrics
- **🔬 Comprehensive Evaluation**: Includes ablation studies and baseline comparisons
- **📊 Visualization Tools**: Built-in plotting and analysis utilities
- **🔁 Reproducible Evaluation Protocol**: Standalone `evaluation_protocol/` workflow for placebo, ERA5 conditioning, leakage audit, and event-grouped splits
- **🐍 Modern Python Stack**: Built with PyTorch, scikit-learn, and modern ML libraries
- **📦 Easy Installation**: Simple pip install with comprehensive documentation

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.7+ (for GPU acceleration, optional)
- 8GB+ RAM recommended

### Option 1: Install from Requirements

```bash
# Clone the repository
git clone https://github.com/PANXIONG-CN/WE-FTT.git
cd WE-FTT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/PANXIONG-CN/WE-FTT.git
cd WE-FTT
pip install -e .
```

### Option 3: Docker (Coming Soon)

```bash
docker pull we-ftt:latest
docker run -it --gpus all we-ftt:latest
```

## ⚡ Quick Start

### 1. Prepare Your Data

Place your earthquake catalog and MBT data in the `data/raw/` directory:

```bash
data/raw/
├── earthquake_catalog.csv  # Earthquake catalog
└── mbt_data.parquet        # MBT brightness temperature data
```

### 2. Run Data Preprocessing

```bash
# Run the complete preprocessing pipeline
python scripts/run_preprocessing.py \
    --input_data data/raw/mbt_data.parquet \
    --output_dir data/processed
```

### 3. Train the WE-FTT Model

```bash
# Train the main WE-FTT model
python scripts/train.py \
    --model_name we_ftt \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### 4. Run Baseline Comparisons

```bash
# Train baseline models
python scripts/train.py --model_name random_forest
python scripts/train.py --model_name catboost
python scripts/train.py --model_name ft_transformer
```

### 5. Perform Ablation Study

```bash
# Run comprehensive ablation study
python scripts/run_ablation.py --output_dir results/ablation_study
```

## 🔁 Reproducibility

For revision-oriented, auditable experiments, use the standalone protocol under `evaluation_protocol/` instead of mixing logic into training scripts.

### Protocol entry points

- `evaluation_protocol/README.md`: protocol overview and directory contract
- `evaluation_protocol/run_evaluation.py`: one-command orchestrator
- `evaluation_protocol/reproduce_results.sh`: reproducible command bundle
- `evaluation_protocol/data_splits/event_grouped_splits_v1.json`: versioned event-grouped split manifest

### Typical run

```bash
AMSR2_ROOT="/path/to/AMSR2"
PY="WE-FTT/.venv/bin/python"

"${PY}" "WE-FTT/evaluation_protocol/run_evaluation.py" \
  --python "${PY}" \
  --amsr2_root "${AMSR2_ROOT}" \
  --control_dates_per_event 2 \
  --pixels_per_event_day 200 \
  --placebo_repeats 100 \
  --placebo_pixels_per_event_day 100 \
  --use_weights
```

For full details and output paths, see `evaluation_protocol/README.md`.

## 📖 Usage

### Training with Custom Configuration

```python
from src.config import WEFTTConfig
from src.models.we_ftt import create_we_ftt_model
from src.data_processing import DatasetCreator

# Load configuration
config = WEFTTConfig()

# Create model
model = create_we_ftt_model(
    num_features=10,
    num_classes=2,
    config=config.BEST_PARAMS,
    use_weight_enhancement=True
)

# Prepare data
dataset_creator = DatasetCreator(config)
train_data, val_data, test_data = dataset_creator.create_training_datasets()

# Train model (see scripts/train.py for complete example)
```

### Knowledge Mining

```python
from src.association_mining import KnowledgeMiner
from src.config import DataProcessingConfig

# Initialize knowledge miner
config = DataProcessingConfig()
miner = KnowledgeMiner(config)

# Run complete knowledge mining pipeline
results = miner.mine_knowledge(
    data=your_dataframe,
    features=config.COLUMNS_FEATURES,
    output_dir="knowledge_mining_results"
)

# Access results
cluster_info = results['cluster_info']
frequent_itemsets = results['frequent_itemsets']
association_rules = results['association_rules']
feature_weights = results['feature_weights']
```

### Custom Model Configuration

```python
# Define custom model parameters
custom_config = {
    'hidden_dim': 512,
    'n_heads': 8,
    'n_layers': 6,
    'dropout_rate': 0.1,
    'activation': 'relu',
    'use_weight_enhancement': True,
    'learnable_pos_enc': False
}

# Create model with custom config
model = create_we_ftt_model(config=custom_config)
```

## 📁 Project Structure

```
WE-FTT/
├── data/
│   ├── raw/                    # Raw, immutable data
│   └── processed/              # Processed datasets and mining results
├── results/                    # Training and experiment results (excluded from git)
│   ├── main_model/             # WE-FTT model results
│   ├── baseline_models/        # Baseline model results
│   └── ablation_study/         # Ablation study results
├── src/
│   ├── config.py               # Centralized configuration management
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── association_mining.py   # Knowledge mining algorithms
│   ├── models/
│   │   ├── we_ftt.py          # WE-FTT model implementation
│   │   ├── baselines.py       # Baseline models implementation
│   │   └── components.py      # Reusable model components
│   └── utils.py               # Utility functions
├── scripts/
│   ├── train.py               # Unified training script
│   ├── run_preprocessing.py   # Data preprocessing pipeline
│   └── run_ablation.py        # Ablation study script
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🏗️ Model Architecture

### WE-FTT Components

1. **Feature Embedding Layer**: Projects input features to model dimension
2. **Weight Enhancement Module**: Integrates mined knowledge as feature weights
3. **Positional Encoding**: Encodes feature position information
4. **Transformer Encoder**: Multi-head attention and feed-forward layers
5. **Classification Head**: Final prediction layer

### Key Innovations

- **Weight Fusion Layer**: Novel mechanism to combine feature embeddings with mined weights
- **Dynamic Focal Loss**: Adaptive loss function for imbalanced earthquake data
- **Environment-Specific Weights**: Learned weights tailored to different geographical zones

## 🧪 Experiments

### Available Experiments

1. **Main Model Training**: Train WE-FTT with full knowledge enhancement
2. **Baseline Comparisons**: RandomForest, CatBoost, TabNet, XGBoost, LightGBM
3. **Ablation Studies**: 20+ experiments testing individual components
4. **Hyperparameter Search**: Automated optimization with Optuna

### Running Experiments

```bash
# Single model training
python scripts/train.py --model_name we_ftt --epochs 50

# All baseline models
for model in random_forest catboost tabnet xgboost lightgbm; do
    python scripts/train.py --model_name $model
done

# Parallel ablation study
python scripts/run_ablation.py --parallel --num_epochs 20
```

### Experiment Tracking

Results are automatically saved with:
- Model checkpoints and configurations
- Training metrics and validation curves
- Test set evaluation with multiple metrics
- Visualization plots and confusion matrices

## 📊 Results

### Key Findings

1. Knowledge-guided weights significantly improve performance
2. Environment-specific processing is crucial for different geographical zones
3. The model shows robust performance across different earthquake magnitudes
4. Attention mechanisms effectively capture frequency-polarization interactions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/PANXIONG-CN/WE-FTT.git
cd WE-FTT

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Areas for Contribution

- 🐛 Bug fixes and improvements
- 📚 Documentation enhancements
- 🔬 New experiments and analysis
- 🚀 Performance optimizations
- 🌐 Additional data sources and formats

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact

- Email: xiong.pan@gmail.com

### Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/PANXIONG-CN/WE-FTT/issues)
- 💬 **Questions**: [Discussions](https://github.com/PANXIONG-CN/WE-FTT/discussions)
- 📧 **Email**: For research collaboration inquiries

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[🏠 Homepage](https://github.com/PANXIONG-CN/WE-FTT) • 
[📖 Documentation](https://we-ftt.readthedocs.io) • 
[🐛 Issues](https://github.com/PANXIONG-CN/WE-FTT/issues) • 
[💬 Discussions](https://github.com/PANXIONG-CN/WE-FTT/discussions)

</div>
