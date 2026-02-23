# Knowledge-Guided Deep Learning for Earthquake Precursor Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of a Weight-Enhanced Feature-based Transformer (WE-FTT), a knowledge-guided deep learning model that analyzes Microwave Brightness Temperature (MBT) data to reveal environment-specific precursors to major earthquakes. This research was published in... (Please add your publication reference here).

[**Read the Paper**](docs/paper/main.tex)

---

## Abstract

Identifying reliable satellite-based earthquake precursors remains challenging due to weak signals obscured by strong environmental noise and inconsistent anomaly occurrences. We present a knowledge-guided framework that addresses these issues by first classifying global regions into distinct environmental zones and using association rule mining to discover zone-specific microwave frequency–polarization signatures indicative of pre-seismic conditions for major earthquakes (M≥7.0). We integrate these signatures as prior weights into a novel Weight-Enhanced Feature-based Transformer (WE-FTT) deep learning model, guiding it to focus on the most informative channels. Applied to AMSR-2 brightness temperature data (2013–2023), our approach reveals distinct precursor signatures across different environments and achieves significantly improved detection performance, outperforming baseline models. This environment-specific, knowledge-guided method paves the way toward more robust identification of potential seismic precursors from satellite microwave observations.

## Workflow

The core idea of this project is to integrate domain knowledge (obtained through association rule mining) with a deep learning model to improve the accuracy of seismic precursor detection. The complete workflow is illustrated below:

```mermaid
flowchart TD
    subgraph "A. Data Preparation & Preprocessing"
        A1[Raw Data Collection<br>(AMSR-2, Catalog, Environment)] --> A2{Surface Type Classification<br>(5 Zones)};
        A2 --> A3{Spatio-temporal Sampling<br>(Dobrovolsky Radius)};
        A3 --> A4[Generate Seismic &<br>Non-Seismic Samples];
    end

    subgraph "B. Knowledge Mining"
        A4 --> B1{K-Means Clustering<br>(Discretize MBT Data)};
        B1 --> B2{Apriori Association<br>Rule Mining};
        B2 --> B3[Extract High-Confidence<br>Freq-Polarization Combinations];
        B3 --> B4[Calculate Feature Weights];
    end

    subgraph "C. Knowledge-Guided Deep Learning"
        C1(MBT Features) --> C3;
        B4 -- Weights --> C2(Feature Weights);
        C2 --> C3{WE-FTT Model};
        C3 --> C4[Model Training & Evaluation];
    end

    subgraph "D. Result Analysis"
        C4 --> D1[Performance Comparison];
        C4 --> D2[Ablation Study];
        C4 --> D3[Physical Interpretation];
    end
    
    A --> B --> C --> D;
```

## Directory Structure

The project uses a standardized directory structure to separate code, data, and results:

```
WE-FTT/
├── data/
│   ├── raw/                  # Raw, immutable data
│   └── processed/            # Processed data
├── docs/                     # Documentation (Paper, Notes)
├── results/                  # Model outputs (Logs, Figures)
├── src/                      # Source code
│   ├── config.py             # Configuration file
│   ├── data_processing.py    # Data processing module
│   ├── association_mining.py # Association rule mining module
│   └── models/               # Model definitions
├── scripts/                  # Executable scripts
│   ├── run_preprocessing.py  # Script to run the preprocessing pipeline
│   └── train.py              # Script to run model training
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/WE-FTT.git
    cd WE-FTT
    ```

2.  It is recommended to use `conda` to create a new virtual environment:
    ```bash
    conda create -n seismic-precursors python=3.9
    conda activate seismic-precursors
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Follow these steps to reproduce the key results of this study.

### 1. Data Preparation

Place your raw data (e.g., `13-23EQ.csv` and other data downloaded from public sources) into the `data/raw/` directory.

### 2. Preprocessing and Knowledge Mining

Run the following script to perform surface classification, data sampling, clustering, and association rule mining. The processed data and mined weights will be saved in the `data/processed/` directory.

```bash
python scripts/run_preprocessing.py
```

### 3. Model Training

Use the unified training script to train the main WE-FTT model or any of the baseline models. All configurations (e.g., learning rate, batch size) can be modified in `src/config.py`.

- **Train the WE-FTT model:**
  ```bash
  python scripts/train.py --model_name we_ftt
  ```

- **Train a baseline model (e.g., RandomForest):**
  ```bash
  python scripts/train.py --model_name random_forest
  ```
  Supported baseline models include: `random_forest`, `lightgbm`, `tabnet`, `catboost`, `xgboost`.

Training logs and final results will be saved in the corresponding folders under the `results/` directory.

### 4. Run Ablation Study

To reproduce the ablation study results, run:

```bash
python scripts/run_ablation.py
```

## How to Cite

If you use this code or ideas from this work in your research, please cite our paper:

```bibtex
@article{Xiong2024,
  title   = {Knowledge-guided deep learning reveals environment-specific microwave precursors to major earthquakes},
  author  = {Pan Xiong and Cheng Long and Huiyu Zhou and Roberto Battiston and Angelo De Santis and Xuhui Shen},
  journal = {Your Journal/Preprint Server},
  year    = {2024},
  volume  = {},
  pages   = {}
}
```

## License

This project is licensed under the [MIT License](LICENSE). 