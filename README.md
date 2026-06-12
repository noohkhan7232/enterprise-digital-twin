# Wind Turbine Acoustic Monitoring

Research-grade project for acoustic monitoring of wind turbines.

## Project Structure

```
├── config/             # Configuration files (experiments, model hyperparameters)
├── data/
│   ├── raw/            # Original, immutable acoustic recordings
│   ├── processed/      # Cleaned and feature-extracted data
│   └── augmented/      # Augmented datasets for training
├── docs/               # Project documentation
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── paper/              # Research paper drafts and figures
├── scripts/            # Standalone utility and automation scripts
├── src/
│   ├── preprocessing/  # Signal preprocessing and feature extraction
│   ├── models/         # Model architectures
│   ├── training/       # Training loops and experiment management
│   ├── inference/      # Inference and deployment code
│   └── utils/          # Shared utilities
└── tests/              # Unit and integration tests
```

## Getting Started

```bash
pip install -r requirements.txt
```
