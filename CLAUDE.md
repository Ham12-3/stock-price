# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Time Series Forecasting with Transformers** project that builds an AI-powered stock price prediction system. The project implements a Transformer-based deep learning model to forecast stock market movements using historical price data and external market signals.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Quick start and environment check
python scripts/quick_start.py

# View configuration
python config.py
```

### Key Scripts
- `scripts/quick_start.py` - Environment check and sample data generation
- `config.py` - Centralized configuration management for all project parameters

## Project Architecture

### Directory Structure
```
stock-price/
├── data/                   # Raw and processed datasets
│   ├── raw/               # Original data files
│   └── processed/         # Cleaned and preprocessed data
├── models/                 # Model implementations (Transformer, baselines)
├── scripts/                # Data processing and training scripts
├── notebooks/              # Jupyter notebooks for exploration
├── utils/                  # Utility functions
├── results/                # Model outputs and visualizations
├── docs/                   # Documentation (includes IMPLEMENTATION_GUIDE.md)
└── config.py              # Centralized configuration
```

### Core Components

**Configuration System (config.py)**
- Centralized configuration management with sections for data, model, training, evaluation
- Uses pathlib for cross-platform path handling
- Configurable parameters for Transformer architecture (d_model=64, num_heads=8, num_layers=4)
- Default sequence length of 24 time steps, forecast horizon of 1

**Model Architecture**
- Transformer-based model for time series forecasting
- Multi-head attention with positional encoding for temporal data
- Designed to handle both target variable and external features
- Implements encoder-decoder architecture adapted for continuous numeric data

**Data Pipeline**
- Uses yfinance for stock data collection
- Supports technical indicators (RSI, MACD, SMA, Bollinger Bands)
- Handles lag features, rolling statistics, and volatility calculations
- Preprocessing includes outlier removal, missing value handling, and feature scaling

**Training Configuration**
- Default training split: 70% train, 15% validation, 15% test
- Uses callbacks: EarlyStopping (patience=15), ReduceLROnPlateau, ModelCheckpoint
- Default batch size: 32, learning rate: 1e-4, max epochs: 100

**Evaluation System**
- Metrics: MSE, RMSE, MAE, MAPE
- Baseline model comparisons: ARIMA, LSTM, Prophet
- Includes attention weight visualization for interpretability

## Key Configuration Options

From config.py, important settings that can be modified:

### Model Parameters
```python
MODEL_CONFIG = {
    "d_model": 64,           # Model dimension
    "num_heads": 8,          # Number of attention heads  
    "num_layers": 4,         # Number of transformer layers
    "sequence_length": 24,   # Input sequence length
    "forecast_horizon": 1,   # Prediction steps
}
```

### Data Features
```python
DATA_CONFIG = {
    "stock_symbol": "AAPL",  # Default symbol
    "lag_features": [1, 2, 3, 5, 10],
    "rolling_windows": [5, 10, 20],
    "sma_windows": [5, 10, 20, 50],
}
```

### Training Settings
```python
TRAINING_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "early_stopping_patience": 15,
}
```

## Development Workflow

The project follows a structured ML pipeline:

1. **Data Collection** - Stock prices + technical indicators + external features
2. **Preprocessing** - Feature engineering, scaling, sequence creation  
3. **Model Building** - Transformer architecture with positional encoding
4. **Training** - With validation and callbacks for optimization
5. **Evaluation** - Against baseline models with multiple metrics
6. **Inference** - Model saving/loading pipeline for predictions

## Dependencies

Key packages (from requirements.txt):
- tensorflow>=2.10.0 - Deep learning framework
- pandas>=1.5.0, numpy>=1.21.0 - Data manipulation
- yfinance>=0.1.87 - Stock data source
- scikit-learn>=1.1.0 - ML utilities
- statsmodels>=0.13.0 - ARIMA baseline
- matplotlib, seaborn, plotly - Visualization
- prophet>=1.1.0 - Time series baseline

## Implementation Notes

- The project is currently in planning/setup phase - most implementation files referenced in docs/IMPLEMENTATION_GUIDE.md are not yet created
- The quick_start.py script generates sample data and provides environment checks
- Configuration is highly modular and can be updated via config.update_config()
- GPU support is configurable via GPU_CONFIG settings
- The Transformer model adapts attention mechanisms for continuous time series data rather than discrete tokens

## Getting Started

1. Run `python scripts/quick_start.py` for environment verification and sample data
2. Refer to `docs/IMPLEMENTATION_GUIDE.md` for detailed step-by-step implementation
3. Modify `config.py` parameters as needed for your specific use case
4. Follow the 6-phase implementation workflow outlined in the README