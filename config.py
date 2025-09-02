"""
Configuration file for Time Series Forecasting with Transformers
Contains all key parameters and settings for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    # Stock data settings
    "stock_symbol": "AAPL",  # Default stock symbol
    "start_date": "2022-01-01",
    "end_date": "2025-09-01",  # Updated to current date
    "data_frequency": "1d",  # Daily data
    
    # Technical indicators
    "sma_windows": [5, 10, 20, 50],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_window": 20,
    "bollinger_std": 2,
    
    # Feature engineering
    "lag_features": [1, 2, 3, 5, 10],
    "rolling_windows": [5, 10, 20],
    "volatility_window": 20,
}

# Model configuration
MODEL_CONFIG = {
    # Transformer architecture
    "d_model": 64,           # Model dimension
    "num_heads": 8,          # Number of attention heads
    "num_layers": 4,         # Number of transformer layers
    "ff_dim": 128,           # Feed-forward dimension
    "dropout_rate": 0.1,     # Dropout rate
    
    # Positional encoding options: 'learned', 'rope', 'legacy'
    "pos_encoding_type": "learned",   # Use learned embeddings (best for fixed sequences)
    
    # Advanced Attention Settings (Major Performance Boost!)
    "use_advanced_attention": True,   # Use advanced encoder blocks with better architecture
    "causal_masking": True,          # Prevent future peeking (essential for forecasting)
    "local_window": 14,              # Limit attention to +/- 14 days (None for global)
    "use_swiglu": True,              # Use SwiGLU MLP (better than standard)
    "mlp_ratio": 4,                  # MLP expansion ratio (4x is optimal)
    
    # Sequence settings
    "sequence_length": 24,   # Input sequence length (e.g., 24 days)
    "forecast_horizon": 1,   # Number of steps to forecast
    
    # Training settings - More aggressive training
    "batch_size": 16,          # Smaller batches for better learning
    "learning_rate": 1e-4,     # Lower learning rate for stability
    "epochs": 150,             # More epochs
    "validation_split": 0.1,
    "test_split": 0.1,
    
    # Callbacks - More patient training
    "early_stopping_patience": 25,    # More patience
    "reduce_lr_patience": 8,          # More patience before reducing LR  
    "reduce_lr_factor": 0.3,          # Bigger LR reduction
    "min_lr": 1e-8,
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    # Scaling
    "scaler_type": "minmax",  # "minmax" or "standard"
    
    # Outlier handling
    "outlier_method": "iqr",  # "iqr" or "zscore"
    "iqr_multiplier": 1.5,
    "zscore_threshold": 3,
    
    # Missing value handling
    "missing_value_method": "ffill",  # "ffill", "bfill", "interpolate"
    
    # Feature selection
    "correlation_threshold": 0.95,  # Remove highly correlated features
    "variance_threshold": 0.01,     # Remove low variance features
}

# Training configuration
TRAINING_CONFIG = {
    # Data splitting - More training data
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Callbacks
    "save_best_model": True,
    "save_training_history": True,
    "model_save_path": str(MODELS_DIR / "best_transformer.h5"),
    "history_save_path": str(RESULTS_DIR / "training_history.json"),
    
    # Logging
    "tensorboard_logs": str(RESULTS_DIR / "logs"),
    "save_plots": True,
    "plot_save_path": str(RESULTS_DIR / "plots"),
}

# Evaluation configuration
EVALUATION_CONFIG = {
    # Metrics
    "metrics": ["mse", "rmse", "mae", "mape"],
    
    # Baseline models
    "baseline_models": ["arima", "lstm", "prophet"],
    
    # ARIMA settings
    "arima_order": (1, 1, 1),
    
    # LSTM settings
    "lstm_units": [50, 50],
    "lstm_dropout": 0.2,
    
    # Visualization
    "plot_predictions": True,
    "plot_attention_weights": True,
    "save_evaluation_results": True,
}

# External features configuration
EXTERNAL_FEATURES_CONFIG = {
    # Economic indicators
    "include_economic_data": True,
    "economic_indicators": [
        "GDP", "inflation", "interest_rate", "unemployment"
    ],
    
    # Market sentiment
    "include_sentiment": True,
    "sentiment_sources": [
        "news_sentiment", "social_media_sentiment", "fear_greed_index"
    ],
    
    # Sector/industry data
    "include_sector_data": True,
    "sector_indicators": [
        "sector_performance", "industry_trends"
    ],
    
    # Global factors
    "include_global_factors": True,
    "global_indicators": [
        "vix", "dollar_index", "commodity_prices"
    ],
}

# File paths
FILE_PATHS = {
    "raw_data": str(RAW_DATA_DIR / "stock_data.csv"),
    "processed_data": str(PROCESSED_DATA_DIR / "processed_data.csv"),
    "sequences": str(PROCESSED_DATA_DIR / "sequences.npz"),
    "scaler": str(MODELS_DIR / "scaler.pkl"),
    "config": str(MODELS_DIR / "config.json"),
    "results": str(RESULTS_DIR / "results.json"),
    "plots": str(RESULTS_DIR / "plots"),
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(RESULTS_DIR / "project.log"),
    "console": True,
}

# GPU configuration
GPU_CONFIG = {
    "use_gpu": True,
    "memory_growth": True,
    "allow_memory_growth": True,
    "visible_devices": "0",  # GPU device to use
}

def get_config():
    """Return the complete configuration dictionary"""
    return {
        "data": DATA_CONFIG,
        "model": MODEL_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "training": TRAINING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "external_features": EXTERNAL_FEATURES_CONFIG,
        "file_paths": FILE_PATHS,
        "logging": LOGGING_CONFIG,
        "gpu": GPU_CONFIG,
    }

def update_config(new_config):
    """Update configuration with new values"""
    global DATA_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG, TRAINING_CONFIG
    global EVALUATION_CONFIG, EXTERNAL_FEATURES_CONFIG, FILE_PATHS, LOGGING_CONFIG, GPU_CONFIG
    
    if "data" in new_config:
        DATA_CONFIG.update(new_config["data"])
    if "model" in new_config:
        MODEL_CONFIG.update(new_config["model"])
    if "preprocessing" in new_config:
        PREPROCESSING_CONFIG.update(new_config["preprocessing"])
    if "training" in new_config:
        TRAINING_CONFIG.update(new_config["training"])
    if "evaluation" in new_config:
        EVALUATION_CONFIG.update(new_config["evaluation"])
    if "external_features" in new_config:
        EXTERNAL_FEATURES_CONFIG.update(new_config["external_features"])
    if "file_paths" in new_config:
        FILE_PATHS.update(new_config["file_paths"])
    if "logging" in new_config:
        LOGGING_CONFIG.update(new_config["logging"])
    if "gpu" in new_config:
        GPU_CONFIG.update(new_config["gpu"])

def print_config():
    """Print the current configuration"""
    config = get_config()
    print("Current Configuration:")
    print("=" * 50)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config()
