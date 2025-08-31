#!/usr/bin/env python3
"""
Utility Functions for Stock Price Forecasting
Contains helper functions for data processing, visualization, and model utilities
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, PREPROCESSING_CONFIG

def create_sequences(data, target_col, feature_cols, sequence_length=24, forecast_horizon=1):
    """
    Create sequences for time series modeling
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target column name
        feature_cols (list): List of feature column names
        sequence_length (int): Length of input sequences
        forecast_horizon (int): Number of steps to forecast
        
    Returns:
        tuple: (X, y) where X is sequences and y is targets
    """
    print(f"🔄 Creating sequences with length {sequence_length}...")
    
    # Remove rows with NaN values
    data_clean = data.dropna()
    
    if len(data_clean) < sequence_length + forecast_horizon:
        raise ValueError(f"Not enough data. Need at least {sequence_length + forecast_horizon} rows, got {len(data_clean)}")
    
    # Extract features and target
    features = data_clean[feature_cols].values
    target = data_clean[target_col].values
    
    X, y = [], []
    
    for i in range(sequence_length, len(data_clean) - forecast_horizon + 1):
        # Input sequence
        X.append(features[i-sequence_length:i])
        # Target value(s)
        y.append(target[i:i+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Created {len(X)} sequences")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X, y

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Args:
        X (np.array): Input sequences
        y (np.array): Target values
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation
        
    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print(f"📊 Splitting data: {train_ratio:.1%} train, {val_ratio:.1%} val, {1-train_ratio-val_ratio:.1%} test")
    
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def scale_data(X_train, X_val, X_test, scaler_type='minmax'):
    """
    Scale the data using specified scaler
    
    Args:
        X_train, X_val, X_test (np.array): Input data splits
        scaler_type (str): 'minmax' or 'standard'
        
    Returns:
        tuple: Scaled data and fitted scaler
    """
    print(f"⚖️  Scaling data using {scaler_type} scaler...")
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    # Get original shapes
    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape
    
    # Reshape for scaling (combine sequence and feature dimensions)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data only
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to original dimensions
    X_train_scaled = X_train_scaled.reshape(train_shape)
    X_val_scaled = X_val_scaled.reshape(val_shape)
    X_test_scaled = X_test_scaled.reshape(test_shape)
    
    print("✅ Data scaling completed")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def scale_targets(y_train, y_val, y_test, scaler_type='minmax'):
    """Scale target variables"""
    print(f"🎯 Scaling targets using {scaler_type} scaler...")
    
    if scaler_type == 'minmax':
        target_scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        target_scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    # Reshape targets for scaling
    y_train_reshaped = y_train.reshape(-1, 1)
    y_val_reshaped = y_val.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    # Fit on training data only
    y_train_scaled = target_scaler.fit_transform(y_train_reshaped)
    y_val_scaled = target_scaler.transform(y_val_reshaped)
    y_test_scaled = target_scaler.transform(y_test_reshaped)
    
    # Reshape back
    y_train_scaled = y_train_scaled.reshape(y_train.shape)
    y_val_scaled = y_val_scaled.reshape(y_val.shape)
    y_test_scaled = y_test_scaled.reshape(y_test.shape)
    
    print("✅ Target scaling completed")
    
    return y_train_scaled, y_val_scaled, y_test_scaled, target_scaler

def calculate_metrics(y_true, y_pred, scaler=None):
    """
    Calculate evaluation metrics
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        scaler: Optional scaler to inverse transform values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Inverse transform if scaler provided
    if scaler is not None:
        y_true_flat = scaler.inverse_transform(y_true_flat.reshape(-1, 1)).flatten()
        y_pred_flat = scaler.inverse_transform(y_pred_flat.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # Handle MAPE calculation (avoid division by zero)
    mask = y_true_flat != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = float('inf')
    
    # Additional metrics
    mae_percentage = (mae / np.mean(np.abs(y_true_flat))) * 100 if np.mean(np.abs(y_true_flat)) != 0 else float('inf')
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MAE_Percentage': mae_percentage
    }
    
    return metrics

def plot_predictions(y_true, y_pred, title="Model Predictions", save_path=None, n_samples=200):
    """
    Plot true vs predicted values
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        title (str): Plot title
        save_path (str): Path to save plot
        n_samples (int): Number of samples to plot
    """
    plt.figure(figsize=(15, 8))
    
    # Flatten arrays and take subset for visualization
    y_true_flat = y_true.flatten()[:n_samples]
    y_pred_flat = y_pred.flatten()[:n_samples]
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true_flat, label='True Values', alpha=0.7, linewidth=1)
    plt.plot(y_pred_flat, label='Predictions', alpha=0.7, linewidth=1)
    plt.title(f'{title} - Time Series Comparison')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=10)
    plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{title} - Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to {save_path}")
    
    plt.show()

def save_model_artifacts(model, scaler, target_scaler, config, save_dir):
    """Save model and associated artifacts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'transformer_model.h5')
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Save scalers
    scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    target_scaler_path = os.path.join(save_dir, 'target_scaler.pkl')
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    
    # Save configuration
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"✅ Model artifacts saved to {save_dir}")

def load_model_artifacts(load_dir):
    """Load model and associated artifacts"""
    import tensorflow as tf
    
    # Import custom classes for loading
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.transformer_model import TimeSeriesTransformer, TransformerEncoderBlock, PositionalEncoding
    
    # Load model with custom objects
    model_path = os.path.join(load_dir, 'transformer_model.h5')
    custom_objects = {
        'TimeSeriesTransformer': TimeSeriesTransformer,
        'TransformerEncoderBlock': TransformerEncoderBlock,
        'PositionalEncoding': PositionalEncoding
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Load scalers
    scaler_path = os.path.join(load_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    target_scaler_path = os.path.join(load_dir, 'target_scaler.pkl')
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load configuration
    config_path = os.path.join(load_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Model artifacts loaded from {load_dir}")
    return model, scaler, target_scaler, config

def print_metrics(metrics, title="Model Performance"):
    """Print formatted metrics"""
    print(f"\n{title}")
    print("=" * len(title))
    for metric, value in metrics.items():
        if value == float('inf'):
            print(f"{metric}: ∞")
        else:
            print(f"{metric}: {value:.6f}")

def get_feature_columns(data, exclude_cols=None):
    """Get feature columns excluding specified columns"""
    if exclude_cols is None:
        exclude_cols = ['date', 'close']  # Default exclusions
    
    # Get only numeric columns to avoid timestamp errors
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_columns if col not in exclude_cols]
    
    print(f"🔍 Selected {len(feature_cols)} feature columns from {len(numeric_columns)} numeric columns")
    return feature_cols

def check_data_quality(data, target_col='close'):
    """Check data quality and report issues"""
    print("🔍 Checking data quality...")
    
    issues = []
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    if missing_counts.sum() > 0:
        issues.append(f"Missing values: {missing_counts.sum()} total")
    
    # Check for infinite values
    inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        issues.append(f"Infinite values: {inf_counts.sum()} total")
    
    # Check for duplicate dates
    if 'date' in data.columns:
        duplicates = data['date'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate dates: {duplicates}")
    
    # Check target variable
    if target_col in data.columns:
        target_issues = []
        if data[target_col].isnull().sum() > 0:
            target_issues.append(f"{data[target_col].isnull().sum()} missing values")
        if (data[target_col] <= 0).sum() > 0:
            target_issues.append(f"{(data[target_col] <= 0).sum()} non-positive values")
        
        if target_issues:
            issues.append(f"Target column issues: {', '.join(target_issues)}")
    
    if issues:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No data quality issues found")
    
    return issues