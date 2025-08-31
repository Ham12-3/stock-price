#!/usr/bin/env python3
"""
Training Script for Stock Price Forecasting Transformer Model
Orchestrates the complete training pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, MODEL_CONFIG, TRAINING_CONFIG
from models.transformer_model import build_transformer_model, create_training_callbacks, print_model_summary
from utils.utils import calculate_metrics, plot_predictions, plot_training_history, save_model_artifacts

def load_preprocessed_data(sequences_path=None, metadata_path=None):
    """
    Load preprocessed data from files
    
    Args:
        sequences_path (str): Path to sequences file
        metadata_path (str): Path to metadata file
        
    Returns:
        tuple: Loaded data and metadata
    """
    if sequences_path is None:
        from config import FILE_PATHS
        sequences_path = FILE_PATHS['processed_data'].replace('.csv', '_sequences.npz')
    
    if metadata_path is None:
        from config import FILE_PATHS
        metadata_path = FILE_PATHS['processed_data'].replace('.csv', '_metadata.json')
    
    print(f"📂 Loading preprocessed data from {sequences_path}...")
    
    # Load sequences
    data = np.load(sequences_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    loaded_data = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
        'X_test': data['X_test'],
        'y_test': data['y_test'],
        'metadata': metadata
    }
    
    print(f"✅ Data loaded successfully")
    print(f"   Training set: {loaded_data['X_train'].shape}")
    print(f"   Validation set: {loaded_data['X_val'].shape}")
    print(f"   Test set: {loaded_data['X_test'].shape}")
    
    return loaded_data

def train_model(X_train, y_train, X_val, y_val, model_config=None, training_config=None):
    """
    Train the Transformer model
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_config (dict): Model configuration
        training_config (dict): Training configuration
        
    Returns:
        tuple: (trained_model, training_history)
    """
    if model_config is None:
        model_config = MODEL_CONFIG
    
    if training_config is None:
        training_config = TRAINING_CONFIG
    
    print("🚀 Starting model training...")
    print("=" * 50)
    
    # Get data dimensions
    sequence_length = X_train.shape[1]
    num_features = X_train.shape[2]
    
    print(f"Training configuration:")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of features: {num_features}")
    print(f"   Batch size: {model_config['batch_size']}")
    print(f"   Learning rate: {model_config['learning_rate']}")
    print(f"   Max epochs: {model_config['epochs']}")
    
    # Build model
    model = build_transformer_model(
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        ff_dim=model_config['ff_dim'],
        dropout_rate=model_config['dropout_rate'],
        forecast_horizon=model_config['forecast_horizon'],
        learning_rate=model_config['learning_rate']
    )
    
    # Print model summary
    print_model_summary(model, input_shape=(None, sequence_length, num_features))
    
    # Create callbacks
    model_save_path = training_config['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = create_training_callbacks(
        model_save_path=model_save_path,
        patience=model_config['early_stopping_patience'],
        reduce_lr_patience=model_config['reduce_lr_patience']
    )
    
    # Train model
    print(f"\n🏋️  Training model...")
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Training completed in {training_time}")
    
    # Save training history
    if training_config['save_training_history']:
        history_path = training_config['history_save_path']
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        # Convert history to JSON-serializable format
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"💾 Training history saved to {history_path}")
    
    return model, history

def evaluate_model(model, X_test, y_test, target_scaler=None):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        target_scaler: Target scaler for inverse transform
        
    Returns:
        dict: Evaluation metrics
    """
    print("📊 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, scaler=target_scaler)
    
    # Print metrics
    print("\n📈 Model Performance on Test Set:")
    print("=" * 40)
    for metric, value in metrics.items():
        if value == float('inf'):
            print(f"{metric}: ∞")
        else:
            print(f"{metric}: {value:.6f}")
    
    return metrics, y_pred

def create_synthetic_data(num_samples=1000, sequence_length=24, num_features=10):
    """
    Create synthetic data for testing when real data isn't available
    
    Args:
        num_samples (int): Number of samples to generate
        sequence_length (int): Length of each sequence
        num_features (int): Number of features
        
    Returns:
        dict: Synthetic data in the same format as preprocessed data
    """
    print(f"🎲 Creating synthetic data for testing...")
    print(f"   Samples: {num_samples}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Features: {num_features}")
    
    # Generate synthetic sequences with some pattern
    np.random.seed(42)
    
    # Create base time series with trend and seasonality
    time = np.arange(num_samples + sequence_length)
    trend = time * 0.01
    seasonality = np.sin(time * 2 * np.pi / 50) * 0.5
    noise = np.random.normal(0, 0.1, len(time))
    base_series = trend + seasonality + noise + 100  # Stock price-like values
    
    # Create features (technical indicators, lags, etc.)
    X, y = [], []
    
    for i in range(num_samples):
        # Input sequence
        sequence = np.zeros((sequence_length, num_features))
        
        # Feature 0: Price (main feature)
        sequence[:, 0] = base_series[i:i+sequence_length]
        
        # Feature 1: Price change
        sequence[1:, 1] = np.diff(sequence[:, 0])
        
        # Features 2-5: Simple moving averages
        for j, window in enumerate([5, 10, 15, 20]):
            if sequence_length >= window:
                for k in range(window-1, sequence_length):
                    sequence[k, 2+j] = np.mean(sequence[k-window+1:k+1, 0])
        
        # Features 6-9: Random technical indicators
        for j in range(4):
            sequence[:, 6+j] = np.random.normal(0, 1, sequence_length)
        
        X.append(sequence)
        # Target: next day's price
        y.append([base_series[i + sequence_length]])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/val/test
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)
    
    synthetic_data = {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
        'metadata': {
            'num_features': num_features,
            'sequence_length': sequence_length,
            'forecast_horizon': 1,
            'target_column': 'synthetic_price',
            'feature_columns': [f'feature_{i}' for i in range(num_features)]
        }
    }
    
    print(f"✅ Synthetic data created")
    print(f"   Training set: {synthetic_data['X_train'].shape}")
    print(f"   Validation set: {synthetic_data['X_val'].shape}")
    print(f"   Test set: {synthetic_data['X_test'].shape}")
    
    return synthetic_data

def main(use_synthetic_data=True):
    """
    Main training function
    
    Args:
        use_synthetic_data (bool): Whether to use synthetic data for testing
    """
    print("🚀 Starting Stock Price Forecasting Training Pipeline")
    print("=" * 60)
    
    try:
        # Load or create data
        if use_synthetic_data:
            print("⚠️  Using synthetic data for demonstration")
            data = create_synthetic_data(
                num_samples=2000,
                sequence_length=MODEL_CONFIG['sequence_length'],
                num_features=15
            )
        else:
            # Load real preprocessed data
            data = load_preprocessed_data()
        
        # Extract data
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Train model
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        # Plot training history
        plot_training_history(history, save_path='results/training_history.png')
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Plot predictions
        plot_predictions(
            y_test, y_pred,
            title="Transformer Model Predictions",
            save_path='results/predictions.png',
            n_samples=200
        )
        
        # Save model and results
        config = {
            'model_config': MODEL_CONFIG,
            'training_config': TRAINING_CONFIG,
            'metadata': data['metadata'],
            'metrics': metrics,
            'training_time': str(datetime.now())
        }
        
        # Note: In a real implementation, you would save the actual scalers
        # For synthetic data, we create dummy scalers
        from sklearn.preprocessing import MinMaxScaler
        dummy_scaler = MinMaxScaler()
        dummy_target_scaler = MinMaxScaler()
        
        # Fit dummy scalers with some data to make them serializable
        dummy_data = np.array([[0], [1]])
        dummy_scaler.fit(dummy_data)
        dummy_target_scaler.fit(dummy_data)
        
        save_model_artifacts(
            model=model,
            scaler=dummy_scaler,
            target_scaler=dummy_target_scaler,
            config=config,
            save_dir='models/saved_model'
        )
        
        print("\n🏆 Training pipeline completed successfully!")
        print("=" * 60)
        print("FINAL RESULTS:")
        print(f"   Best validation loss: {min(history.history['val_loss']):.6f}")
        print(f"   Test RMSE: {metrics['RMSE']:.6f}")
        print(f"   Test MAE: {metrics['MAE']:.6f}")
        print(f"   Test MAPE: {metrics['MAPE']:.2f}%")
        
        return model, history, metrics
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    # Run with real data for production use
    trained_model, training_history, final_metrics = main(use_synthetic_data=False)