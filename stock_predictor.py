#!/usr/bin/env python3
"""
UNIFIED Stock Price Prediction System
Replaces all the redundant models and scripts with one clean system
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add current directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIG
from models.transformer_model import build_transformer_model
from scripts.data_collection import download_stock_data, calculate_technical_indicators

class StockPredictor:
    """
    Unified Stock Price Prediction System
    Handles everything: data collection, training, prediction
    """
    
    def __init__(self, symbol='AAPL'):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.sequence_length = 24
        self.model_path = 'models/unified_model.h5'
        
    def collect_data(self, start_date='2020-01-01', end_date=None):
        """Collect and prepare stock data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"📊 Collecting {self.symbol} data from {start_date} to {end_date}...")
        
        # Download data
        data = download_stock_data(self.symbol, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data found for {self.symbol}")
            
        # Add technical indicators
        data = calculate_technical_indicators(data)
        
        # Remove NaN values
        data = data.dropna()
        
        print(f"✅ Collected {len(data)} rows with {len(data.columns)} features")
        return data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        print("🔄 Preparing data for training...")
        
        # Select features (exclude date and non-numeric columns)
        feature_cols = []
        for col in data.columns:
            if col != 'date' and data[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
        
        # Remove target from features
        if 'close' in feature_cols:
            feature_cols.remove('close')
            
        print(f"📋 Using {len(feature_cols)} features: {feature_cols[:5]}...")
        
        # Create sequences
        X, y = self._create_sequences(data, feature_cols, 'close')
        
        # Split data
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Scale data
        from sklearn.preprocessing import MinMaxScaler
        
        # Feature scaling
        self.scaler = MinMaxScaler()
        X_train_scaled = self._scale_sequences(X_train, fit=True)
        X_val_scaled = self._scale_sequences(X_val, fit=False)
        X_test_scaled = self._scale_sequences(X_test, fit=False)
        
        # Target scaling
        self.target_scaler = MinMaxScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        return {
            'X_train': X_train_scaled, 'y_train': y_train_scaled,
            'X_val': X_val_scaled, 'y_val': y_val_scaled,
            'X_test': X_test_scaled, 'y_test': y_test_scaled,
            'feature_cols': feature_cols
        }
    
    def _create_sequences(self, data, feature_cols, target_col):
        """Create sequences for time series prediction"""
        features = data[feature_cols].values
        target = data[target_col].values
        
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(features[i-self.sequence_length:i])
            y.append(target[i])
            
        return np.array(X), np.array(y)
    
    def _scale_sequences(self, X, fit=False):
        """Scale sequence data"""
        batch_size, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
            
        return X_scaled.reshape(batch_size, seq_len, n_features)
    
    def train(self, data=None, epochs=100, clean_start=True):
        """Train the model"""
        if clean_start:
            self._cleanup_old_files()
            
        if data is None:
            data = self.collect_data()
            
        print("🚀 Starting model training...")
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Build model
        model_config = MODEL_CONFIG.copy()
        self.model = build_transformer_model(
            sequence_length=self.sequence_length,
            num_features=prepared_data['X_train'].shape[2],
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            ff_dim=model_config['ff_dim'],
            dropout_rate=model_config['dropout_rate'],
            forecast_horizon=model_config['forecast_horizon']
        )
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True)
        ]
        
        # Train
        history = self.model.fit(
            prepared_data['X_train'], prepared_data['y_train'],
            validation_data=(prepared_data['X_val'], prepared_data['y_val']),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss = self.model.evaluate(prepared_data['X_test'], prepared_data['y_test'], verbose=0)
        
        print(f"\n🎉 Training completed!")
        print(f"   Final loss: {history.history['loss'][-1]:.6f}")
        print(f"   Best val loss: {min(history.history['val_loss']):.6f}")
        print(f"   Test loss: {test_loss[0]:.6f}")
        
        # Save training history
        self._save_results(history, prepared_data, test_loss)
        
        return history
    
    def predict_future(self, days=7):
        """Predict future stock prices"""
        if self.model is None:
            self._load_model()
            
        print(f"🔮 Predicting {self.symbol} prices for next {days} days...")
        
        # Get recent data with more history for technical indicators
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')  # More history needed
        
        recent_data = self.collect_data(start_date, end_date)
        
        if len(recent_data) < self.sequence_length + 50:  # Need enough for tech indicators
            print(f"⚠️  Not enough recent data ({len(recent_data)} rows). Using longer history...")
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year
            recent_data = self.collect_data(start_date, end_date)
        
        # Prepare last sequence
        feature_cols = [col for col in recent_data.columns 
                       if col not in ['date', 'close'] and recent_data[col].dtype in ['float64', 'int64']]
        
        last_sequence = recent_data[feature_cols].values[-self.sequence_length:]
        # Scale using the same approach as training: transform the feature matrix directly
        last_sequence_scaled = self.scaler.transform(last_sequence)
        last_sequence = last_sequence_scaled.reshape(1, self.sequence_length, -1)
        
        # Make predictions
        predictions = []
        current_seq = last_sequence.copy()
        
        for i in range(days):
            pred_scaled = self.model.predict(current_seq, verbose=0)
            pred_price = self.target_scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]
            predictions.append(pred_price)
            
            # Update sequence for next prediction (simple approach)
            current_seq = np.roll(current_seq, -1, axis=1)
            # Update last timestep with prediction (simplified)
            current_seq[0, -1, 0] = pred_scaled[0][0]  # Assuming close price is first feature
        
        # Create prediction dates
        last_date = pd.to_datetime(recent_data['date'].iloc[-1])
        pred_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Create results
        results = pd.DataFrame({
            'date': pred_dates,
            'predicted_price': predictions
        })
        
        current_price = recent_data['close'].iloc[-1]
        print(f"📊 Prediction Results:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Predicted Price ({days} days): ${predictions[-1]:.2f}")
        print(f"   Expected Change: {((predictions[-1] - current_price) / current_price * 100):.2f}%")
        
        # Plot results
        self._plot_predictions(recent_data, results)
        
        return results
    
    def _load_model(self):
        """Load saved model and scalers"""
        print("📂 Loading saved model...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("No trained model found. Please train first.")
        
        try:
            # Import custom objects for loading
            from models.transformer_model import (TimeSeriesTransformer, TransformerEncoderBlock, 
                                                PositionalEncoding, LearnedPositionalEmbedding,
                                                RoPEMultiHeadSelfAttention, RoPETransformerEncoderBlock,
                                                AdvancedTransformerEncoderBlock, AdvancedRoPEMultiHeadSelfAttention)
            
            # Load model with custom objects
            custom_objects = {
                'TimeSeriesTransformer': TimeSeriesTransformer,
                'TransformerEncoderBlock': TransformerEncoderBlock,
                'PositionalEncoding': PositionalEncoding,
                'LearnedPositionalEmbedding': LearnedPositionalEmbedding,
                'RoPEMultiHeadSelfAttention': RoPEMultiHeadSelfAttention,
                'RoPETransformerEncoderBlock': RoPETransformerEncoderBlock,
                'AdvancedTransformerEncoderBlock': AdvancedTransformerEncoderBlock,
                'AdvancedRoPEMultiHeadSelfAttention': AdvancedRoPEMultiHeadSelfAttention
            }
            
            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
            
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("🔄 Rebuilding model architecture...")
            
            # Rebuild model architecture and load weights
            # Get recent data to determine feature count
            recent_data = self.collect_data(start_date='2024-01-01')
            feature_cols = [col for col in recent_data.columns 
                           if col not in ['date', 'close'] and recent_data[col].dtype in ['float64', 'int64']]
            
            # Rebuild model
            model_config = MODEL_CONFIG.copy()
            self.model = build_transformer_model(
                sequence_length=self.sequence_length,
                num_features=len(feature_cols),
                d_model=model_config['d_model'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                ff_dim=model_config['ff_dim'],
                dropout_rate=model_config['dropout_rate'],
                forecast_horizon=model_config['forecast_horizon'],
                pos_encoding_type=model_config.get('pos_encoding_type', 'learned')
            )
            
            # Load only the weights
            try:
                self.model.load_weights(self.model_path)
            except Exception as weight_error:
                raise RuntimeError(f"Cannot load model weights: {weight_error}. Please retrain the model.")
        
        # Load scalers
        import pickle
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('models/target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
            
        print("✅ Model and scalers loaded")
    
    def _save_results(self, history, data, test_loss):
        """Save training results"""
        # Save scalers
        import pickle
        os.makedirs('models', exist_ok=True)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('models/target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        # Save history
        os.makedirs('results', exist_ok=True)
        history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
        
        with open('results/training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
            
        # Plot training history
        self._plot_training_history(history)
        
        print("💾 Results saved to results/ and models/")
    
    def _plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Training MAE')
            ax2.plot(history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, recent_data, predictions):
        """Plot prediction results"""
        plt.figure(figsize=(12, 6))
        
        # Plot recent prices
        recent_dates = pd.to_datetime(recent_data['date'])
        plt.plot(recent_dates[-30:], recent_data['close'].values[-30:], 
                label='Historical Prices', color='blue', linewidth=2)
        
        # Plot predictions
        plt.plot(predictions['date'], predictions['predicted_price'], 
                label='Predicted Prices', color='red', linewidth=2, linestyle='--', marker='o')
        
        plt.title(f'{self.symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('results/price_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Prediction plot saved to results/price_predictions.png")
    
    def _cleanup_old_files(self):
        """Remove old result files"""
        files_to_remove = [
            'results/training_history.json',
            'results/training_history.png', 
            'results/predictions.png',
            'results/price_predictions.png',
            'models/best_transformer.h5',
            'models/unified_model.h5'
        ]
        
        removed = 0
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                removed += 1
                
        if removed > 0:
            print(f"🧹 Cleaned {removed} old result files")

def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Stock Price Prediction System')
    parser.add_argument('action', choices=['train', 'predict'], help='Action to perform')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=7, help='Days to predict (default: 7)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    
    args = parser.parse_args()
    
    predictor = StockPredictor(symbol=args.symbol)
    
    if args.action == 'train':
        predictor.train(epochs=args.epochs)
    elif args.action == 'predict':
        predictions = predictor.predict_future(days=args.days)
        print("\n📋 Prediction Summary:")
        print(predictions.to_string(index=False))

if __name__ == "__main__":
    # If no command line args, run interactive mode
    import sys
    if len(sys.argv) == 1:
        print("🚀 Unified Stock Prediction System")
        print("=" * 50)
        
        predictor = StockPredictor('AAPL')
        
        print("1. Training model...")
        predictor.train(epochs=50)  # Shorter for demo
        
        print("\n2. Making predictions...")
        predictions = predictor.predict_future(days=5)
        
        print("\n🎉 Demo completed! Check results/ folder for plots.")
    else:
        main()