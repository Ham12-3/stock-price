# Detailed Implementation Guide: Time Series Forecasting with External Influences

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Collection & Exploration](#data-collection--exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Transformer Architecture Implementation](#transformer-architecture-implementation)
5. [Model Training](#model-training)
6. [Evaluation & Comparison](#evaluation--comparison)
7. [Deployment & Inference](#deployment--inference)

---

## Environment Setup

### Step 1: Create Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

---

## Data Collection & Exploration

### Step 1: Choose Dataset
**Option A: Stock Data with Technical Indicators**
- Use `yfinance` for stock prices
- Include technical indicators (RSI, MACD, Bollinger Bands)
- Add market sentiment data

**Option B: Energy Load with Weather Data**
- PJM hourly energy consumption
- Temperature, humidity, wind speed
- Holiday and calendar features

### Step 2: Data Loading Script
```python
# scripts/data_collection.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_stock_data(symbol, start_date, end_date):
    """Collect stock data with technical indicators"""
    # Download stock data
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    
    # Add technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line
```

### Step 3: Data Exploration
```python
# notebooks/01_data_exploration.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data):
    """Comprehensive data exploration"""
    print("Dataset Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Plot time series
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Close'])
    plt.title('Stock Price Over Time')
    plt.ylabel('Price ($)')
    
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['Volume'])
    plt.title('Trading Volume Over Time')
    plt.ylabel('Volume')
    
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['RSI'])
    plt.title('RSI Indicator Over Time')
    plt.ylabel('RSI')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    
    plt.tight_layout()
    plt.show()
```

---

## Data Preprocessing

### Step 1: Handle Missing Values & Outliers
```python
# scripts/data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def clean_data(data):
    """Clean and prepare data for modeling"""
    # Handle missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Remove outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return data

def create_features(data, target_col='Close'):
    """Create additional features for the model"""
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        data[f'{target_col}_sma_{window}'] = data[target_col].rolling(window=window).mean()
        data[f'{target_col}_std_{window}'] = data[target_col].rolling(window=window).std()
    
    # Price changes
    data['price_change'] = data[target_col].pct_change()
    data['price_change_abs'] = data['price_change'].abs()
    
    # Volatility
    data['volatility'] = data['price_change'].rolling(window=20).std()
    
    return data

def prepare_sequences(data, target_col, feature_cols, sequence_length=24, forecast_horizon=1):
    """Create sequences for time series modeling"""
    # Remove rows with NaN values
    data = data.dropna()
    
    # Prepare features and target
    features = data[feature_cols].values
    target = data[target_col].values
    
    X, y = [], []
    
    for i in range(sequence_length, len(data) - forecast_horizon + 1):
        X.append(features[i-sequence_length:i])
        y.append(target[i:i+forecast_horizon])
    
    return np.array(X), np.array(y)

def scale_data(X_train, X_val, X_test, scaler_type='minmax'):
    """Scale the data using specified scaler"""
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

### Step 2: Data Splitting
```python
def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

---

## Transformer Architecture Implementation

### Step 1: Positional Encoding
```python
# models/transformer_model.py
import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    """Positional encoding layer for time series data"""
    
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
    
    def get_angles(self, position, i, d_model):
        """Calculate angles for positional encoding"""
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        """Generate positional encoding matrix"""
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
```

### Step 2: Transformer Encoder Block
```python
class TransformerEncoderBlock(layers.Layer):
    """Transformer encoder block with multi-head attention"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training):
        """Forward pass through encoder block"""
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)
```

### Step 3: Complete Transformer Model
```python
class TimeSeriesTransformer(tf.keras.Model):
    """Complete Transformer model for time series forecasting"""
    
    def __init__(self, 
                 sequence_length, 
                 num_features, 
                 d_model=64, 
                 num_heads=8, 
                 num_layers=4, 
                 ff_dim=128, 
                 dropout_rate=0.1,
                 forecast_horizon=1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        # Input projection layer
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Transformer encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Global pooling and output layers
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.output_layer = layers.Dense(forecast_horizon)
    
    def call(self, inputs, training=False):
        """Forward pass through the model"""
        # Input projection
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
        
        # Global pooling
        x = self.global_pooling(x)
        
        # Output layers
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        
        return self.output_layer(x)

def build_transformer_model(sequence_length, num_features, **kwargs):
    """Build and compile the Transformer model"""
    model = TimeSeriesTransformer(
        sequence_length=sequence_length,
        num_features=num_features,
        **kwargs
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

---

## Model Training

### Step 1: Training Configuration
```python
# scripts/training.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def create_callbacks(model_save_path):
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, 
                batch_size=32, epochs=100, callbacks=None):
    """Train the Transformer model"""
    
    # Training history
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

### Step 2: Training Execution
```python
def main_training():
    """Main training function"""
    # Load preprocessed data
    X_train, y_train = load_preprocessed_data('train')
    X_val, y_val = load_preprocessed_data('val')
    
    # Model parameters
    sequence_length = 24
    num_features = X_train.shape[-1]
    
    # Build model
    model = build_transformer_model(
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=64,
        num_heads=8,
        num_layers=4,
        ff_dim=128,
        dropout_rate=0.1,
        forecast_horizon=1
    )
    
    # Create callbacks
    callbacks = create_callbacks('models/best_transformer.h5')
    
    # Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=32,
        epochs=100,
        callbacks=callbacks
    )
    
    # Save training history
    save_training_history(history, 'results/training_history.json')
    
    return model, history
```

---

## Evaluation & Comparison

### Step 1: Model Evaluation
```python
# scripts/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_test_original = y_test.flatten()
        y_pred_original = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    print("Model Performance Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results, y_pred_original
```

### Step 2: Baseline Models
```python
# models/baseline_models.py
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_arima_model(data, order=(1, 1, 1)):
    """Build and train ARIMA model"""
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

def build_lstm_model(sequence_length, num_features):
    """Build LSTM baseline model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def compare_models(transformer_results, baseline_results):
    """Compare different model performances"""
    comparison_df = pd.DataFrame({
        'Model': ['Transformer', 'ARIMA', 'LSTM'],
        'RMSE': [transformer_results['RMSE'], 
                baseline_results['ARIMA']['RMSE'], 
                baseline_results['LSTM']['RMSE']],
        'MAE': [transformer_results['MAE'], 
               baseline_results['ARIMA']['MAE'], 
               baseline_results['LSTM']['MAE']]
    })
    
    print("Model Comparison:")
    print(comparison_df)
    
    return comparison_df
```

---

## Deployment & Inference

### Step 1: Model Saving & Loading
```python
def save_model_pipeline(model, scaler, config, save_path):
    """Save the complete model pipeline"""
    # Save model
    model.save(f"{save_path}/transformer_model.h5")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, f"{save_path}/scaler.pkl")
    
    # Save configuration
    import json
    with open(f"{save_path}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

def load_model_pipeline(load_path):
    """Load the complete model pipeline"""
    # Load model
    model = tf.keras.models.load_model(f"{load_path}/transformer_model.h5")
    
    # Load scaler
    import joblib
    scaler = joblib.load(f"{load_path}/scaler.pkl")
    
    # Load configuration
    import json
    with open(f"{load_path}/config.json", 'r') as f:
        config = json.load(f)
    
    return model, scaler, config
```

### Step 2: Inference Script
```python
def make_prediction(model, scaler, new_data, sequence_length):
    """Make predictions on new data"""
    # Preprocess new data
    if len(new_data) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} data points")
    
    # Create sequence
    sequence = new_data[-sequence_length:]
    
    # Scale data
    sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
    
    # Reshape for model input
    X = sequence_scaled.reshape(1, sequence_length, -1)
    
    # Make prediction
    prediction_scaled = model.predict(X)
    
    # Inverse transform
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    return prediction.flatten()

def main_inference():
    """Main inference function"""
    # Load model pipeline
    model, scaler, config = load_model_pipeline('models/saved_model')
    
    # Load new data (example)
    new_data = load_new_data()
    
    # Make prediction
    prediction = make_prediction(
        model, scaler, new_data, 
        sequence_length=config['sequence_length']
    )
    
    print(f"Predicted value: {prediction[0]:.2f}")
    return prediction
```

---

## Next Steps & Recommendations

1. **Start with a small dataset** to test the pipeline
2. **Experiment with different hyperparameters** (sequence length, model size)
3. **Add more external features** (news sentiment, economic indicators)
4. **Implement attention visualization** for interpretability
5. **Consider ensemble methods** combining multiple models
6. **Add cross-validation** for more robust evaluation
7. **Implement online learning** for real-time updates

## Troubleshooting Common Issues

- **Memory issues**: Reduce batch size or sequence length
- **Overfitting**: Increase dropout or reduce model complexity
- **Training instability**: Adjust learning rate or use gradient clipping
- **Poor performance**: Check data quality and feature engineering

This guide provides a comprehensive roadmap for implementing your time series forecasting project. Follow each step carefully and experiment with different approaches to find what works best for your specific dataset and requirements.
