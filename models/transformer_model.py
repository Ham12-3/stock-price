#!/usr/bin/env python3
"""
Transformer Model for Time Series Forecasting
Implements a Transformer architecture adapted for stock price prediction
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, MODEL_CONFIG

class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer for time series data
    Adds positional information to the input embeddings
    """
    
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        self.pos_encoding = self.get_positional_encoding(sequence_length, d_model)
    
    def get_angles(self, position, i, d_model):
        """Calculate angles for positional encoding"""
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def get_positional_encoding(self, sequence_length, d_model):
        """Generate positional encoding matrix"""
        # Create position indices
        position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        
        # Create dimension indices
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        
        # Calculate angles
        angle_rads = self.get_angles(position, i, d_model)
        
        # Apply sin to even indices (0, 2, 4...)
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices (1, 3, 5...)
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # Interleave sines and cosines
        if d_model % 2 == 0:
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            # Reshape to interleave properly
            pos_encoding = tf.stack([sines, cosines], axis=2)
            pos_encoding = tf.reshape(pos_encoding, [sequence_length, d_model])
        else:
            # Handle odd d_model
            pos_encoding = tf.concat([sines, cosines[:, :-1]], axis=-1)
            pos_encoding = tf.stack([sines, cosines[:, :-1]], axis=2)
            pos_encoding = tf.reshape(pos_encoding, [sequence_length, d_model - 1])
            # Add the last cosine value
            pos_encoding = tf.concat([pos_encoding, cosines[:, -1:]], axis=-1)
        
        # Add batch dimension
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

class TransformerEncoderBlock(layers.Layer):
    """
    Transformer encoder block with multi-head attention and feed-forward network
    """
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False, mask=None):
        """Forward pass through encoder block"""
        # Multi-head attention
        attn_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

class TimeSeriesTransformer(Model):
    """
    Complete Transformer model for time series forecasting
    """
    
    def __init__(self, 
                 sequence_length, 
                 num_features, 
                 d_model=64, 
                 num_heads=8, 
                 num_layers=4, 
                 ff_dim=128, 
                 dropout_rate=0.1,
                 forecast_horizon=1,
                 **kwargs):
        super(TimeSeriesTransformer, self).__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        
        # Input projection layer (project features to d_model dimensions)
        self.input_projection = layers.Dense(d_model, name='input_projection')
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Transformer encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f'encoder_block_{i}'
            ) for i in range(num_layers)
        ]
        
        # Global pooling
        self.global_pooling = layers.GlobalAveragePooling1D(name='global_pooling')
        
        # Output layers
        self.dropout = layers.Dropout(dropout_rate, name='output_dropout')
        self.dense1 = layers.Dense(64, activation="relu", name='dense1')
        self.dense2 = layers.Dense(32, activation="relu", name='dense2')
        self.output_layer = layers.Dense(forecast_horizon, name='output')
    
    def call(self, inputs, training=False, mask=None):
        """Forward pass through the model"""
        # Input projection to d_model dimensions
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training, mask=mask)
        
        # Global pooling to get fixed-size representation
        x = self.global_pooling(x)
        
        # Output layers
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self):
        return {
            'sequence_length': self.sequence_length,
            'num_features': self.num_features,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'forecast_horizon': self.forecast_horizon
        }

def build_transformer_model(sequence_length, num_features, compile_model=True, **kwargs):
    """
    Build and optionally compile the Transformer model
    
    Args:
        sequence_length (int): Length of input sequences
        num_features (int): Number of input features
        compile_model (bool): Whether to compile the model
        **kwargs: Additional model parameters
        
    Returns:
        TimeSeriesTransformer: Built (and optionally compiled) model
    """
    print("🏗️  Building Transformer model...")
    
    # Get default configuration
    config = MODEL_CONFIG.copy()
    config.update(kwargs)
    
    # Build model
    model = TimeSeriesTransformer(
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        dropout_rate=config['dropout_rate'],
        forecast_horizon=config['forecast_horizon']
    )
    
    # Build the model by calling it with dummy data
    dummy_input = tf.random.normal((1, sequence_length, num_features))
    _ = model(dummy_input)
    
    if compile_model:
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        print("✅ Model compiled with MSE loss and MAE metric")
    
    # Print model summary
    print(f"✅ Transformer model built successfully")
    print(f"   - Sequence length: {sequence_length}")
    print(f"   - Number of features: {num_features}")
    print(f"   - Model dimension: {config['d_model']}")
    print(f"   - Number of heads: {config['num_heads']}")
    print(f"   - Number of layers: {config['num_layers']}")
    print(f"   - Feed-forward dimension: {config['ff_dim']}")
    print(f"   - Dropout rate: {config['dropout_rate']}")
    print(f"   - Forecast horizon: {config['forecast_horizon']}")
    
    return model

def create_training_callbacks(model_save_path, patience=15, reduce_lr_patience=5):
    """
    Create training callbacks for model training
    
    Args:
        model_save_path (str): Path to save the best model
        patience (int): Early stopping patience
        reduce_lr_patience (int): Learning rate reduction patience
        
    Returns:
        list: List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
    ]
    
    print(f"✅ Created training callbacks:")
    print(f"   - Early stopping patience: {patience}")
    print(f"   - Learning rate reduction patience: {reduce_lr_patience}")
    print(f"   - Model checkpoint path: {model_save_path}")
    
    return callbacks

def print_model_summary(model, input_shape=None):
    """Print detailed model summary"""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    
    if input_shape:
        model.build(input_shape)
    
    model.summary()
    
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    
    print("="*50)

def get_attention_weights(model, inputs):
    """
    Extract attention weights from the model
    This is useful for interpretability
    
    Args:
        model: Trained Transformer model
        inputs: Input sequences
        
    Returns:
        list: Attention weights from each encoder block
    """
    attention_weights = []
    
    # Get intermediate outputs from each encoder block
    x = model.input_projection(inputs)
    x = model.pos_encoding(x)
    
    for i, encoder_block in enumerate(model.encoder_blocks):
        # For simplicity, we'll return the final output
        # In a full implementation, you'd modify the encoder blocks
        # to return attention weights
        x = encoder_block(x)
    
    # Note: Getting actual attention weights requires modifying the 
    # MultiHeadAttention layer to return weights, which is more complex
    print("⚠️  Attention weight extraction requires model modification")
    print("   This is a placeholder for attention visualization")
    
    return attention_weights

def main():
    """Test the model building"""
    print("🧪 Testing Transformer model building...")
    
    # Test parameters
    sequence_length = 24
    num_features = 10
    
    # Build model
    model = build_transformer_model(
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=32,  # Smaller for testing
        num_heads=4,
        num_layers=2
    )
    
    # Print summary
    print_model_summary(model, input_shape=(None, sequence_length, num_features))
    
    # Test forward pass
    test_input = tf.random.normal((2, sequence_length, num_features))
    output = model(test_input)
    
    print(f"\n✅ Test completed successfully!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")

if __name__ == "__main__":
    main()