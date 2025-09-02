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

class LearnedPositionalEmbedding(layers.Layer):
    """
    Learned Positional Embedding - Superior to sine/cosine for fixed sequences
    Learns optimal position representations during training
    """
    
    def __init__(self, max_len, d_model, dropout=0.0, **kwargs):
        super(LearnedPositionalEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        # Learnable position embeddings [max_len, d_model]
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.max_len, self.d_model),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=False):
        # x: [batch, seq, d_model]
        seq_len = tf.shape(x)[1]
        pos = self.pos_emb[:seq_len]            # [seq, d_model]
        pos = tf.expand_dims(pos, axis=0)       # [1, seq, d_model]
        x = x + pos
        return self.dropout(x, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config

def _rotate_half(x):
    """Helper function for RoPE - rotates half the dimensions"""
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([-x2, x1], axis=-1)

def _rope_cache(seq_len, head_dim, base=10000.0):
    """Generate RoPE (Rotary Positional Embedding) frequency cache"""
    # head_dim must be even
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (tf.range(0, half, dtype=tf.float32) / float(half)))
    t = tf.cast(tf.range(seq_len), tf.float32)                    # [seq]
    freqs = tf.einsum('s,d->sd', t, inv_freq)                     # [seq, half]
    emb = tf.concat([freqs, freqs], axis=-1)                      # [seq, head_dim]
    cos = tf.cos(emb)[None, None, :, None, :]                     # [1,1,seq,1,head_dim]
    sin = tf.sin(emb)[None, None, :, None, :]                     # [1,1,seq,1,head_dim]
    return cos, sin

def _apply_rope(x, cos, sin):
    """Apply rotary positional embedding to tensor x"""
    # x: [b, h, s, d]
    return (x * cos) + (_rotate_half(x) * sin)

class RoPEMultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention with Rotary Positional Embeddings (RoPE)
    Used in modern LLMs like GPT-NeoX, PaLM - superior length extrapolation
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0, **kwargs):
        super(RoPEMultiHeadSelfAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        
        self.qkv = layers.Dense(3 * d_model, use_bias=False)
        self.out = layers.Dense(d_model, use_bias=False)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        # x: [b, s, d_model]
        b, s = tf.shape(x)[0], tf.shape(x)[1]
        qkv = self.qkv(x)                                         # [b, s, 3*d]
        q, k, v = tf.split(qkv, 3, axis=-1)

        def split_heads(t):
            t = tf.reshape(t, [b, s, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])                  # [b, h, s, d_h]

        q, k, v = map(split_heads, (q, k, v))

        # Apply RoPE to queries and keys
        cos, sin = _rope_cache(s, self.head_dim)                  # broadcast caches
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = tf.cast(self.head_dim, tf.float32) ** -0.5
        attn_logits = tf.einsum('bhid,bhjd->bhij', q, k) * scale  # [b, h, s, s]

        if mask is not None:
            # mask: [b, 1, 1, s] or [b, s] -> convert to aligned shape
            if tf.rank(mask) == 2:
                mask = tf.cast(mask[:, None, None, :], tf.float32)
            attn_logits += (1.0 - mask) * (-1e9)

        attn = tf.nn.softmax(attn_logits, axis=-1)
        attn = self.dropout(attn, training=training)
        out = tf.einsum('bhij,bhjd->bhid', attn, v)               # [b, h, s, d_h]

        out = tf.transpose(out, [0, 2, 1, 3])                     # [b, s, h, d_h]
        out = tf.reshape(out, [b, s, self.d_model])               # [b, s, d_model]
        return self.out(out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config

# Legacy sine/cosine encoding for backward compatibility
class PositionalEncoding(layers.Layer):
    """
    Legacy Positional encoding - kept for backward compatibility
    Use LearnedPositionalEmbedding or RoPE for better performance
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

class AdvancedRoPEMultiHeadSelfAttention(layers.Layer):
    """
    Enhanced RoPE Multi-Head Self-Attention with proper masking support
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0, **kwargs):
        super(AdvancedRoPEMultiHeadSelfAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        
        self.qkv = layers.Dense(3 * d_model, use_bias=False)
        self.out = layers.Dense(d_model, use_bias=False)
        self.drop = layers.Dropout(dropout)

    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the dimensions"""
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([-x2, x1], axis=-1)

    def _rope_cache(self, s, d, base=10000.0):
        """Generate RoPE frequency cache"""
        half = d // 2
        inv_freq = 1.0 / (base ** (tf.range(0, half, dtype=tf.float32) / float(half)))
        t = tf.cast(tf.range(s), tf.float32)
        freqs = tf.einsum('s,d->sd', t, inv_freq)    # [s, half]
        emb = tf.concat([freqs, freqs], axis=-1)      # [s, d]
        cos = tf.cos(emb)[None, None, :, None, :]     # [1,1,s,1,d]
        sin = tf.sin(emb)[None, None, :, None, :]
        return cos, sin

    def _apply_rope(self, x, cos, sin):
        """Apply rotary positional embedding"""
        return (x * cos) + (self._rotate_half(x) * sin)

    def call(self, x, attention_mask=None, training=False):
        b = tf.shape(x)[0]
        s = tf.shape(x)[1]

        qkv = self.qkv(x)                              # [b,s,3*d]
        q, k, v = tf.split(qkv, 3, axis=-1)

        def split_heads(t):
            t = tf.reshape(t, [b, s, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])      # [b,h,s,d_h]

        q, k, v = map(split_heads, (q, k, v))

        # Apply RoPE to queries and keys
        cos, sin = self._rope_cache(s, self.head_dim)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = tf.cast(self.head_dim, tf.float32) ** -0.5
        logits = tf.einsum('bhid,bhjd->bhij', q, k) * scale

        # Apply attention mask
        if attention_mask is not None:
            # Expect boolean [b,s,s]; expand to [b,1,s,s] if needed
            if tf.rank(attention_mask) == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(mask, logits, neg_inf)

        attn = tf.nn.softmax(logits, axis=-1)
        attn = self.drop(attn, training=training)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)   # [b,h,s,d_h]
        out = tf.transpose(out, [0, 2, 1, 3])         # [b,s,h,d_h]
        out = tf.reshape(out, [b, s, self.d_model])   # [b,s,d]
        return self.out(out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config

class AdvancedTransformerEncoderBlock(layers.Layer):
    """
    Advanced Transformer encoder block with:
    - Correct key_dim (head_dim instead of d_model)
    - Pre-norm + residual dropout
    - Causal/local masking for time-series
    - SwiGLU MLP for better performance
    - GELU activation
    """
    
    def __init__(
        self,
        d_model=64,
        num_heads=8,
        mlp_ratio=4,
        attn_dropout=0.1,
        dropout=0.1,
        causal=True,          # True for forecasting: forbid peeking at future
        local_window=None,    # e.g., 7 to limit attention to +/- 7 days
        use_rope=False,       # Use RoPE attention instead of standard
        use_swiglu=True,      # Use SwiGLU instead of standard MLP
        **kwargs
    ):
        super(AdvancedTransformerEncoderBlock, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # CORRECT key_dim
        self.causal = causal
        self.local_window = local_window
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.attn_dropout = attn_dropout

        # Pre-norm layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        
        # Attention layer - choose between standard and RoPE
        if use_rope:
            self.attn = AdvancedRoPEMultiHeadSelfAttention(
                d_model=d_model, 
                num_heads=num_heads, 
                dropout=attn_dropout
            )
        else:
            self.attn = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=self.head_dim,          # FIXED: Use head_dim not d_model
                dropout=attn_dropout,
                output_shape=d_model
            )
        
        self.drop1 = layers.Dropout(dropout)

        # Second pre-norm and MLP
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        
        if use_swiglu:
            # SwiGLU MLP - more powerful than standard MLP
            self.gate_proj = layers.Dense(mlp_ratio * d_model)
            self.up_proj = layers.Dense(mlp_ratio * d_model)
            self.down_proj = layers.Dense(d_model)
            self.ffn_dropout = layers.Dropout(dropout)
        else:
            # Standard MLP with GELU
            self.ffn = tf.keras.Sequential([
                layers.Dense(mlp_ratio * d_model, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ])
        
        self.drop2 = layers.Dropout(dropout)

    def _build_mask(self, s):
        """Build causal and/or local attention mask for time-series"""
        # boolean mask [1, s, s] where True means "can attend"
        i = tf.range(s)[:, None]
        j = tf.range(s)[None, :]
        can = tf.ones((s, s), dtype=tf.bool)

        # Apply local window constraint
        if self.local_window is not None:
            can = tf.abs(j - i) <= self.local_window

        # Apply causal constraint (no future peeking)
        if self.causal:
            can = tf.logical_and(can, j <= i)

        return can[None, :, :]  # [1, s, s]

    def _swiglu_mlp(self, x, training=False):
        """SwiGLU MLP: more powerful than standard MLP"""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated = gate * tf.nn.swish(up)  # SwiGLU gating
        gated = self.ffn_dropout(gated, training=training)
        return self.down_proj(gated)

    def call(self, x, training=False, mask=None):
        """Forward pass with pre-norm architecture"""
        # x: [b, s, d]
        b = tf.shape(x)[0]
        s = tf.shape(x)[1]

        # Build time-series mask (causal / local) and combine with padding mask
        attention_mask = self._build_mask(s)  # [1, s, s], broadcastable
        if mask is not None:
            # mask: [b, s] True=keep/False=pad
            pad2d = tf.cast(mask[:, None, :], tf.bool)  # [b,1,s]
            attention_mask = tf.logical_and(attention_mask, pad2d)  # broadcast to [b,s,s]

        # PreNorm -> Attention -> Residual
        y = self.norm1(x)
        if self.use_rope:
            y = self.attn(y, attention_mask=attention_mask, training=training)
        else:
            y = self.attn(y, y, attention_mask=attention_mask, training=training)
        x = x + self.drop1(y, training=training)

        # PreNorm -> MLP -> Residual
        y = self.norm2(x)
        if self.use_swiglu:
            y = self._swiglu_mlp(y, training=training)
        else:
            y = self.ffn(y, training=training)
        x = x + self.drop2(y, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'attn_dropout': self.attn_dropout,
            'causal': self.causal,
            'local_window': self.local_window,
            'use_rope': self.use_rope,
            'use_swiglu': self.use_swiglu
        })
        return config

# Legacy encoder block for backward compatibility
class TransformerEncoderBlock(layers.Layer):
    """
    Legacy Transformer encoder block - kept for backward compatibility
    Use AdvancedTransformerEncoderBlock for better performance
    """
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention - FIXED key_dim
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,  # FIXED: Use head_dim not d_model
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

class RoPETransformerEncoderBlock(layers.Layer):
    """
    Transformer encoder block with RoPE Multi-Head Attention
    """
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(RoPETransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # RoPE Multi-head attention
        self.attention = RoPEMultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads,
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
        """Forward pass through encoder block with RoPE attention"""
        # RoPE Multi-head attention
        attn_output = self.attention(inputs, mask=mask, training=training)
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
    Now supports 3 positional encoding types: learned, rope, legacy
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
                 pos_encoding_type='learned',  # 'learned', 'rope', 'legacy'
                 use_advanced_attention=True,  # Use AdvancedTransformerEncoderBlock
                 causal_masking=True,          # Enable causal masking for forecasting
                 local_window=None,            # Local attention window (e.g., 14 days)
                 use_swiglu=True,             # Use SwiGLU MLP
                 mlp_ratio=4,                 # MLP expansion ratio
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
        self.pos_encoding_type = pos_encoding_type
        self.use_advanced_attention = use_advanced_attention
        self.causal_masking = causal_masking
        self.local_window = local_window
        self.use_swiglu = use_swiglu
        self.mlp_ratio = mlp_ratio
        
        # Input projection layer (project features to d_model dimensions)
        self.input_projection = layers.Dense(d_model, name='input_projection')
        
        # Choose positional encoding type
        if pos_encoding_type == 'learned':
            self.pos_encoding = LearnedPositionalEmbedding(
                max_len=sequence_length * 2,  # Allow some extrapolation
                d_model=d_model,
                dropout=dropout_rate * 0.5
            )
            self.use_rope = False
        elif pos_encoding_type == 'rope':
            self.pos_encoding = None  # RoPE is applied in attention layers
            self.use_rope = True
        else:  # legacy
            self.pos_encoding = PositionalEncoding(sequence_length, d_model)
            self.use_rope = False
        
        # Transformer encoder blocks - choose between advanced and legacy
        if use_advanced_attention:
            self.encoder_blocks = [
                AdvancedTransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=dropout_rate,
                    dropout=dropout_rate,
                    causal=causal_masking,
                    local_window=local_window,
                    use_rope=self.use_rope,
                    use_swiglu=use_swiglu,
                    name=f'advanced_encoder_block_{i}'
                ) for i in range(num_layers)
            ]
        else:
            # Legacy encoder blocks
            if self.use_rope:
                self.encoder_blocks = [
                    RoPETransformerEncoderBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        dropout_rate=dropout_rate,
                        name=f'rope_encoder_block_{i}'
                    ) for i in range(num_layers)
                ]
            else:
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
        
        # Add positional encoding (if not using RoPE)
        if not self.use_rope and self.pos_encoding is not None:
            x = self.pos_encoding(x, training=training)
        
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
            'forecast_horizon': self.forecast_horizon,
            'pos_encoding_type': self.pos_encoding_type,
            'use_advanced_attention': self.use_advanced_attention,
            'causal_masking': self.causal_masking,
            'local_window': self.local_window,
            'use_swiglu': self.use_swiglu,
            'mlp_ratio': self.mlp_ratio
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
        forecast_horizon=config['forecast_horizon'],
        pos_encoding_type=config.get('pos_encoding_type', 'learned'),
        use_advanced_attention=config.get('use_advanced_attention', True),
        causal_masking=config.get('causal_masking', True),
        local_window=config.get('local_window', None),
        use_swiglu=config.get('use_swiglu', True),
        mlp_ratio=config.get('mlp_ratio', 4)
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
    print(f"   - Number of heads: {config['num_heads']} (head_dim: {config['d_model'] // config['num_heads']})")
    print(f"   - Number of layers: {config['num_layers']}")
    print(f"   - Feed-forward dimension: {config['ff_dim']}")
    print(f"   - Dropout rate: {config['dropout_rate']}")
    print(f"   - Forecast horizon: {config['forecast_horizon']}")
    
    # Print advanced features
    if config.get('use_advanced_attention', False):
        print(f"🚀 ADVANCED FEATURES ENABLED:")
        print(f"   - Position encoding: {config.get('pos_encoding_type', 'learned')}")
        print(f"   - Causal masking: {config.get('causal_masking', False)} (prevents future peeking)")
        print(f"   - Local window: {config.get('local_window', 'Global')} days")
        print(f"   - SwiGLU MLP: {config.get('use_swiglu', False)} (vs standard ReLU)")
        print(f"   - MLP ratio: {config.get('mlp_ratio', 4)}x expansion")
        print(f"   - Pre-norm architecture: Yes (better training stability)")
        print(f"   - Correct key_dim: Yes (head_dim={config['d_model'] // config['num_heads']} vs {config['d_model']})")
    
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