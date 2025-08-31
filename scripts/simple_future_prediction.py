#!/usr/bin/env python3
"""
Simple Future Stock Price Prediction
Rebuilds model and makes future predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, DATA_CONFIG, MODEL_CONFIG
from models.transformer_model import build_transformer_model
from scripts.data_collection import calculate_technical_indicators, add_lag_features, add_rolling_features

def load_model_and_scalers():
    """Load model weights and scalers"""
    print("📂 Loading model and scalers...")
    
    # Load metadata
    metadata_path = '/home/abdulhamid/stock-price/data/processed/processed_data_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Rebuild model with same architecture
    model = build_transformer_model(
        sequence_length=metadata['sequence_length'],
        num_features=metadata['num_features'],
        d_model=MODEL_CONFIG['d_model'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_layers=MODEL_CONFIG['num_layers'],
        ff_dim=MODEL_CONFIG['ff_dim'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        forecast_horizon=metadata['forecast_horizon']
    )
    
    # Load weights from the best saved model
    model.load_weights('/home/abdulhamid/stock-price/models/best_transformer.h5')
    print("✅ Model weights loaded successfully")
    
    # Load scalers
    feature_scaler = joblib.load('/home/abdulhamid/stock-price/models/scaler_feature_scaler.pkl')
    target_scaler = joblib.load('/home/abdulhamid/stock-price/models/scaler_target_scaler.pkl')
    
    return model, feature_scaler, target_scaler, metadata

def get_latest_data(symbol='AAPL', days=60):
    """Get the most recent stock data"""
    print(f"📊 Downloading latest {symbol} data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 60)  # Extra buffer for weekends
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                         end=end_date.strftime('%Y-%m-%d'))
    
    if data.empty:
        raise ValueError(f"No recent data for {symbol}")
    
    # Reset index and rename columns
    data.reset_index(inplace=True)
    data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # Calculate same features as training
    data = calculate_technical_indicators(data)
    data = add_lag_features(data)
    data = add_rolling_features(data)
    data = data.dropna()
    
    print(f"✅ Got {len(data)} days of recent data")
    return data

def predict_future_prices(model, recent_data, feature_columns, feature_scaler, target_scaler, 
                         sequence_length=24, future_days=30):
    """Make future price predictions"""
    print(f"🔮 Predicting next {future_days} days...")
    
    # Check if we have enough data
    if len(recent_data) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} days of data, got {len(recent_data)}")
    
    # Get available features from recent data
    available_features = [col for col in feature_columns if col in recent_data.columns]
    
    if len(available_features) < len(feature_columns):
        print(f"⚠️  Using {len(available_features)} of {len(feature_columns)} features")
    
    # Get last sequence for prediction
    last_sequence = recent_data[available_features].tail(sequence_length).values
    
    # Create full feature matrix with proper dimensions
    full_sequence = np.zeros((sequence_length, len(feature_columns)))
    
    # Fill available features
    for i, feature in enumerate(available_features):
        if i < len(feature_columns):
            feature_idx = feature_columns.index(feature) if feature in feature_columns else i
            if feature_idx < len(feature_columns):
                full_sequence[:, feature_idx] = last_sequence[:, i]
    
    # Use the full sequence
    last_sequence = full_sequence
    
    # Scale the sequence
    scaled_sequence = feature_scaler.transform(last_sequence.reshape(-1, len(feature_columns)))
    scaled_sequence = scaled_sequence.reshape(1, sequence_length, len(feature_columns))
    
    future_predictions = []
    current_sequence = scaled_sequence.copy()
    
    for day in range(future_days):
        # Predict next price
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        
        # Convert to actual price
        next_price = target_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]
        future_predictions.append(next_price)
        
        # Update sequence for next prediction (simplified)
        # Roll the sequence and update with prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred_scaled[0, 0]  # Update price feature
        
        if (day + 1) % 10 == 0:
            print(f"   Day {day + 1}: ${next_price:.2f}")
    
    print(f"✅ Generated {len(future_predictions)} future predictions")
    return np.array(future_predictions)

def create_future_prediction_plot(recent_data, future_predictions, symbol='AAPL'):
    """Create a comprehensive future prediction plot"""
    print("📊 Creating future prediction visualization...")
    
    # Prepare data
    recent_prices = recent_data['close'].values
    recent_dates = pd.to_datetime(recent_data['date'].values)
    
    # Create future dates (business days only)
    last_date = recent_dates[-1]
    future_dates = []
    current_date = last_date
    
    for i in range(len(future_predictions)):
        current_date += timedelta(days=1)
        # Skip weekends for stock market
        while current_date.weekday() > 4:  # 5=Saturday, 6=Sunday
            current_date += timedelta(days=1)
        future_dates.append(current_date)
    
    # Create comprehensive plot
    plt.figure(figsize=(20, 12))
    
    # Main plot: Historical + Future
    plt.subplot(2, 1, 1)
    
    # Plot recent historical data
    plt.plot(recent_dates, recent_prices, 'b-', linewidth=2, label='Recent Actual Prices', marker='o', markersize=3)
    
    # Plot future predictions
    all_dates = list(recent_dates[-5:]) + future_dates  # Show connection
    all_prices = list(recent_prices[-5:]) + list(future_predictions)
    plt.plot(all_dates[4:], all_prices[4:], 'r--', linewidth=3, label='AI Future Predictions', marker='s', markersize=5)
    
    # Add vertical line for "today"
    plt.axvline(x=recent_dates[-1], color='orange', linestyle=':', linewidth=3, alpha=0.8, label='Today')
    
    plt.title(f'{symbol} Stock Price: Recent History + AI Future Predictions', fontsize=16, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Zoomed plot: Future only
    plt.subplot(2, 1, 2)
    
    # Show last few actual prices for context
    context_dates = recent_dates[-7:]
    context_prices = recent_prices[-7:]
    plt.plot(context_dates, context_prices, 'g-', linewidth=3, label='Last 7 Days (Actual)', marker='o', markersize=6)
    
    # Future predictions with confidence band
    confidence = np.array(future_predictions) * 0.05  # ±5% confidence
    upper_band = future_predictions + confidence
    lower_band = future_predictions - confidence
    
    plt.fill_between(future_dates, lower_band, upper_band, alpha=0.2, color='red', label='Confidence Band (±5%)')
    plt.plot(future_dates, future_predictions, 'r-', linewidth=3, marker='s', markersize=6, label='AI Future Predictions')
    
    plt.title(f'AI Predictions: Next {len(future_predictions)} Days', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Price ($)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    save_path = 'results/future_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Future prediction plot saved to {save_path}")
    plt.show()

def print_future_forecast(future_predictions, symbol='AAPL'):
    """Print detailed future forecast"""
    print(f"\n🔮 AI FUTURE FORECAST FOR {symbol}")
    print("=" * 60)
    
    today = datetime.now()
    business_day_count = 0
    
    print("📅 NEXT 2 WEEKS:")
    for i in range(min(10, len(future_predictions))):
        # Calculate next business day
        pred_date = today + timedelta(days=i+1)
        while pred_date.weekday() > 4:  # Skip weekends
            pred_date += timedelta(days=1)
        
        day_name = pred_date.strftime("%A")
        date_str = pred_date.strftime("%Y-%m-%d")
        price = future_predictions[i]
        
        print(f"   {date_str} ({day_name[:3]}): ${price:.2f}")
        business_day_count += 1
    
    print(f"\n📊 FORECAST SUMMARY:")
    print(f"   Tomorrow's prediction: ${future_predictions[0]:.2f}")
    print(f"   1-week forecast: ${future_predictions[4]:.2f}")
    print(f"   2-week forecast: ${future_predictions[9]:.2f}")
    print(f"   1-month forecast: ${future_predictions[-1]:.2f}")
    
    # Calculate trends
    short_term_change = ((future_predictions[4] - future_predictions[0]) / future_predictions[0]) * 100
    long_term_change = ((future_predictions[-1] - future_predictions[0]) / future_predictions[0]) * 100
    
    print(f"\n📈 PREDICTED TRENDS:")
    print(f"   Week 1 trend: {short_term_change:+.2f}%")
    print(f"   Month trend: {long_term_change:+.2f}%")
    
    if long_term_change > 2:
        print(f"   🚀 AI predicts {symbol} will RISE")
    elif long_term_change < -2:
        print(f"   📉 AI predicts {symbol} will FALL") 
    else:
        print(f"   ➡️  AI predicts {symbol} will stay STABLE")

def main():
    """Main future prediction function"""
    print("🔮 Starting Future Stock Price Prediction")
    print("=" * 60)
    
    try:
        # Load model and scalers
        model, feature_scaler, target_scaler, metadata = load_model_and_scalers()
        
        # Get recent data (ensure we have enough for sequence_length)
        required_days = max(60, metadata['sequence_length'] + 30)  # Ensure enough data
        recent_data = get_latest_data(DATA_CONFIG['stock_symbol'], days=required_days)
        
        # Make future predictions
        future_predictions = predict_future_prices(
            model=model,
            recent_data=recent_data,
            feature_columns=metadata['feature_columns'],
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            sequence_length=metadata['sequence_length'],
            future_days=30
        )
        
        # Create visualization
        create_future_prediction_plot(recent_data, future_predictions, DATA_CONFIG['stock_symbol'])
        
        # Print forecast
        print_future_forecast(future_predictions, DATA_CONFIG['stock_symbol'])
        
        print(f"\n🎉 FUTURE PREDICTION COMPLETE!")
        print("🔮 Your AI has predicted the next 30 days of Apple stock prices!")
        print("📊 Check 'results/future_predictions.png' for the visualization")
        
        return future_predictions
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise e

if __name__ == "__main__":
    predictions = main()