#!/usr/bin/env python3
"""
Future Stock Price Prediction Script
Makes actual future predictions beyond the training data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, DATA_CONFIG
from utils.utils import load_model_artifacts, calculate_metrics
from scripts.data_collection import calculate_technical_indicators, add_lag_features, add_rolling_features

def load_recent_data(symbol='AAPL', days_back=50):
    """
    Load recent data including today for future prediction
    
    Args:
        symbol (str): Stock symbol
        days_back (int): How many days back to get for context
        
    Returns:
        pd.DataFrame: Recent stock data with features
    """
    print(f"📊 Loading recent {symbol} data for future prediction...")
    
    # Get recent data including today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 30)  # Extra buffer
    
    try:
        # Download recent data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                            end=end_date.strftime('%Y-%m-%d'), 
                            interval='1d')
        
        if data.empty:
            raise ValueError(f"No recent data found for {symbol}")
        
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
        
        # Calculate features (same as training)
        data = calculate_technical_indicators(data)
        data = add_lag_features(data)
        data = add_rolling_features(data)
        
        # Remove rows with NaN
        data = data.dropna()
        
        print(f"✅ Loaded {len(data)} days of recent data")
        return data
        
    except Exception as e:
        print(f"❌ Error loading recent data: {str(e)}")
        return pd.DataFrame()

def prepare_future_prediction_data(recent_data, feature_columns, feature_scaler, sequence_length=24):
    """
    Prepare recent data for future prediction
    
    Args:
        recent_data (pd.DataFrame): Recent stock data
        feature_columns (list): Feature columns to use
        feature_scaler: Fitted scaler from training
        sequence_length (int): Length of input sequences
        
    Returns:
        tuple: (scaled_sequence, last_actual_prices, recent_dates)
    """
    print("🔧 Preparing data for future prediction...")
    
    # Get the most recent sequence_length days
    if len(recent_data) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} days of data, got {len(recent_data)}")
    
    # Select features (same order as training)
    available_features = [col for col in feature_columns if col in recent_data.columns]
    if len(available_features) != len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"⚠️  Missing features: {missing}")
    
    # Get last sequence_length days
    recent_features = recent_data[available_features].tail(sequence_length).values
    
    # Scale the features
    scaled_sequence = feature_scaler.transform(recent_features.reshape(-1, len(available_features)))
    scaled_sequence = scaled_sequence.reshape(1, sequence_length, len(available_features))
    
    # Get actual recent prices and dates for plotting
    last_actual_prices = recent_data['close'].tail(sequence_length).values
    recent_dates = recent_data['date'].tail(sequence_length).values
    
    print(f"✅ Prepared sequence shape: {scaled_sequence.shape}")
    
    return scaled_sequence, last_actual_prices, recent_dates, available_features

def make_future_predictions(model, scaled_sequence, target_scaler, num_future_days=30):
    """
    Make predictions for future days
    
    Args:
        model: Trained model
        scaled_sequence: Recent data sequence
        target_scaler: Target scaler from training
        num_future_days (int): Number of future days to predict
        
    Returns:
        np.array: Future price predictions
    """
    print(f"🔮 Making predictions for next {num_future_days} days...")
    
    future_predictions = []
    current_sequence = scaled_sequence.copy()
    
    for day in range(num_future_days):
        # Predict next day
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        
        # Convert back to actual price
        next_pred_price = target_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]
        future_predictions.append(next_pred_price)
        
        # Update sequence for next prediction (simplified approach)
        # In reality, you'd need to update all features, but for demo we'll approximate
        new_feature_values = current_sequence[0, -1, :].copy()
        new_feature_values[0] = next_pred_scaled[0, 0]  # Update close price feature
        
        # Shift sequence and add new values
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = new_feature_values
        
        if (day + 1) % 10 == 0:
            print(f"   Predicted day {day + 1}: ${next_pred_price:.2f}")
    
    print(f"✅ Generated {len(future_predictions)} future predictions")
    return np.array(future_predictions)

def create_comprehensive_plot(historical_data, recent_data, recent_dates, future_predictions, 
                            symbol='AAPL', save_path='results/future_predictions.png'):
    """
    Create comprehensive plot showing historical, recent, and future predictions
    """
    print("📊 Creating comprehensive prediction plot...")
    
    # Create future dates
    last_date = pd.to_datetime(recent_dates[-1])
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_predictions))]
    
    # Create the plot
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Complete timeline
    plt.subplot(3, 1, 1)
    
    # Historical training data (first part)
    if historical_data is not None and len(historical_data) > 0:
        hist_dates = pd.to_datetime(historical_data['date'])
        plt.plot(hist_dates, historical_data['close'], 'b-', label='Historical Training Data', alpha=0.7, linewidth=1)
    
    # Recent actual data
    plt.plot(pd.to_datetime(recent_dates), recent_data, 'g-', label='Recent Actual Prices', linewidth=2)
    
    # Future predictions
    all_future_dates = [pd.to_datetime(recent_dates[-1])] + future_dates
    all_future_prices = [recent_data[-1]] + list(future_predictions)
    plt.plot(all_future_dates, all_future_prices, 'r--', label='Future Predictions', linewidth=3, marker='o', markersize=4)
    
    plt.title(f'{symbol} Stock Price: Historical, Recent & Future Predictions', fontsize=16, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 2: Recent + Future focus
    plt.subplot(3, 1, 2)
    
    # Recent actual
    plt.plot(pd.to_datetime(recent_dates[-15:]), recent_data[-15:], 'g-', label='Recent Actual', linewidth=3, marker='o')
    
    # Future predictions  
    plt.plot(all_future_dates, all_future_prices, 'r--', label='AI Future Predictions', linewidth=3, marker='s', markersize=6)
    
    # Add vertical line to separate actual from predictions
    plt.axvline(x=pd.to_datetime(recent_dates[-1]), color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Today')
    
    plt.title('Recent Prices vs Future AI Predictions (Zoomed)', fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 3: Future predictions only with confidence bands
    plt.subplot(3, 1, 3)
    
    # Calculate simple confidence bands (±10% of prediction)
    confidence_band = np.array(future_predictions) * 0.1
    upper_band = future_predictions + confidence_band
    lower_band = future_predictions - confidence_band
    
    plt.fill_between(future_dates, lower_band, upper_band, alpha=0.2, color='red', label='Confidence Band (±10%)')
    plt.plot(future_dates, future_predictions, 'r-', linewidth=3, marker='o', markersize=6, label='AI Predictions')
    
    plt.title('Future Price Predictions with Confidence Band', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive plot saved to {save_path}")
    plt.show()

def print_future_predictions(future_predictions, symbol='AAPL'):
    """Print future predictions in a nice format"""
    print(f"\n🔮 AI FUTURE PREDICTIONS FOR {symbol}")
    print("=" * 50)
    
    today = datetime.now()
    
    for i, pred in enumerate(future_predictions[:14]):  # Show first 2 weeks
        future_date = today + timedelta(days=i+1)
        day_name = future_date.strftime("%A")
        date_str = future_date.strftime("%Y-%m-%d")
        print(f"   {date_str} ({day_name}): ${pred:.2f}")
    
    if len(future_predictions) > 14:
        print(f"   ... and {len(future_predictions)-14} more days")
    
    print(f"\n📈 PREDICTION SUMMARY:")
    print(f"   Current Price: ${future_predictions[0]:.2f} (tomorrow)")
    print(f"   7-day prediction: ${future_predictions[6]:.2f}")
    print(f"   14-day prediction: ${future_predictions[13]:.2f}")
    print(f"   30-day prediction: ${future_predictions[-1]:.2f}")
    
    # Calculate trends
    weekly_change = ((future_predictions[6] - future_predictions[0]) / future_predictions[0]) * 100
    monthly_change = ((future_predictions[-1] - future_predictions[0]) / future_predictions[0]) * 100
    
    print(f"\n📊 PREDICTED TRENDS:")
    print(f"   7-day change: {weekly_change:+.2f}%")
    print(f"   30-day change: {monthly_change:+.2f}%")

def main():
    """Main future prediction function"""
    print("🔮 Starting Future Stock Price Prediction")
    print("=" * 60)
    
    try:
        # Step 1: Load trained model and artifacts
        print("📂 Loading trained model...")
        model, feature_scaler, target_scaler, config = load_model_artifacts('models/saved_model')
        
        feature_columns = config['metadata']['feature_columns']
        sequence_length = config['metadata']['sequence_length']
        symbol = DATA_CONFIG['stock_symbol']
        
        print(f"✅ Model loaded - Features: {len(feature_columns)}, Sequence: {sequence_length}")
        
        # Step 2: Load recent data
        recent_data = load_recent_data(symbol, days_back=60)
        if recent_data.empty:
            raise ValueError("Could not load recent data")
        
        # Step 3: Prepare data for prediction
        scaled_sequence, last_actual_prices, recent_dates, available_features = prepare_future_prediction_data(
            recent_data, feature_columns, feature_scaler, sequence_length
        )
        
        # Step 4: Make future predictions
        future_predictions = make_future_predictions(
            model, scaled_sequence, target_scaler, num_future_days=30
        )
        
        # Step 5: Load historical training data for complete plot
        try:
            historical_data = pd.read_csv('/home/abdulhamid/stock-price/data/raw/AAPL_raw_data.csv')
        except:
            historical_data = None
        
        # Step 6: Create comprehensive visualization
        create_comprehensive_plot(
            historical_data, last_actual_prices, recent_dates, future_predictions, symbol
        )
        
        # Step 7: Print predictions
        print_future_predictions(future_predictions, symbol)
        
        print(f"\n🎉 FUTURE PREDICTION COMPLETED!")
        print("=" * 60)
        print("📊 Check 'results/future_predictions.png' for the complete visualization")
        print("🔮 Your AI has predicted the next 30 days of stock prices!")
        
        return future_predictions
        
    except Exception as e:
        print(f"❌ Error in future prediction: {str(e)}")
        raise e

if __name__ == "__main__":
    future_preds = main()