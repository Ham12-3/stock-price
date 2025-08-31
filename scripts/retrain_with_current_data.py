#!/usr/bin/env python3
"""
Retrain model with current data to fix scaling issues
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_collection import download_stock_data, calculate_technical_indicators, add_lag_features, add_rolling_features
from scripts.data_preprocessing import clean_data, feature_selection, prepare_model_data, save_processed_data
from scripts.training import train_model
from config import DATA_CONFIG

def get_current_comprehensive_data():
    """Get comprehensive Apple data including recent prices"""
    print("📊 Downloading comprehensive Apple stock data...")
    
    # Get data from 2020 to today (including current high prices)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2020-01-01"
    
    print(f"   Period: {start_date} to {end_date}")
    
    # Download stock data
    stock_data = download_stock_data('AAPL', start_date, end_date)
    
    if stock_data.empty:
        raise ValueError("Could not download Apple stock data")
    
    print(f"✅ Downloaded {len(stock_data)} days of Apple data")
    print(f"   Price range: ${stock_data['close'].min():.2f} - ${stock_data['close'].max():.2f}")
    
    # Set date as index to avoid it being included as a feature
    if 'date' in stock_data.columns:
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
    
    # Calculate technical indicators
    stock_data = calculate_technical_indicators(stock_data)
    stock_data = add_lag_features(stock_data)
    stock_data = add_rolling_features(stock_data)
    
    # Save raw data  
    raw_file = '/home/abdulhamid/stock-price/data/raw/AAPL_current_data.csv'
    stock_data.to_csv(raw_file)
    print(f"💾 Saved comprehensive data to {raw_file}")
    
    return stock_data

def retrain_model_with_current_data():
    """Retrain model with data that includes current price levels"""
    print("🔄 Retraining model with current data...")
    
    try:
        # Step 1: Get current comprehensive data
        current_data = get_current_comprehensive_data()
        
        # Step 2: Clean and prepare data
        print("🧹 Cleaning data...")
        clean_data_df = clean_data(current_data)
        
        print("🎯 Feature selection...")
        selected_data = feature_selection(clean_data_df)
        
        print(f"   Data range after cleaning: ${selected_data['close'].min():.2f} - ${selected_data['close'].max():.2f}")
        
        # Step 3: Prepare for training (this will create new scalers with current range)
        print("🔧 Preparing data for training...")
        prepared_data = prepare_model_data(selected_data, target_col='close')
        
        print(f"   Training sequences: {prepared_data['X_train'].shape}")
        print(f"   Validation sequences: {prepared_data['X_val'].shape}")
        print(f"   Test sequences: {prepared_data['X_test'].shape}")
        
        # Step 4: Save processed data
        save_processed_data(prepared_data)
        
        # Step 5: Train model with new data
        print("🏋️ Training model with current data...")
        model, history = train_model(
            prepared_data['X_train'], 
            prepared_data['y_train'],
            prepared_data['X_val'], 
            prepared_data['y_val']
        )
        
        print("✅ Model retrained successfully!")
        print("   New scalers fit to current Apple price range")
        print("   Model now understands $200+ Apple prices")
        
        return model, history, prepared_data
        
    except Exception as e:
        print(f"❌ Error retraining model: {str(e)}")
        raise e

def main():
    """Main retraining function"""
    print("🔄 RETRAINING MODEL WITH CURRENT APPLE DATA")
    print("=" * 60)
    print("🎯 Goal: Fix scaling issues and handle current $230+ prices")
    print()
    
    model, history, data = retrain_model_with_current_data()
    
    print("\n🎉 RETRAINING COMPLETED!")
    print("=" * 60)
    print("✅ Model now trained on data including current Apple prices")
    print("✅ New scalers fit to proper price range ($70-$230+)")
    print("✅ Should give much more realistic future predictions")
    print()
    print("🔮 Now run the future prediction script:")
    print("   python scripts/simple_future_prediction.py")
    
    return model, history, data

if __name__ == "__main__":
    retrained_model, training_history, training_data = main()