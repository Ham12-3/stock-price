#!/usr/bin/env python3
"""
Stock Data Collection Script
Collects stock data from Yahoo Finance and calculates technical indicators
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
from config import get_config, DATA_CONFIG, FILE_PATHS

def download_stock_data(symbol, start_date, end_date, interval='1d'):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d', '1h', etc.)
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        print(f"📊 Downloading {symbol} data from {start_date} to {end_date}...")
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Download historical data
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Rename columns for consistency
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        data.rename(columns=column_mapping, inplace=True)
        
        # Remove dividends and stock splits columns if present
        columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
        data = data[columns_to_keep]
        
        print(f"✅ Downloaded {len(data)} rows of data for {symbol}")
        return data
        
    except Exception as e:
        print(f"❌ Error downloading data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(data):
    """
    Calculate technical indicators from OHLCV data
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        
    Returns:
        pd.DataFrame: Data with additional technical indicator columns
    """
    df = data.copy()
    
    try:
        print("🔧 Calculating technical indicators...")
        
        # Simple Moving Averages
        for window in DATA_CONFIG['sma_windows']:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # RSI (Relative Strength Index)
        df['rsi'] = calculate_rsi(df['close'], DATA_CONFIG['rsi_period'])
        
        # MACD
        macd_line, macd_signal, macd_histogram = calculate_macd(
            df['close'], 
            DATA_CONFIG['macd_fast'], 
            DATA_CONFIG['macd_slow'], 
            DATA_CONFIG['macd_signal']
        )
        df['macd_line'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(
            df['close'],
            DATA_CONFIG['bollinger_window'],
            DATA_CONFIG['bollinger_std']
        )
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower  
        df['bb_middle'] = bb_middle
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Volume-based features
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(window=DATA_CONFIG['volatility_window']).std()
        
        print("✅ Technical indicators calculated successfully")
        return df
        
    except Exception as e:
        print(f"❌ Error calculating technical indicators: {str(e)}")
        return data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    bb_upper = sma + (std * std_dev)
    bb_lower = sma - (std * std_dev)
    bb_middle = sma
    
    return bb_upper, bb_lower, bb_middle

def add_lag_features(data, target_col='close'):
    """Add lag features for the target variable"""
    df = data.copy()
    
    print("🔄 Adding lag features...")
    
    for lag in DATA_CONFIG['lag_features']:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df

def add_rolling_features(data, target_col='close'):
    """Add rolling statistical features"""
    df = data.copy()
    
    print("📈 Adding rolling features...")
    
    for window in DATA_CONFIG['rolling_windows']:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    
    return df

def collect_market_indices():
    """Collect market indices for external context"""
    indices = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ', 
        '^VIX': 'Volatility Index'  # Fixed VIX symbol
    }
    
    market_data = {}
    
    for symbol, name in indices.items():
        try:
            print(f"📊 Downloading {name} ({symbol}) data...")
            data = download_stock_data(
                symbol, 
                DATA_CONFIG['start_date'], 
                DATA_CONFIG['end_date']
            )
            if not data.empty:
                market_data[symbol] = data[['date', 'close']].rename(
                    columns={'close': f'{symbol.lower()}_close'}
                )
        except Exception as e:
            print(f"⚠️  Warning: Could not download {symbol}: {str(e)}")
    
    return market_data

def merge_market_data(stock_data, market_data):
    """Merge stock data with market indices"""
    df = stock_data.copy()
    
    print("🔗 Merging with market data...")
    
    for symbol, data in market_data.items():
        df = df.merge(data, on='date', how='left')
    
    return df

def save_raw_data(data, symbol):
    """Save raw collected data"""
    filename = FILE_PATHS['raw_data'].replace('stock_data.csv', f'{symbol}_raw_data.csv')
    data.to_csv(filename, index=False)
    print(f"💾 Raw data saved to {filename}")

def main():
    """Main data collection function"""
    print("🚀 Starting stock data collection...")
    print("=" * 50)
    
    # Get configuration
    config = get_config()
    symbol = DATA_CONFIG['stock_symbol']
    
    # Step 1: Download stock data
    stock_data = download_stock_data(
        symbol,
        DATA_CONFIG['start_date'],
        DATA_CONFIG['end_date'],
        DATA_CONFIG['data_frequency']
    )
    
    if stock_data.empty:
        print("❌ Failed to download stock data. Exiting.")
        return None
    
    # Step 2: Calculate technical indicators
    stock_data = calculate_technical_indicators(stock_data)
    
    # Step 3: Add lag features
    stock_data = add_lag_features(stock_data)
    
    # Step 4: Add rolling features
    stock_data = add_rolling_features(stock_data)
    
    # Step 5: Collect market indices
    market_data = collect_market_indices()
    
    # Step 6: Merge with market data
    if market_data:
        stock_data = merge_market_data(stock_data, market_data)
    
    # Step 7: Save raw data
    save_raw_data(stock_data, symbol)
    
    # Display summary
    print("\n📊 Data Collection Summary:")
    print(f"Symbol: {symbol}")
    print(f"Date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
    print(f"Total rows: {len(stock_data)}")
    print(f"Total features: {len(stock_data.columns)}")
    print(f"Missing values: {stock_data.isnull().sum().sum()}")
    
    print("\n🏆 Data collection completed successfully!")
    return stock_data

if __name__ == "__main__":
    collected_data = main()