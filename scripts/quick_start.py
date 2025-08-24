#!/usr/bin/env python3
"""
Quick Start Script for Time Series Forecasting with Transformers
This script demonstrates the basic workflow for the project.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main function to demonstrate the workflow"""
    print("🚀 Starting Time Series Forecasting with Transformers Project")
    print("=" * 60)
    
    # Step 1: Check environment
    print("\n📋 Step 1: Environment Check")
    check_environment()
    
    # Step 2: Create sample data
    print("\n📊 Step 2: Create Sample Data")
    sample_data = create_sample_data()
    
    # Step 3: Basic preprocessing
    print("\n🔧 Step 3: Basic Preprocessing")
    processed_data = basic_preprocessing(sample_data)
    
    # Step 4: Show data structure
    print("\n📈 Step 4: Data Structure Overview")
    show_data_overview(processed_data)
    
    # Step 5: Next steps
    print("\n🎯 Step 5: Next Steps")
    show_next_steps()
    
    print("\n✅ Quick start completed! Check the docs/IMPLEMENTATION_GUIDE.md for detailed instructions.")

def check_environment():
    """Check if required packages are available"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} is available")
    except ImportError:
        print("❌ TensorFlow not found. Install with: pip install tensorflow")
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} is available")
    except ImportError:
        print("❌ Pandas not found. Install with: pip install pandas")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} is available")
    except ImportError:
        print("❌ NumPy not found. Install with: pip install numpy")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} is available")
    except ImportError:
        print("❌ Scikit-learn not found. Install with: pip install scikit-learn")

def create_sample_data():
    """Create sample stock price data for demonstration"""
    print("Creating sample stock price data...")
    
    # Generate sample dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample stock prices (random walk)
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [100]  # Starting price
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure price doesn't go below 1
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    })
    
    data.set_index('Date', inplace=True)
    
    # Add some technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_simple_rsi(data['Close'])
    
    print(f"✅ Created sample data with {len(data)} rows")
    return data

def calculate_simple_rsi(prices, period=14):
    """Calculate a simple RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def basic_preprocessing(data):
    """Apply basic preprocessing steps"""
    print("Applying basic preprocessing...")
    
    # Handle missing values
    data_clean = data.fillna(method='ffill').fillna(method='bfill')
    
    # Create additional features
    data_clean['price_change'] = data_clean['Close'].pct_change()
    data_clean['volatility'] = data_clean['price_change'].rolling(window=20).std()
    
    # Remove rows with NaN values
    data_clean = data_clean.dropna()
    
    print(f"✅ Preprocessing completed. Clean data has {len(data_clean)} rows")
    return data_clean

def show_data_overview(data):
    """Show overview of the processed data"""
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Features: {list(data.columns)}")
    
    print("\nSample data (first 5 rows):")
    print(data.head())
    
    print("\nData statistics:")
    print(data.describe())
    
    # Create a simple plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['Close'])
    plt.title('Stock Price Over Time')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.plot(data.index, data['SMA_20'])
    plt.title('20-Day Moving Average')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.plot(data.index, data['RSI'])
    plt.title('RSI Indicator')
    plt.ylabel('RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    plt.plot(data.index, data['Volume'])
    plt.title('Trading Volume')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/sample_data_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Saved sample data overview plot to results/sample_data_overview.png")
    
    plt.show()

def show_next_steps():
    """Show the next steps for the project"""
    print("Here's what you should do next:")
    print("\n1. 📚 Read the detailed guide: docs/IMPLEMENTATION_GUIDE.md")
    print("2. 🐍 Install required packages: pip install -r requirements.txt")
    print("3. 📊 Collect real data using scripts/data_collection.py")
    print("4. 🔧 Preprocess data using scripts/data_preprocessing.py")
    print("5. 🏗️  Build Transformer model using models/transformer_model.py")
    print("6. 🚀 Train model using scripts/training.py")
    print("7. 📊 Evaluate results using scripts/evaluation.py")
    
    print("\n💡 Tips:")
    print("- Start with a small dataset to test the pipeline")
    print("- Experiment with different hyperparameters")
    print("- Use the notebooks/ folder for exploration")
    print("- Check the results/ folder for outputs")

if __name__ == "__main__":
    main()
