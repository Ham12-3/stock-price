#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Stock Price Forecasting
Handles cleaning, feature engineering, and sequence creation for model training
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, PREPROCESSING_CONFIG, MODEL_CONFIG, FILE_PATHS
from utils.utils import create_sequences, split_data, scale_data, scale_targets, check_data_quality, get_feature_columns

def load_raw_data(symbol=None):
    """
    Load raw data collected from data_collection.py
    
    Args:
        symbol (str): Stock symbol (if None, uses default from config)
        
    Returns:
        pd.DataFrame: Raw stock data
    """
    if symbol is None:
        from config import DATA_CONFIG
        symbol = DATA_CONFIG['stock_symbol']
    
    raw_file = FILE_PATHS['raw_data'].replace('stock_data.csv', f'{symbol}_raw_data.csv')
    
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")
    
    print(f"📂 Loading raw data from {raw_file}...")
    data = pd.read_csv(raw_file)
    
    # Convert date column to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    
    print(f"✅ Loaded {len(data)} rows of raw data")
    return data

def clean_data(data):
    """
    Clean and prepare data for modeling
    
    Args:
        data (pd.DataFrame): Raw stock data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    print("🧹 Cleaning data...")
    df = data.copy()
    
    # Check data quality before cleaning
    issues = check_data_quality(df)
    
    # Handle missing values
    missing_method = PREPROCESSING_CONFIG['missing_value_method']
    if missing_method == 'ffill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif missing_method == 'bfill':
        df = df.fillna(method='bfill').fillna(method='ffill')
    elif missing_method == 'interpolate':
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate()
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Handle outliers
    outlier_method = PREPROCESSING_CONFIG['outlier_method']
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, PREPROCESSING_CONFIG['iqr_multiplier'])
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df, PREPROCESSING_CONFIG['zscore_threshold'])
    
    # Remove infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Ensure we have enough data
    min_required = MODEL_CONFIG['sequence_length'] + 100  # Buffer for train/val/test
    if len(df) < min_required:
        raise ValueError(f"Not enough data after cleaning. Need at least {min_required}, got {len(df)}")
    
    print(f"✅ Data cleaned. {len(df)} rows remaining")
    return df

def remove_outliers_iqr(data, multiplier=1.5):
    """Remove outliers using IQR method"""
    df = data.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    initial_count = len(df)
    
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Keep rows where all values are within bounds
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    removed_count = initial_count - len(df)
    print(f"   Removed {removed_count} outliers using IQR method")
    
    return df

def remove_outliers_zscore(data, threshold=3):
    """Remove outliers using Z-score method"""
    df = data.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    initial_count = len(df)
    
    for column in numeric_columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]
    
    removed_count = initial_count - len(df)
    print(f"   Removed {removed_count} outliers using Z-score method")
    
    return df

def feature_selection(data, target_col='close'):
    """
    Select relevant features and remove highly correlated ones
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Data with selected features
    """
    print("🎯 Performing feature selection...")
    df = data.copy()
    
    # Get numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from features for correlation analysis
    feature_columns = [col for col in numeric_columns if col != target_col]
    
    if len(feature_columns) == 0:
        print("⚠️  No numeric features found for selection")
        return df
    
    # Remove highly correlated features
    correlation_threshold = PREPROCESSING_CONFIG['correlation_threshold']
    correlation_matrix = df[feature_columns].corr().abs()
    
    # Find pairs of highly correlated features
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > correlation_threshold:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
    
    # Remove one feature from each highly correlated pair
    features_to_remove = set()
    for feat1, feat2 in high_corr_pairs:
        # Remove the feature with lower correlation to target
        if target_col in df.columns:
            corr1 = abs(df[target_col].corr(df[feat1]))
            corr2 = abs(df[target_col].corr(df[feat2]))
            if corr1 < corr2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat2)  # Default: remove second feature
    
    if features_to_remove:
        print(f"   Removing {len(features_to_remove)} highly correlated features")
        df = df.drop(columns=list(features_to_remove))
    
    # Remove low variance features
    variance_threshold = PREPROCESSING_CONFIG['variance_threshold']
    remaining_features = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    
    low_variance_features = []
    for feature in remaining_features:
        if df[feature].var() < variance_threshold:
            low_variance_features.append(feature)
    
    if low_variance_features:
        print(f"   Removing {len(low_variance_features)} low variance features")
        df = df.drop(columns=low_variance_features)
    
    final_features = len([col for col in df.columns if col != target_col])
    print(f"✅ Feature selection completed. {final_features} features remaining")
    
    return df

def prepare_model_data(data, target_col='close', test_mode=False):
    """
    Prepare data for model training (create sequences, split, scale)
    
    Args:
        data (pd.DataFrame): Preprocessed data
        target_col (str): Target column name
        test_mode (bool): If True, use smaller parameters for testing
        
    Returns:
        dict: Dictionary containing prepared data and scalers
    """
    print("🔧 Preparing data for model training...")
    
    # Get configuration
    if test_mode:
        sequence_length = 12  # Smaller for testing
        forecast_horizon = 1
    else:
        sequence_length = MODEL_CONFIG['sequence_length']
        forecast_horizon = MODEL_CONFIG['forecast_horizon']
    
    # Get feature columns (exclude target and non-numeric columns)
    feature_columns = get_feature_columns(data, exclude_cols=[target_col])
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    print(f"   Using {len(feature_columns)} features for model input")
    print(f"   Target column: {target_col}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Forecast horizon: {forecast_horizon}")
    
    # Create sequences
    X, y = create_sequences(
        data=data,
        target_col=target_col,
        feature_cols=feature_columns,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )
    
    # Split data into train/val/test
    train_data, val_data, test_data = split_data(
        X, y,
        train_ratio=MODEL_CONFIG['validation_split'] if 'validation_split' in MODEL_CONFIG else 0.7,
        val_ratio=MODEL_CONFIG['test_split'] if 'test_split' in MODEL_CONFIG else 0.15
    )
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Scale features
    scaler_type = PREPROCESSING_CONFIG['scaler_type']
    X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler = scale_data(
        X_train, X_val, X_test, scaler_type=scaler_type
    )
    
    # Scale targets
    y_train_scaled, y_val_scaled, y_test_scaled, target_scaler = scale_targets(
        y_train, y_val, y_test, scaler_type=scaler_type
    )
    
    # Prepare return dictionary
    prepared_data = {
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'X_val': X_val_scaled,
        'y_val': y_val_scaled,
        'X_test': X_test_scaled,
        'y_test': y_test_scaled,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_columns': feature_columns,
        'target_column': target_col,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'num_features': len(feature_columns)
    }
    
    print("✅ Data preparation completed successfully")
    return prepared_data

def save_processed_data(prepared_data, save_path=None):
    """Save processed data and scalers"""
    if save_path is None:
        save_path = FILE_PATHS['processed_data']
    
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save sequences
    sequences_path = save_path.replace('.csv', '_sequences.npz')
    np.savez(
        sequences_path,
        X_train=prepared_data['X_train'],
        y_train=prepared_data['y_train'],
        X_val=prepared_data['X_val'],
        y_val=prepared_data['y_val'],
        X_test=prepared_data['X_test'],
        y_test=prepared_data['y_test']
    )
    print(f"💾 Sequences saved to {sequences_path}")
    
    # Save scalers
    scaler_dir = os.path.dirname(FILE_PATHS['scaler'])
    os.makedirs(scaler_dir, exist_ok=True)
    
    feature_scaler_path = FILE_PATHS['scaler'].replace('.pkl', '_feature_scaler.pkl')
    target_scaler_path = FILE_PATHS['scaler'].replace('.pkl', '_target_scaler.pkl')
    
    joblib.dump(prepared_data['feature_scaler'], feature_scaler_path)
    joblib.dump(prepared_data['target_scaler'], target_scaler_path)
    
    print(f"💾 Feature scaler saved to {feature_scaler_path}")
    print(f"💾 Target scaler saved to {target_scaler_path}")
    
    # Save metadata
    metadata = {
        'feature_columns': prepared_data['feature_columns'],
        'target_column': prepared_data['target_column'],
        'sequence_length': prepared_data['sequence_length'],
        'forecast_horizon': prepared_data['forecast_horizon'],
        'num_features': prepared_data['num_features'],
        'data_shapes': {
            'X_train': prepared_data['X_train'].shape,
            'y_train': prepared_data['y_train'].shape,
            'X_val': prepared_data['X_val'].shape,
            'y_val': prepared_data['y_val'].shape,
            'X_test': prepared_data['X_test'].shape,
            'y_test': prepared_data['y_test'].shape
        }
    }
    
    metadata_path = save_path.replace('.csv', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"💾 Metadata saved to {metadata_path}")

def main():
    """Main preprocessing function"""
    print("🚀 Starting data preprocessing pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Load raw data
        raw_data = load_raw_data()
        
        # Step 2: Clean data
        clean_data_df = clean_data(raw_data)
        
        # Step 3: Feature selection
        selected_data = feature_selection(clean_data_df)
        
        # Step 4: Prepare model data
        prepared_data = prepare_model_data(selected_data, target_col='close')
        
        # Step 5: Save processed data
        save_processed_data(prepared_data)
        
        # Summary
        print("\n🏆 Data preprocessing completed successfully!")
        print("=" * 60)
        print("PREPROCESSING SUMMARY:")
        print(f"   Original data shape: {raw_data.shape}")
        print(f"   After cleaning: {clean_data_df.shape}")
        print(f"   After feature selection: {selected_data.shape}")
        print(f"   Training sequences: {prepared_data['X_train'].shape}")
        print(f"   Validation sequences: {prepared_data['X_val'].shape}")
        print(f"   Test sequences: {prepared_data['X_test'].shape}")
        print(f"   Number of features: {prepared_data['num_features']}")
        print(f"   Sequence length: {prepared_data['sequence_length']}")
        print(f"   Forecast horizon: {prepared_data['forecast_horizon']}")
        
        return prepared_data
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    processed_data = main()