#!/usr/bin/env python3
"""
Cleanup Redundant Files
Removes all the overlapping/redundant model files and scripts
"""

import os
import shutil

def cleanup_redundant_files():
    """Remove redundant and overlapping files"""
    
    print("🧹 Cleaning up redundant files...")
    
    # Files to remove (keeping only the essentials)
    files_to_remove = [
        # Redundant model files (keep only basic transformer_model.py)
        'models/advanced_transformer.py',
        
        # Redundant training scripts (replaced by unified system)
        'scripts/advanced_training.py',
        'scripts/clean_training.py', 
        'scripts/retrain_with_current_data.py',
        'scripts/quick_start.py',
        
        # Redundant prediction scripts (replaced by unified system)
        'scripts/simple_future_prediction.py',
        'scripts/future_prediction.py',
        
        # Duplicate preprocessing (data_collection.py handles this)
        'scripts/data_preprocessing.py',
        
        # Old result files that might be confusing
        'results/future_predictions.png',
        'models/scaler_feature_scaler.pkl',
        'models/scaler_target_scaler.pkl',
        
        # Extra directories/files
        'run_fresh.sh',
        'test_advanced_system.py',
        'test_everything.py',
    ]
    
    # Directories to clean
    dirs_to_clean = [
        'models/saved_model',
        'models/__pycache__',
        'scripts/__pycache__',
        'utils/__pycache__',
    ]
    
    removed_files = []
    removed_dirs = []
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            removed_files.append(file)
            
    # Remove directories
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            removed_dirs.append(dir_path)
    
    print(f"✅ Cleanup complete!")
    print(f"   Removed {len(removed_files)} redundant files")
    print(f"   Removed {len(removed_dirs)} cache directories")
    
    if removed_files:
        print("\n🗑️  Files removed:")
        for file in removed_files[:10]:  # Show first 10
            print(f"   - {file}")
        if len(removed_files) > 10:
            print(f"   ... and {len(removed_files) - 10} more")
    
    return removed_files, removed_dirs

def show_clean_structure():
    """Show the clean file structure"""
    
    print("\n📁 Clean Project Structure:")
    print("=" * 50)
    
    structure = {
        "🎯 MAIN SYSTEM": ["stock_predictor.py"],
        "📊 DATA & CONFIG": ["config.py"],
        "🤖 MODELS": ["models/transformer_model.py"],
        "📈 DATA COLLECTION": ["scripts/data_collection.py", "scripts/training.py"],
        "🛠️ UTILITIES": ["utils/utils.py"],
        "📋 DOCS": ["README.md", "CLAUDE.md", "requirements.txt"]
    }
    
    for category, files in structure.items():
        print(f"\n{category}:")
        for file in files:
            status = "✅" if os.path.exists(file) else "❌"
            print(f"  {status} {file}")

def main():
    print("🧹 CLEANUP REDUNDANT FILES")
    print("=" * 50)
    print("This will remove overlapping/redundant model files and scripts")
    print("Keeping only the essential, unified system.")
    print()
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled")
        return
    
    # Perform cleanup
    removed_files, removed_dirs = cleanup_redundant_files()
    
    # Show clean structure
    show_clean_structure()
    
    print(f"\n🎉 PROJECT CLEANED UP!")
    print("=" * 30)
    print("✅ Now you have ONE unified system:")
    print("   📍 Use: python stock_predictor.py")
    print("   📍 Or: python stock_predictor.py train --symbol AAPL")
    print("   📍 Or: python stock_predictor.py predict --days 7")
    
    print(f"\n🗑️ Summary:")
    print(f"   - Removed {len(removed_files)} redundant files")
    print(f"   - Removed {len(removed_dirs)} cache directories") 
    print(f"   - Kept only essential files")
    print(f"   - One unified system replaces all scripts")

if __name__ == "__main__":
    main()