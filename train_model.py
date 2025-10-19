"""
Main Training Script
Trains and evaluates models for Instagram Fake Account Detection
"""

import sys
import os
sys.path.append('src')

from data_preprocessing import DataPreprocessor, get_feature_descriptions
from model_training import ModelTrainer, print_classification_report, print_confusion_matrix
import pandas as pd
import numpy as np

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("INSTAGRAM FAKE ACCOUNT DETECTION - MODEL TRAINING")
    print("=" * 80)

    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()

    # Step 1: Load Data
    print("\n[STEP 1] Loading Data...")
    print("-" * 80)
    train_df = preprocessor.load_data('data/train.csv')
    test_df = preprocessor.load_data('data/test.csv')

    if train_df is None or test_df is None:
        print("✗ Failed to load data. Exiting.")
        return

    # Step 2: Data Quality Check
    print("\n[STEP 2] Checking Data Quality...")
    print("-" * 80)
    train_df = preprocessor.check_data_quality(train_df)

    # Step 3: Prepare Features
    print("\n[STEP 3] Preparing Features...")
    print("-" * 80)
    X_train_full, y_train_full = preprocessor.prepare_features(train_df, include_derived=True)
    X_test, y_test = preprocessor.prepare_features(test_df, include_derived=True)

    print(f"\nFeature set: {X_train_full.shape[1]} features")
    print(f"Features: {', '.join(preprocessor.feature_names)}")

    # Step 4: Split Data
    print("\n[STEP 4] Splitting Data...")
    print("-" * 80)
    X_train, X_val, y_train, y_val = preprocessor.split_data(X_train_full, y_train_full)

    # Step 5: Scale Features
    print("\n[STEP 5] Scaling Features...")
    print("-" * 80)
    X_train_scaled, X_val_scaled = preprocessor.scale_features(X_train, X_val)
    X_test_scaled = preprocessor.scaler.transform(X_test)
    print("✓ Features scaled using StandardScaler")

    # Step 6: Train Models
    print("\n[STEP 6] Training Models...")
    print("-" * 80)
    best_model, best_name = trainer.train_all_models(X_train_scaled, y_train, X_val_scaled, y_val)

    # Step 7: Final Evaluation on Test Set
    print("\n[STEP 7] Final Evaluation on Test Set...")
    print("-" * 80)
    test_metrics = trainer.evaluate_model(best_model, best_name, X_test_scaled, y_test)

    # Print detailed results
    y_pred = best_model.predict(X_test_scaled)
    print_classification_report(y_test, y_pred)
    print_confusion_matrix(y_test, y_pred)

    # Step 8: Feature Importance
    print("\n[STEP 8] Feature Importance Analysis...")
    print("-" * 80)
    feature_importance = trainer.get_feature_importance(preprocessor.feature_names)
    if feature_importance is not None:
        print(feature_importance.to_string(index=False))
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        print("\n✓ Feature importance saved to models/feature_importance.csv")

    # Step 9: Save Models and Results
    print("\n[STEP 9] Saving Models and Results...")
    print("-" * 80)
    trainer.save_model('models/best_model.pkl')
    trainer.save_results('models/training_results.json')
    preprocessor.save_scaler('models/scaler.pkl')

    # Save model metadata
    metadata = {
        'best_model': best_name,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_score': test_metrics['f1_score'],
        'features': preprocessor.feature_names,
        'num_features': len(preprocessor.feature_names)
    }

    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✓ Model metadata saved to models/model_metadata.json")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest Model: {best_name}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print("\nAll models and results saved in 'models/' directory")

if __name__ == "__main__":
    main()
