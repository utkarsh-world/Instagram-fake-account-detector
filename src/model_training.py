"""
Model Training Module
Handles model training, evaluation, and saving for Instagram Fake Account Detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pickle
import json

class ModelTrainer:
    """Class for training and evaluating machine learning models"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}

    def initialize_models(self):
        """Initialize multiple models for comparison"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                max_depth=5
            )
        }
        print(f"✓ Initialized {len(self.models)} models for training")

    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        if model_name not in self.models:
            print(f"✗ Model {model_name} not found")
            return None

        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        print(f"✓ {model_name} trained successfully")
        return model

    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
            'f1_score': f1_score(y_test, y_pred, pos_label=1),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        self.results[model_name] = metrics

        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate all models"""
        self.initialize_models()

        for model_name in self.models.keys():
            model = self.train_model(model_name, X_train, y_train)
            self.evaluate_model(model, model_name, X_val, y_val)

        # Select best model based on F1-score
        best_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name

        print(f"\n{'='*60}")
        print(f"Best Model: {best_name}")
        print(f"F1-Score: {self.results[best_name]['f1_score']:.4f}")
        print(f"{'='*60}")

        return self.best_model, self.best_model_name

    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            return feature_importance
        else:
            print("Model does not support feature importance")
            return None

    def save_model(self, filepath):
        """Save the best model to disk"""
        if self.best_model is None:
            print("✗ No model to save. Train models first.")
            return

        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Best model ({self.best_model_name}) saved to {filepath}")

    def save_results(self, filepath):
        """Save training results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✓ Results saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            self.best_model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return self.best_model

    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            print("✗ No model loaded. Train or load a model first.")
            return None

        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1] if hasattr(self.best_model, 'predict_proba') else None

        return predictions, probabilities

def print_classification_report(y_true, y_pred, target_names=['Genuine', 'Fake']):
    """Print detailed classification report"""
    print("\nDetailed Classification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))

def print_confusion_matrix(y_true, y_pred):
    """Print confusion matrix with labels"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("=" * 60)
    print(f"                Predicted")
    print(f"                Genuine  Fake")
    print(f"Actual Genuine    {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"       Fake       {cm[1][0]:3d}    {cm[1][1]:3d}")
    print("=" * 60)
