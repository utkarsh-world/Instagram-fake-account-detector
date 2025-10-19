"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering for Instagram Fake Account Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class DataPreprocessor:
    """Class for preprocessing Instagram account data"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None

    def check_data_quality(self, df):
        """Check for missing values and data quality issues"""
        print("\nData Quality Check:")
        print("-" * 50)
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nDuplicate rows: {df.duplicated().sum()}")
        return df

    def create_derived_features(self, df):
        """Create additional features from existing ones"""
        df_copy = df.copy()

        # Follower to following ratio
        df_copy['follower_following_ratio'] = df_copy.apply(
            lambda x: x['#followers'] / (x['#follows'] + 1), axis=1
        )

        # Post engagement rate (posts per follower)
        df_copy['post_per_follower'] = df_copy.apply(
            lambda x: x['#posts'] / (x['#followers'] + 1), axis=1
        )

        # Profile completeness score
        df_copy['profile_completeness'] = (
            df_copy['profile pic'] + 
            (df_copy['description length'] > 0).astype(int) +
            df_copy['external URL']
        ) / 3.0

        print("✓ Derived features created:")
        print("  - follower_following_ratio")
        print("  - post_per_follower")
        print("  - profile_completeness")

        return df_copy

    def prepare_features(self, df, target_col='fake', include_derived=True):
        """Prepare features and target for modeling"""
        if include_derived:
            df = self.create_derived_features(df)

        # Separate features and target
        if target_col in df.columns:
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            self.feature_names = X.columns.tolist()
            return X, y
        else:
            self.feature_names = df.columns.tolist()
            return df, None

    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and validation sets"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✓ Data split: {len(X_train)} train, {len(X_val)} validation")
        return X_train, X_val, y_train, y_val

    def save_scaler(self, filepath):
        """Save the fitted scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {filepath}")

    def load_scaler(self, filepath):
        """Load a fitted scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {filepath}")

def get_feature_descriptions():
    """Return descriptions of all features"""
    descriptions = {
        'profile pic': 'Binary indicator (1 if profile picture exists, 0 otherwise)',
        'nums/length username': 'Ratio of numerical characters to username length',
        'fullname words': 'Number of words in the full name',
        'nums/length fullname': 'Ratio of numerical characters to fullname length',
        'name==username': 'Binary indicator (1 if name matches username exactly)',
        'description length': 'Character count of bio/description',
        'external URL': 'Binary indicator (1 if external URL present)',
        'private': 'Binary indicator (1 if account is private)',
        '#posts': 'Total number of posts',
        '#followers': 'Total number of followers',
        '#follows': 'Total number of accounts followed',
        'follower_following_ratio': 'Ratio of followers to following',
        'post_per_follower': 'Average posts per follower',
        'profile_completeness': 'Score indicating profile completeness (0-1)'
    }
    return descriptions
