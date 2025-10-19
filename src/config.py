"""
Configuration settings for the Instagram Fake Account Detection project
"""

# Model settings
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    },
    'decision_tree': {
        'max_depth': 10,
        'random_state': 42
    },
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42
    }
}

# Data paths
DATA_PATHS = {
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}

# Model paths
MODEL_PATHS = {
    'best_model': 'models/best_model.pkl',
    'scaler': 'models/scaler.pkl',
    'metadata': 'models/model_metadata.json',
    'results': 'models/training_results.json',
    'feature_importance': 'models/feature_importance.csv'
}

# Feature engineering
DERIVED_FEATURES = [
    'follower_following_ratio',
    'post_per_follower',
    'profile_completeness'
]

# Original features
ORIGINAL_FEATURES = [
    'profile pic',
    'nums/length username',
    'fullname words',
    'nums/length fullname',
    'name==username',
    'description length',
    'external URL',
    'private',
    '#posts',
    '#followers',
    '#follows'
]

# Training parameters
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Streamlit app settings
APP_CONFIG = {
    'page_title': 'Instagram Fake Account Detector',
    'page_icon': '🔍',
    'layout': 'wide',
    'primary_color': '#E4405F'
}
