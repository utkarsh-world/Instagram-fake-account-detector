"""
Instagram Fake Account Detector
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Instagram Fake Account Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E4405F;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .fake-account {
        background-color: #ff4444;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
    }
    .genuine-account {
        background-color: #00C851;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def create_derived_features(data):
    """Create derived features from input data"""
    # Follower to following ratio
    data['follower_following_ratio'] = data['#followers'] / (data['#follows'] + 1)

    # Post engagement rate
    data['post_per_follower'] = data['#posts'] / (data['#followers'] + 1)

    # Profile completeness
    data['profile_completeness'] = (
        data['profile pic'] + 
        (data['description length'] > 0) +
        data['external URL']
    ) / 3.0

    return data

def predict_account(model, scaler, features_df):
    """Make prediction on account features"""
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">🔍 Instagram Fake Account Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect fake and spam Instagram accounts using Machine Learning</div>', unsafe_allow_html=True)

    # Load model
    model, scaler, metadata = load_model_and_scaler()

    if model is None:
        st.error("⚠️ Model not found. Please run 'python train_model.py' first to train the model.")
        return

    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Model Type", metadata['best_model'])
        st.metric("Test Accuracy", f"{metadata['test_accuracy']:.2%}")
        st.metric("F1-Score", f"{metadata['test_f1_score']:.2%}")

        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            "This app uses machine learning to detect fake Instagram accounts based on "
            "profile characteristics, username patterns, and account activity metrics."
        )

        st.markdown("---")
        st.header("🎯 Features Used")
        st.write(f"Total Features: {metadata['num_features']}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Single Prediction", "📤 Batch Prediction", "📈 Analytics", "ℹ️ Feature Guide"])

    with tab1:
        st.header("Single Account Prediction")
        st.write("Enter account features to check if an Instagram account is genuine or fake.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Profile Information")
            profile_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])
            profile_pic_val = 1 if profile_pic == "Yes" else 0

            external_url = st.selectbox("Has External URL?", ["Yes", "No"])
            external_url_val = 1 if external_url == "Yes" else 0

            private = st.selectbox("Private Account?", ["Yes", "No"])
            private_val = 1 if private == "Yes" else 0

        with col2:
            st.subheader("Username & Name")
            nums_username = st.slider("Numeric Ratio in Username", 0.0, 1.0, 0.0, 0.01)
            fullname_words = st.number_input("Number of Words in Full Name", 0, 20, 2)
            nums_fullname = st.slider("Numeric Ratio in Full Name", 0.0, 1.0, 0.0, 0.01)
            name_equals_username = st.selectbox("Name equals Username?", ["No", "Yes"])
            name_equals_val = 1 if name_equals_username == "Yes" else 0

        with col3:
            st.subheader("Activity Metrics")
            description_length = st.number_input("Bio Length (characters)", 0, 200, 50)
            num_posts = st.number_input("Number of Posts", 0, 10000, 100)
            num_followers = st.number_input("Number of Followers", 0, 10000000, 1000)
            num_follows = st.number_input("Number of Following", 0, 10000, 500)

        if st.button("🔍 Analyze Account", type="primary", use_container_width=True):
            # Create feature dictionary
            features = {
                'profile pic': profile_pic_val,
                'nums/length username': nums_username,
                'fullname words': fullname_words,
                'nums/length fullname': nums_fullname,
                'name==username': name_equals_val,
                'description length': description_length,
                'external URL': external_url_val,
                'private': private_val,
                '#posts': num_posts,
                '#followers': num_followers,
                '#follows': num_follows
            }

            # Create DataFrame and add derived features
            features_df = pd.DataFrame([features])
            features_df = create_derived_features(features_df)

            # Make prediction
            prediction, probability = predict_account(model, scaler, features_df)

            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if prediction == 1:
                    st.markdown('<div class="fake-account">⚠️ FAKE ACCOUNT DETECTED</div>', unsafe_allow_html=True)
                    confidence = probability[1] * 100
                else:
                    st.markdown('<div class="genuine-account">✅ GENUINE ACCOUNT</div>', unsafe_allow_html=True)
                    confidence = probability[0] * 100

                st.markdown(f"<h3 style='text-align: center; margin-top: 1rem;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)

            # Probability gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "Fake Account Probability"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Risk indicators
            st.subheader("🚨 Risk Indicators")
            risk_factors = []

            if profile_pic_val == 0:
                risk_factors.append("❌ No profile picture")
            if nums_username > 0.2:
                risk_factors.append("❌ High numeric ratio in username")
            if description_length < 10:
                risk_factors.append("❌ Very short or no bio")
            if external_url_val == 0:
                risk_factors.append("⚠️ No external URL")
            if num_posts < 10:
                risk_factors.append("❌ Very few posts")
            if num_followers < 50:
                risk_factors.append("❌ Very few followers")

            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("✅ No major risk indicators detected")

    with tab2:
        st.header("Batch Prediction")
        st.write("Upload a CSV file with multiple accounts to analyze in batch.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} accounts")

                st.subheader("Preview Data")
                st.dataframe(df.head())

                if st.button("🔍 Analyze All Accounts", type="primary"):
                    # Add derived features
                    df_processed = create_derived_features(df.copy())

                    # Remove target column if exists
                    if 'fake' in df_processed.columns:
                        X = df_processed.drop('fake', axis=1)
                    else:
                        X = df_processed

                    # Make predictions
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]

                    # Add predictions to dataframe
                    df['Prediction'] = ['Fake' if p == 1 else 'Genuine' for p in predictions]
                    df['Confidence'] = probabilities * 100

                    # Display results
                    st.subheader("📊 Results Summary")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Accounts", len(df))
                    with col2:
                        st.metric("Fake Accounts", sum(predictions == 1))
                    with col3:
                        st.metric("Genuine Accounts", sum(predictions == 0))

                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(df)

                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fake_account_predictions.csv",
                        mime="text/csv"
                    )

                    # Visualization
                    fig = px.pie(
                        values=[sum(predictions == 0), sum(predictions == 1)],
                        names=['Genuine', 'Fake'],
                        title="Account Distribution",
                        color_discrete_sequence=['#00C851', '#ff4444']
                    )
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    with tab3:
        st.header("Model Analytics")

        try:
            # Load feature importance
            feature_importance = pd.read_csv('models/feature_importance.csv')

            st.subheader("🎯 Feature Importance")
            st.write("The most important features for detecting fake accounts:")

            fig = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Feature Importance Table")
            st.dataframe(feature_importance, use_container_width=True)

        except:
            st.warning("Feature importance data not available. Train the model first.")

        try:
            # Load training results
            with open('models/training_results.json', 'r') as f:
                results = json.load(f)

            st.subheader("📈 Model Comparison")

            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })

            comparison_df = pd.DataFrame(comparison_data)

            fig = px.bar(
                comparison_df,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title="Model Performance Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        except:
            st.warning("Training results not available.")

    with tab4:
        st.header("Feature Guide")
        st.write("Understanding the features used for detection:")

        features_info = {
            "Profile Picture": "Presence of a profile picture. Fake accounts often lack profile pictures.",
            "Numeric Ratio in Username": "Proportion of numbers in username. Fake accounts tend to have more numbers.",
            "Full Name Words": "Number of words in the full name field.",
            "Numeric Ratio in Full Name": "Proportion of numbers in full name.",
            "Name equals Username": "Whether the name and username are identical.",
            "Description Length": "Length of bio in characters. Fake accounts usually have short or empty bios.",
            "External URL": "Presence of external link. Genuine accounts more likely to have links.",
            "Private Account": "Whether the account is private or public.",
            "Number of Posts": "Total posts. Fake accounts typically have very few posts.",
            "Number of Followers": "Total followers. Fake accounts usually have low follower counts.",
            "Number of Following": "Total accounts followed.",
            "Follower/Following Ratio": "Derived metric showing balance between followers and following.",
            "Post per Follower": "Derived metric showing engagement rate.",
            "Profile Completeness": "Derived score indicating how complete the profile is."
        }

        for feature, description in features_info.items():
            with st.expander(f"📌 {feature}"):
                st.write(description)

        st.markdown("---")
        st.subheader("🎯 Red Flags for Fake Accounts")
        st.error("❌ No profile picture")
        st.error("❌ Username contains many numbers (e.g., user12345678)")
        st.error("❌ Empty or very short bio (< 10 characters)")
        st.error("❌ No external URL in bio")
        st.error("❌ Very few or zero posts")
        st.error("❌ Very low follower count (< 50)")
        st.error("❌ Unusual follower-to-following ratio")

if __name__ == "__main__":
    main()
