# 📖 Setup and Deployment Guide

## Table of Contents
1. [Local Setup](#local-setup)
2. [Running the Application](#running-the-application)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Troubleshooting](#troubleshooting)

---

## Local Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/instagram-fake-account-detection.git
cd instagram-fake-account-detection
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python train_model.py
```

This will:
- Process the training data
- Train multiple ML models
- Save the best model to `models/` directory
- Generate evaluation metrics

---

## Running the Application

### Start the Streamlit App
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Single Prediction Tab**
   - Enter account features manually
   - Click "Analyze Account"
   - View prediction results and risk indicators

2. **Batch Prediction Tab**
   - Upload a CSV file with account data
   - Click "Analyze All Accounts"
   - Download results with predictions

3. **Analytics Tab**
   - View feature importance
   - Compare model performance
   - Understand prediction factors

---

## Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

Ensure your repository has:
- ✅ `app.py` (main Streamlit app)
- ✅ `requirements.txt` (dependencies)
- ✅ `data/` folder with datasets
- ✅ Pre-trained models in `models/` folder

### Step 2: Deploy to Streamlit Cloud

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Go to Streamlit Cloud**
   - Visit https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"

3. **Configure Deployment**
   - Repository: `yourusername/instagram-fake-account-detection`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Streamlit will install dependencies
   - Your app will be live at `https://yourusername-instagram-fake-detection.streamlit.app`

### Step 3: Important Notes

⚠️ **Model Files**: Make sure pre-trained models are committed to the repository
```bash
git add models/
git commit -m "Add pre-trained models"
git push
```

⚠️ **Data Files**: Ensure CSV files are in the repository
```bash
git add data/
git commit -m "Add datasets"
git push
```

---

## Troubleshooting

### Common Issues

#### 1. Module Not Found Error
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

#### 2. Model File Not Found
```bash
# Solution: Train the model first
python train_model.py
```

#### 3. Port Already in Use
```bash
# Solution: Use a different port
streamlit run app.py --server.port 8502
```

#### 4. Streamlit Cloud Deployment Issues

**Issue**: App crashes on startup  
**Solution**: Check that all model files and data files are committed

**Issue**: Import errors  
**Solution**: Verify `requirements.txt` has all dependencies with correct versions

**Issue**: Memory errors  
**Solution**: Reduce model size or optimize data loading

---

## Configuration

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#E4405F"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

---

## Performance Tips

1. **Use @st.cache_resource** for loading models
2. **Optimize data processing** with vectorized operations
3. **Reduce memory usage** by processing in batches
4. **Enable compression** for large CSV uploads

---

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Plotly Documentation](https://plotly.com/python)

---

## Support

For issues and questions:
- 🐛 Report bugs: [GitHub Issues](https://github.com/yourusername/instagram-fake-account-detection/issues)
- 📧 Email: your.email@example.com
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/instagram-fake-account-detection/discussions)

---

**Happy Detecting! 🔍**
