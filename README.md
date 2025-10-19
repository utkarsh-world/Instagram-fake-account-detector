# 🔍 Instagram Fake Account Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

**A Machine Learning project to detect fake and spam Instagram accounts with 93%+ accuracy**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Results & Insights](#-results--insights)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **machine learning-based system** to detect fake and spam Instagram accounts. Using various profile characteristics, username patterns, and activity metrics, the model achieves **93.33% accuracy** in identifying fraudulent accounts.

The project includes:
- 🤖 **Multiple ML Models** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- 🌐 **Interactive Streamlit Web App** for real-time predictions
- 📊 **Comprehensive Data Analysis** and visualization
- 📦 **Production-ready code** with modular architecture

---

## 🚨 Problem Statement

Fake and spam accounts have become a major problem on social media platforms, including Instagram:

- 👥 **Fake Followers**: Users create fake accounts to inflate follower counts
- 🛒 **Spam & Scams**: Fake accounts promote fraudulent products and services  
- 🎭 **Impersonation**: Criminals impersonate celebrities and regular users
- 📢 **Misinformation**: Fake accounts spread false information and propaganda
- 💔 **Trust Issues**: Damage platform credibility and user trust

This project aims to automatically detect such accounts using machine learning.

---

## ✨ Features

### 🎯 Core Features
- ✅ **Single Account Prediction**: Analyze individual accounts in real-time
- 📤 **Batch Processing**: Upload CSV files to analyze multiple accounts
- 📊 **Interactive Dashboard**: Beautiful Streamlit UI with visualizations
- 🎨 **Feature Importance Analysis**: Understand which features matter most
- 📈 **Model Comparison**: Compare performance of multiple ML algorithms

### 🔍 Detection Criteria

The model analyzes **14 key features**:

| Feature | Description |
|---------|-------------|
| 🖼️ **Profile Picture** | Presence of profile picture |
| 🔢 **Username Numbers** | Ratio of numbers in username |
| 📝 **Full Name** | Number of words and numeric ratio |
| 📄 **Bio Length** | Character count of description |
| 🔗 **External URL** | Presence of external link |
| 🔒 **Privacy** | Public vs private account |
| 📸 **Posts** | Total number of posts |
| 👥 **Followers** | Total follower count |
| 👤 **Following** | Total following count |
| 📊 **Engagement Ratio** | Followers/Following ratio |
| 💬 **Post Engagement** | Posts per follower |
| ✅ **Profile Completeness** | Overall profile quality score |

---

## 📊 Dataset

**Source**: Custom dataset collected from Instagram accounts  
**Total Samples**: 696 accounts  
**Training Set**: 576 accounts (50% fake, 50% genuine)  
**Test Set**: 120 accounts (50% fake, 50% genuine)

### Class Distribution
```
Genuine Accounts: 348 (50%)
Fake Accounts:    348 (50%)
```

### Key Statistics

| Metric | Genuine Accounts | Fake Accounts |
|--------|-----------------|--------------|
| **Profile Picture** | 99.3% have | 41.0% have |
| **Avg Bio Length** | 40 chars | 5.3 chars |
| **Median Posts** | 74 | 0 |
| **Median Followers** | 662 | 40 |
| **External URL** | 23.3% have | 0% have |

---

## 🏗️ Model Architecture

### Models Implemented

1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Non-linear decision boundaries
3. **Random Forest** ⭐ - Best performer (ensemble of trees)
4. **Gradient Boosting** - Advanced ensemble method

### Training Pipeline

```
Data Loading → Feature Engineering → Scaling → Model Training → Evaluation → Deployment
```

### Feature Engineering

The project creates **3 derived features** to enhance prediction:

1. **Follower-Following Ratio**: Indicates account popularity vs activity
2. **Post per Follower**: Measures engagement quality
3. **Profile Completeness Score**: Composite metric of profile quality (0-1)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/instagram-fake-account-detection.git
cd instagram-fake-account-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1️⃣ Train the Model

Train all models and save the best one:

```bash
python train_model.py
```

This will:
- Load and preprocess the training data
- Train multiple ML models
- Evaluate and compare performance
- Save the best model to `models/` directory
- Generate feature importance analysis

**Output**: 
- `models/best_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/feature_importance.csv` - Feature rankings
- `models/training_results.json` - All model metrics

### 2️⃣ Run the Web Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3️⃣ Using the Web Interface

#### Single Prediction
1. Navigate to the **"Single Prediction"** tab
2. Enter account features (profile pic, followers, posts, etc.)
3. Click **"Analyze Account"**
4. View prediction with confidence score and risk indicators

#### Batch Prediction
1. Navigate to the **"Batch Prediction"** tab
2. Upload a CSV file with account data
3. Click **"Analyze All Accounts"**
4. Download results with predictions

#### Analytics
- View feature importance rankings
- Compare model performance metrics
- Understand which features drive predictions

---

## 📈 Model Performance

### Best Model: Random Forest Classifier

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 93.33% |
| **Precision (Fake)** | 94.83% |
| **Recall (Fake)** | 91.67% |
| **F1-Score** | 93.22% |
| **ROC-AUC** | 96.50% |

### Confusion Matrix (Test Set)

```
                Predicted
                Genuine  Fake
Actual Genuine    57      3
       Fake        5     55
```

### Top 5 Most Important Features

1. 👥 **#Followers** (29.62%) - Most predictive feature
2. 📸 **#Posts** (24.35%) - Strong indicator of activity
3. 🔢 **Username Numbers** (12.23%) - Pattern detection
4. 🖼️ **Profile Picture** (10.89%) - Basic verification
5. 👤 **#Following** (7.80%) - Network behavior

---

## 📁 Project Structure

```
instagram-fake-account-detection/
│
├── 📂 data/
│   ├── train.csv              # Training dataset
│   └── test.csv               # Testing dataset
│
├── 📂 models/
│   ├── best_model.pkl         # Trained model
│   ├── scaler.pkl             # Feature scaler
│   ├── feature_importance.csv # Feature rankings
│   ├── training_results.json  # Model metrics
│   └── model_metadata.json    # Model info
│
├── 📂 src/
│   ├── data_preprocessing.py  # Data preprocessing module
│   └── model_training.py      # Model training module
│
├── 📂 notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebook
│
├── 📂 images/
│   └── screenshots/           # App screenshots
│
├── 📄 app.py                  # Streamlit web application
├── 📄 train_model.py          # Model training script
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md              # This file
├── 📄 LICENSE                # MIT License
└── 📄 .gitignore             # Git ignore rules
```

---

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms and tools
- **Matplotlib & Seaborn** - Data visualization

### Web Application
- **Streamlit** - Interactive web framework
- **Plotly** - Interactive charts and graphs

### Development Tools
- **Git** - Version control
- **Jupyter Notebook** - Interactive development
- **pickle** - Model serialization

---

## 💡 Results & Insights

### Key Findings

#### 🚨 Red Flags for Fake Accounts:
- ❌ **No profile picture** (58.3 percentage point difference)
- ❌ **Username with many numbers** (7.6x higher ratio)
- ❌ **Empty or very short bio** (7.5x shorter than genuine)
- ❌ **No external URL** (0% vs 23.3%)
- ❌ **Zero posts** (median of 0 vs 74)
- ❌ **Very few followers** (40 vs 662 median)

#### ✅ Genuine Account Characteristics:
- ✔️ Profile picture present (99.3%)
- ✔️ Meaningful username with few/no numbers
- ✔️ Detailed bio with personality
- ✔️ Regular posting activity
- ✔️ Healthy follower count
- ✔️ Balanced follower-following ratio

### Business Impact

This model can be used for:
- 🏢 **Platform Security**: Automated fake account detection systems
- 🛡️ **Brand Protection**: Identify accounts impersonating brands
- 📱 **Influencer Marketing**: Verify authenticity before partnerships
- 🔒 **Fraud Prevention**: Flag suspicious account creation patterns
- 📊 **Analytics**: Clean follower data for accurate metrics

---

## 🚀 Future Enhancements

### Planned Features
- [ ] **Deep Learning Models**: Implement LSTM/Transformer networks
- [ ] **Image Analysis**: Use CNN to verify profile picture authenticity
- [ ] **Text Analysis**: NLP on bio text and post captions
- [ ] **Temporal Features**: Include account age and activity patterns
- [ ] **API Integration**: Real-time Instagram API connection
- [ ] **Multi-platform**: Extend to Twitter, Facebook, TikTok
- [ ] **Deployment**: Docker containerization and cloud deployment
- [ ] **Mobile App**: React Native mobile application

### Model Improvements
- Hyperparameter tuning with GridSearchCV
- SMOTE for handling real-world class imbalance
- Ensemble stacking of multiple models
- Active learning for continuous improvement

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌟 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation for significant changes
- Ensure all tests pass before submitting PR

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Your Name**  
📧 Email: your.email@example.com  
🔗 LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)  
💼 Portfolio: [Your Portfolio](https://yourportfolio.com)  
🐱 GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Dataset inspired by real Instagram account patterns
- Built with ❤️ using Streamlit and Scikit-learn
- Thanks to the open-source community

---

## ⭐ Show Your Support

If you find this project useful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs and issues
- 💡 Suggesting new features
- 📢 Sharing with others

---

<div align="center">

**Made with ❤️ and Python**

[⬆ Back to Top](#-instagram-fake-account-detection)

</div>
