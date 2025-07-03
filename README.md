# 💰 Bank Marketing Subscription Prediction

A machine learning-powered web application that predicts whether a bank client will subscribe to a term deposit after a marketing campaign. Built using a Decision Tree classifier trained on real-world data from a Portuguese bank marketing campaign.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)

## 🎯 Project Overview

This project addresses the challenge of predicting customer behavior in bank marketing campaigns. Using historical campaign data, the model helps banks identify which clients are most likely to subscribe to term deposits, enabling more targeted and efficient marketing strategies.

## ✨ Features

- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Predictions**: Instant subscription likelihood prediction
- **Comprehensive Data Processing**: Handles missing values, categorical encoding, and feature engineering
- **High Performance Model**: Optimized Decision Tree with perfect recall (1.0) for positive class
- **Professional UI**: Modern interface with animations and responsive design
- **Educational Content**: Built-in help system explaining features and model performance

## 🚀 Quick Start

### Prerequisites

- Python 3.11+


### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bank-marketing-predictor.git
cd bank-marketing-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Dataset

The project uses the Bank Marketing dataset from a Portuguese banking institution, containing:
- **4,119 total records** (after balancing)
- **21 features** including client demographics, campaign details, and economic indicators
- **Target variable**: Term deposit subscription (yes/no)

### Key Features
- **Client Information**: Age, job, marital status, education
- **Campaign Data**: Contact type, duration, number of contacts
- **Previous Campaigns**: Outcome and frequency of previous contacts
- **Economic Indicators**: Employment rate, consumer price index, Euribor rates

## 🤖 Model Performance

Our Decision Tree classifier achieved excellent performance metrics:

| Metric | Score |
|--------|-------|
| **F1 Score (Class 1)** | 0.9633 |
| **Recall (Class 1)** | 1.0000 |
| **Weighted F1 Score** | 0.9618 |
| **Cross-Validation Score** | 0.9565 |

### Why Decision Tree?

✅ **Perfect Recall (1.0)**: Captures all actual subscribers - no missed opportunities  
✅ **Strong F1 Score**: Excellent balance between precision and recall  
✅ **Business-Friendly**: Interpretable results for marketing teams  
✅ **Handles Imbalanced Data**: Optimized for minority class performance  

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
1. **Missing Value Handling**: Replace 'unknown' values with mode
2. **Outlier Detection**: Statistical analysis and visualization
3. **Class Balancing**: Upsampling minority class using resampling
4. **Feature Engineering**: 
   - Previous contact indicator
   - Modified pdays feature
   - Campaign-previous interaction terms
5. **Encoding & Scaling**: One-hot encoding for categories, standardization for numerics

### Model Selection Process
Evaluated multiple algorithms:
- Logistic Regression
- K-Nearest Neighbors
- **Decision Tree** ⭐ (Selected)
- Random Forest
- Gradient Boosting

### Hyperparameter Optimization
- **GridSearchCV** with 5-fold cross-validation
- **Optimized Parameters**:
  - Criterion: entropy
  - Max depth: None (unlimited)
  - Min samples split: 2
  - Min samples leaf: 1

## 📁 Project Structure

```
bank-marketing-predictor/
├── ADA442_Project.ipynb    # Complete data analysis and model development
├── app.py                  # Streamlit web application
├── model.pkl              # Trained Decision Tree model
├── bank-additional.csv    # Dataset
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
├── hello.json            # Animation assets
├── machine.json          # Animation assets
└── README.md             # Project documentation
```

## 💻 Usage

### Web Application

1. **Navigate to "Use Model"** in the sidebar
2. **Fill in client information**:
   - Demographics (age, job, education, etc.)
   - Contact details and campaign data
   - Economic indicators
3. **Click "Predict"** to get instant results
4. **View prediction** with color-coded results

### Jupyter Notebook

The `ADA442_Project.ipynb` contains the complete data science workflow:
- Exploratory Data Analysis
- Data preprocessing and cleaning
- Model comparison and evaluation
- Hyperparameter tuning
- Performance analysis

## 🔧 Configuration

### Environment Variables
No environment variables required for local development.

### Model Retraining
To retrain the model with new data:
1. Replace `bank-additional.csv` with your dataset
2. Run the Jupyter notebook `ADA442_Project.ipynb`
3. The new model will be saved as `model.pkl`

## 📋 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **joblib**: Model persistence
- **streamlit-lottie**: Animation support
- **imbalanced-learn**: Handling imbalanced datasets
