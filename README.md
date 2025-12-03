# Predicting Student Residency Status from Shopping Behaviors

A comprehensive data science project analyzing UCSD student shopping and package delivery behaviors to predict on-campus vs. off-campus residency status.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project investigates whether student residency status (on-campus vs. off-campus) can be predicted based on their shopping and package delivery behaviors. The analysis includes comprehensive data cleaning, exploratory data analysis, statistical testing, and multiple machine learning approaches.

**Key Highlights:**
- ✅ Systematic checking of **all logistic regression assumptions**
- ✅ Feature engineering with **160+ features** from survey data
- ✅ **VIF-based feature selection** to address multicollinearity
- ✅ **PCA dimensionality reduction** with multiple configurations
- ✅ Comparison of 6+ machine learning models
- ✅ Achieved **80% prediction accuracy**

---

## ❓ Research Question

**Can we predict whether a student lives on-campus or off-campus based on their shopping and package delivery behaviors?**

### Why This Matters:
- 📦 **Campus resource planning** - optimize package handling facilities
- 🛍️ **Targeted marketing** - understand student shopping patterns
- 🏫 **Service improvement** - better campus store offerings
- 📊 **Behavioral insights** - understand residential differences

---

## 📊 Dataset

- **Source**: UCSD student survey on shopping and delivery behaviors
- **Size**: 197 observations
- **Original Features**: 19 survey questions
- **Engineered Features**: 160+ features (one-hot encoded, TF-IDF)
- **Target Variable**: Binary (On-Campus vs. Off-Campus)
- **Class Balance**: 57.4% on-campus, 42.6% off-campus

### Key Variables:
- Package delivery frequency
- Item types delivered/purchased
- Store preferences (on-campus vs. off-campus)
- Shopping reasons and transportation issues
- Communication preferences

---

## 📁 Project Structure

```
DSC190_final_project/
│
├── data.csv                    # Original survey data
├── data_cleaned.csv            # Processed data with engineered features
├── analysis.ipynb              # Main analysis notebook
├── data_cleaning.py            # Data preprocessing pipeline
├── ANALYSIS_UPDATES.md         # Detailed changelog and updates
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── requirements.txt            # Python dependencies (if needed)
```

---

## 🔑 Key Findings

### 1. **Predictability**
- ✅ **Student residency can be predicted with 80% accuracy**
- Shopping and delivery behaviors are strong indicators of housing status

### 2. **Most Important Predictors**
Top 5 features for prediction:
1. **Transportation issues** (limited/no car access)
2. **Package delivery frequency** to campus
3. **Items delivered to campus** (general category)
4. **Store preference** (on-campus vs. off-campus)
5. **Specific item categories** (from TF-IDF analysis)

### 3. **Statistical Significance**
- **Chi-square tests**: All categorical variables significantly associated with residency (p < 0.05)
- **Mann-Whitney U tests**: 10/10 top features show significant differences
- Strong evidence for behavioral differences between groups

### 4. **Model Performance Comparison**

| Model | Features | Test Accuracy | ROC AUC | EPV | Notes |
|-------|----------|--------------|---------|-----|-------|
| **Random Forest (Basic)** | 160 | **80.0%** | 0.771 | 0.53 | Best accuracy |
| **Logistic Regression (Tuned)** | 160 | 77.5% | 0.724 | 0.53 | Good baseline |
| **VIF-Reduced Features** | ~100-120 | ~75-78% | ~0.72 | >1.0 | Better assumptions |
| **PCA-73 (90% variance)** | 73 | ~75-77% | ~0.71 | 1.15 | Best balance |
| **PCA-50 (77% variance)** | 50 | ~73-75% | ~0.70 | 1.68 | Efficient |
| **PCA-20 (45% variance)** | 20 | ~70-72% | ~0.68 | 4.20 | Most compact |

**Key Insight**: PCA-50 achieves similar accuracy (~75%) with **68% fewer features** (50 vs 160)

### 5. **Logistic Regression Assumptions**

| Assumption | Original | VIF-Reduced | PCA |
|-----------|----------|-------------|-----|
| Binary Outcome | ✅ Pass | ✅ Pass | ✅ Pass |
| Independence | ✅ Pass | ✅ Pass | ✅ Pass |
| Sample Size (EPV≥10) | ❌ Fail (0.53) | ⚠️ Marginal | ✅ Pass (PCA-20) |
| No Multicollinearity | ❌ Fail | ✅ Pass | ✅ Pass |
| No Outliers | ✅ Pass | ✅ Pass | ✅ Pass |
| Linearity in Logit | ⚠️ Visual check | ⚠️ Visual check | N/A |

**Recommendation**: Use **PCA-20** or **VIF-reduced features** for logistically valid inference.

---

## 🔬 Methodology

### 1. **Data Cleaning & Feature Engineering**
- Custom pipeline (`data_cleaning.py`)
- Handling missing values (smart imputation)
- One-hot encoding for categorical variables
- TF-IDF vectorization for text responses
- Label encoding for ordinal variables

### 2. **Exploratory Data Analysis (EDA)**
- Distribution analysis
- Cross-tabulations
- Correlation analysis
- Visualization of key patterns

### 3. **Statistical Testing**
- **Chi-square tests** for categorical associations
- **Mann-Whitney U tests** for numeric features
- **Correlation analysis** with target variable

### 4. **Multicollinearity Analysis**
- **Variance Inflation Factor (VIF)** for all 160 features
- Identification of high-VIF features (VIF ≥ 10)
- Feature removal for better model stability

### 5. **Principal Component Analysis (PCA)**
- Dimensionality reduction
- Variance explained analysis
- Multiple configurations tested (20, 50, 73 components)

### 6. **Machine Learning Models**
- **Logistic Regression** (basic, regularized, VIF-reduced, PCA-based)
- **Random Forest** (basic, hyperparameter tuned)
- Grid search for optimal hyperparameters
- 5-fold cross-validation

### 7. **Model Evaluation**
- Test accuracy
- ROC AUC score
- Cross-validation accuracy
- Confusion matrices
- Feature importance analysis
- **Events Per Variable (EPV)** for sample size adequacy

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/2549486373/DSC190_final_project.git
cd DSC190_final_project
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
```

Or create a requirements.txt:
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook analysis.ipynb
```

---

## 💻 Usage

### Running the Analysis

1. **Open the notebook**: `analysis.ipynb`
2. **Run all cells** in order (Runtime > Run all)
3. **Review outputs** for each section

### Analysis Sections:

1. **Data Loading** - Load and inspect raw data
2. **Data Cleaning** - Automated preprocessing pipeline
3. **EDA** - Explore distributions and relationships
4. **Correlation Analysis** - Find feature relationships
5. **VIF Analysis** - Check multicollinearity (all 160 features)
6. **PCA** - Dimensionality reduction and visualization
7. **Hypothesis Testing** - Statistical significance tests
8. **Logistic Regression** - Baseline modeling
9. **Random Forest** - Advanced ensemble modeling
10. **Model Comparison** - Compare all approaches
11. **✨ NEW: Comprehensive Assumptions Check** - Systematic verification
12. **✨ NEW: VIF-Based Feature Selection** - Remove multicollinear features
13. **✨ NEW: VIF-Reduced Modeling** - Train on selected features
14. **✨ NEW: PCA-Based Modeling** - Multiple PCA configurations
15. **✨ NEW: Comprehensive Comparison** - 6-model comparison

### Custom Analysis

To modify the pipeline:
- **Adjust VIF threshold**: Change `vif_threshold` in Section 12
- **Try different PCA components**: Modify `pca_configs` in Section 14
- **Change train/test split**: Adjust `test_size` in split functions
- **Add new models**: Insert cells in modeling sections

---

## 📈 Results

### Performance Summary

**Best Overall Model**: Random Forest (Basic)
- Accuracy: **80.0%**
- ROC AUC: 0.771
- CV Accuracy: 73.99%

**Best Logistic Regression**: Tuned with L2 Regularization
- Accuracy: **77.5%**
- ROC AUC: 0.724
- CV Accuracy: 72.72%

**Most Efficient Model**: PCA-50
- Accuracy: ~75%
- Features: 50 (vs. 160 original)
- **68% feature reduction** with minimal accuracy loss
- EPV: 1.68 (better than original 0.53)

### Confusion Matrix (Best Model - Random Forest)
```
                Predicted
              Off  On
Actual  Off  [ 11   6 ]
        On   [  2  21 ]

Precision: 78% (On-Campus), 85% (Off-Campus)
Recall: 91% (On-Campus), 65% (Off-Campus)
```

### Feature Importance (Top 10)
1. Transportation issues (no car access)
2. Package frequency (encoded)
3. General delivered items
4. Store preference (encoded)
5. TF-IDF: specific goods
6. Items delivered: friends/family gifts
7. Clothing/shoes delivered
8. Personal care items delivered
9. TF-IDF: time-specific patterns
10. In-store preferences

---

## 🔧 Technical Details

### Algorithms Used
- **Logistic Regression** with L2 regularization
- **Random Forest Classifier** with grid search
- **Principal Component Analysis (PCA)**
- **Variance Inflation Factor (VIF)** calculation

### Key Metrics
- **Accuracy**: Overall correct predictions
- **ROC AUC**: Area under ROC curve (class separation)
- **Cross-Validation**: 5-fold CV for robustness
- **EPV**: Events Per Variable (sample size adequacy)
- **VIF**: Variance Inflation Factor (multicollinearity)

### Statistical Tests
- **Chi-square test**: Categorical associations
- **Mann-Whitney U test**: Non-parametric comparisons
- **Cook's Distance**: Outlier detection
- **Correlation analysis**: Feature relationships

### Data Processing
- **One-hot encoding**: Binary features for categories
- **TF-IDF vectorization**: Text feature extraction
- **Label encoding**: Ordinal variable encoding
- **StandardScaler**: Feature scaling for PCA

---

## 📊 Visualizations Included

- Distribution plots (target variable)
- Correlation heatmaps
- VIF distribution histograms
- PCA variance plots (scree plot, cumulative)
- 2D/3D PCA visualizations
- ROC curves
- Confusion matrices
- Feature importance plots
- Cook's Distance plots
- Comprehensive 6-panel comparison dashboard

---

## 🎓 Academic Context

**Course**: DSC 190 - Data Science Capstone
**Institution**: University of California, San Diego (UCSD)
**Focus**: Applied machine learning and statistical analysis

### Learning Outcomes Demonstrated:
✅ Data cleaning and preprocessing
✅ Feature engineering
✅ Statistical hypothesis testing
✅ Assumption checking for statistical models
✅ Machine learning model development
✅ Model evaluation and comparison
✅ Dimensionality reduction techniques
✅ Multicollinearity detection and resolution

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **DSC 190 Team** - UCSD Data Science Capstone

---

## 🙏 Acknowledgments

- UCSD survey respondents for providing data
- DSC 190 instructors and TAs
- scikit-learn and statsmodels communities
- Open-source data science tools

---

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Create an issue](https://github.com/2549486373/DSC190_final_project/issues)
- Repository: [DSC190_final_project](https://github.com/2549486373/DSC190_final_project)

---

## 📚 References

### Key Libraries Used:
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [statsmodels](https://www.statsmodels.org/) - Statistical modeling
- [SciPy](https://scipy.org/) - Scientific computing

### Methodology References:
- Variance Inflation Factor (VIF) for multicollinearity detection
- Events Per Variable (EPV) rule for logistic regression sample size
- Box-Tidwell test for linearity in logit
- Cook's Distance for influential point detection

---

## 🔄 Updates & Changelog

See [ANALYSIS_UPDATES.md](ANALYSIS_UPDATES.md) for detailed updates including:
- Comprehensive assumption checking (Section 11)
- VIF-based feature selection (Section 12-13)
- PCA-based modeling (Section 14)
- Comprehensive model comparison (Section 15)

---

## 📊 Quick Start Example

```python
# Load the cleaned data
import pandas as pd
df = pd.read_csv('data_cleaned.csv')

# Prepare features
X = df[feature_cols]  # 160 features
y = df['housing_binary']  # Target (0=Off-Campus, 1=On-Campus)

# Train a model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

---

**⭐ If you find this project useful, please consider giving it a star!**

---

*Last Updated: December 2024*
