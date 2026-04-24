# Comment Category Classification Pipeline

A comprehensive machine learning pipeline for multi-class comment classification with model ensemble and threshold optimization.

## 📋 Overview

This notebook solves a **comment category prediction challenge** using advanced feature engineering, multiple machine learning models, and sophisticated ensemble techniques. The pipeline classifies comments into 4 categories with robust data cleaning, text processing, and model blending.

## 🎯 Challenge

**Dataset**: Comment Category Prediction Challenge (Kaggle)  
**Task**: Multi-class classification (4 classes)  
**Target Variable**: `label` (0, 1, 2, 3)

### Data Statistics
- **Train Set**: ~3,500+ cleaned samples
- **Test Set**: ~1,500+ samples
- **Class Distribution**: Imbalanced (handled via class weights)
- **Features**: Text + Tabular metadata

## 🔧 Features

### Text Features
- **Base**: Comment field (cleaned and tokenized)
- **TF-IDF**: 5,000 n-grams (1-2) with preprocessing
- **Linguistic**: Word count, character count, punctuation (!, ?), capitalization
- **Tokens**: URL, email, mention, hashtag detection

### Tabular Features
**Basic Numeric**:
- `emoticon_1`, `emoticon_2`, `emoticon_3`
- `upvote`, `downvote`, `if_1`, `if_2`

**Engineered Numeric**:
- Engagement ratio: `(upvote + 1) / (downvote + 1)`
- Net votes: `upvote - downvote`
- Total votes: `upvote + downvote`
- Temporal: Hour of day, day of week, is_weekend flag

**Identity Features**:
- Categorical: race, religion, gender, disability
- Flags: Presence indicators for each identity category
- Count: Total identity attributes mentioned

## 📊 Pipeline Architecture

### 1. Data Cleaning & Validation
- Remove outliers (upvote/downvote 99th percentile for training)
- Clip extreme values (95th percentile)
- Filter comments with <5 characters
- Normalize identity columns and datetime handling
- Remove timezone info from dates

### 2. Feature Engineering
```
Text Pipeline:
  - Pattern cleaning (URLs, emails, mentions, hashtags, numbers)
  - Lowercase conversion
  - Whitespace normalization
  - TF-IDF vectorization (5k features)

Tabular Pipeline:
  - Numeric: Robust scaling with median imputation
  - Categorical: One-hot encoding with 'missing' fill
  - Combined: ~40+ total features
```

### 3. Model Training

**Primary Models** (Full preprocessing with TF-IDF):
- **LightGBM** (baseline + tuned)
  - n_estimators: 700
  - max_depth: 6
  - Hyperparameter tuning via RandomizedSearchCV
  
- **XGBoost** (baseline + tuned)
  - n_estimators: 200
  - max_depth: 7
  - Hyperparameter tuning via RandomizedSearchCV

**Secondary Models** (Shared preprocessing):
- **LinearSVC** - Calibrated with sigmoid
- **LogisticRegression** - Multi-class, balanced class weights
- **AdaBoost** - Lightweight ensemble on tabular features only

### 4. Cross-Validation Strategy
- **Method**: 3-fold Stratified K-Fold
- **Approach**: Out-of-Fold (OOF) predictions for blending
- **Metrics**: Macro F1 score (primary), accuracy

### 5. Ensemble & Optimization
```python
Blending Formula:
  ensemble_proba = (
    0.50 * lgb_proba +
    0.25 * lr_proba +
    0.15 * xgb_proba +
    0.05 * svc_proba +
    0.05 * ada_proba
  )

Per-Class Threshold Optimization:
  - Learned thresholds for each class independently
  - Applied on blended probability matrix
  - Optimized on validation set via grid search
```

## 📈 Model Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM (baseline) | ~0.64 | ~0.58 |
| XGBoost (baseline) | ~0.63 | ~0.56 |
| LightGBM (tuned) | ~0.65 | ~0.60 |
| XGBoost (tuned) | ~0.64 | ~0.59 |
| LinearSVC | ~0.60 | ~0.52 |
| LogisticRegression | ~0.62 | ~0.55 |
| AdaBoost | ~0.58 | ~0.50 |
| **5-way Ensemble** | **~0.67** | **~0.62** |

*Note: Actual performance varies based on CV folds and data splits*

## 🚀 Usage

### Requirements
```
numpy
pandas
scikit-learn (0.24+)
lightgbm
xgboost
scipy
matplotlib
seaborn
```

### Execution Flow

1. **Load & Explore** - Import data, check shapes and distributions
2. **Clean Data** - Remove outliers, normalize columns, fix dates
3. **Feature Engineering** - Create text and tabular features
4. **Preprocess** - Build TF-IDF and scaling pipelines
5. **Hyperparameter Tuning** - RandomizedSearchCV for LightGBM & XGBoost
6. **Model Training** - Train all 5 models with cross-validation
7. **Threshold Optimization** - Learn per-class probability thresholds
8. **Ensemble Blending** - Combine predictions with optimal weights
9. **Final Prediction** - Fit on full train, predict test set
10. **Submit** - Generate submission.csv

### Key Parameters
```python
RANDOM_STATE = 42
CV_SPLITS = 3
LGB_ESTIMATORS = 700
XGB_ESTIMATORS = 200
ADA_ESTIMATORS = 60
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
```

## 📁 Input Files

From `/kaggle/input/comment-category-prediction-challenge/`:
- `train.csv` - Training data with labels
- `test.csv` - Test data for predictions
- `Sample.csv` - Sample submission format

## 📤 Output

**submission.csv**
```
ID,label
1,2
2,0
...
```

## 💡 Key Techniques

✅ **Data Quality**: Outlier removal, value clipping, imputation  
✅ **Feature Engineering**: Linguistic, temporal, engagement, identity features  
✅ **Text Processing**: TF-IDF with n-grams, pattern tokenization  
✅ **Class Imbalance**: Weighted class priors and sample weighting  
✅ **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameters  
✅ **Cross-Validation**: Stratified OOF for unbiased ensemble training  
✅ **Threshold Optimization**: Per-class probability thresholds  
✅ **Model Blending**: Weighted average of 5 diverse models  
✅ **Memory Efficiency**: Sparse matrix handling, garbage collection  

## 🎓 Learning Insights

- **Ensemble Power**: 5-way blend outperforms individual models by ~3-5% F1
- **Class Weights**: Critical for imbalanced data (class 3 gets 2.2x weight)
- **Threshold Tuning**: Simple but effective post-processing → +1-2% gain
- **Diversity Matters**: LGB + XGB + SVC + LR + Ada cover different model families
- **Text + Tabular**: Combined feature space better than text-only approach

## 📝 Notes

- Notebook uses Kaggle environment (`/kaggle/input/`, `/kaggle/working/`)
- Supports GPU acceleration for LightGBM/XGBoost (configurable)
- Memory efficient: Handles large datasets with sparse matrices
- Results are reproducible with `RANDOM_STATE = 42`
- All models fitted on **full training data** for final submission

## 🔄 Reproducibility

```python
np.random.seed(42)
pd.set_option('display.max_columns', 60)
RANDOM_STATE = 42
```

All randomness is controlled via the RANDOM_STATE parameter.

---

**Author**: Competition Solution  
**Date**: 2026  
**Challenge**: Kaggle Comment Category Prediction Challenge
