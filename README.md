
# Enhancing Fraud Detection in Financial Transactions through Machine Learning Techniques

## Project Overview
The rise in financial fraud poses significant challenges for institutions and individuals alike. This project leverages advanced Machine Learning techniques to enhance fraud detection in financial transactions. By analyzing patterns and anomalies in transaction data, the project compares Decision Tree, Random Forest, and Neural Network models to determine their effectiveness in identifying fraudulent activities. The Random Forest model emerged as the most reliable, achieving an accuracy of 99.76% and a strong balance between precision (82.61%) and recall (73.20%).

## Key Features
- **Focus Area**: Fraud detection in financial transactions.
- **Dataset**: Kaggle's `fraudTrain.csv`, consisting of 1,048,575 records and 23 features, including transaction time, amount, location, and fraud status.
- **Best Performing Model**: Random Forest, balancing high accuracy and minimal false positives and false negatives.
- **Core Techniques**:
  - Data preprocessing: Handling missing values, removing duplicates, normalization, and feature engineering.
  - SMOTE: Synthetic Minority Over-sampling Technique to address class imbalance.
  - Performance Metrics: Precision, recall, F1-score, ROC-AUC.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [Models Implemented](#models-implemented)
6. [Results](#results)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Future Work](#future-work)
9. [Contributors](#contributors)
10. [License](#license)

---

## Project Structure
```
fraud-detection-ml/
│
├── data/
│   ├── fraudTrain.csv   # Dataset
│
├── notebooks/
│   ├── eda.ipynb        # Exploratory Data Analysis
│   ├── model_comparison.ipynb  # Model comparison and results
│
├── src/
│   ├── preprocess_data.py  # Data preprocessing scripts
│   ├── train_models.py     # Model training scripts
│   ├── evaluate_models.py  # Evaluation and metric calculation
│
├── results/
│   ├── confusion_matrix.png  # Confusion matrix images for models
│   ├── roc_auc_curves.png    # ROC-AUC curve comparison
│
├── requirements.txt      # Dependency list
├── README.md             # This file
└── LICENSE               # License file
```

---

## Data Preprocessing
Data preprocessing is critical for fraud detection. The following steps were taken:
1. **Missing Values**: Checked and confirmed no missing values in the dataset.
2. **Duplicates**: Removed duplicates to ensure data integrity.
3. **Data Transformation**: Converted date and time columns to datetime objects for temporal analysis.
4. **Feature Engineering**: Created new features such as `transaction hour`, `transaction day`, and `high-value transaction flags`.
5. **SMOTE**: Applied to balance the dataset by synthesizing minority class (fraudulent transactions).
6. **Normalization and Scaling**: Standardized numerical columns using StandardScaler to improve model performance.

---

## Models Implemented
The following machine learning models were evaluated:

### 1. Decision Tree
- **Accuracy**: 99.49%
- **Precision**: 54.17%
- **Recall**: 71.80%
- **F1-Score**: 61.75%
- **Strengths**: Intuitive and interpretable.
- **Weaknesses**: Tendency to overfit and higher false negatives.

### 2. Random Forest (Best Model)
- **Accuracy**: 99.76%
- **Precision**: 82.61%
- **Recall**: 73.20%
- **F1-Score**: 77.62%
- **Strengths**: Robust, handles class imbalance well, ensemble learning.
- **Weaknesses**: Computationally intensive with large datasets.

### 3. Neural Network
- **Accuracy**: 95.56%
- **Precision**: 10.92%
- **Recall**: 94.92%
- **F1-Score**: 19.58%
- **Strengths**: Can learn complex patterns.
- **Weaknesses**: Prone to overfitting, high false positives.

---

## Results
| Metric       | Decision Tree  | Random Forest  | Neural Network |
|--------------|----------------|----------------|----------------|
| **Accuracy** | 99.49%         | **99.76%**     | 95.56%         |
| **Precision**| 54.17%         | **82.61%**     | 10.92%         |
| **Recall**   | 71.80%         | **73.20%**     | 94.92%         |
| **F1-Score** | 61.75%         | **77.62%**     | 19.58%         |

The Random Forest model emerged as the most effective algorithm for this dataset, achieving a high balance between precision and recall.

---

## Exploratory Data Analysis
EDA uncovered several insights:
- **Transaction Patterns**: Most fraudulent transactions occurred late at night (3 a.m.).
- **Geographic Trends**: Fraudulent transactions were concentrated in specific regions, notably New York.
- **Category Analysis**: Fraud was highest in online shopping and general internet transactions.
- **Class Imbalance**: Only 0.573% of the dataset's transactions were fraudulent.

---

## Future Work
- Expand the dataset to cover more transaction types and institutions.
- Experiment with deep learning architectures like LSTMs or CNNs for time-series data.
- Build a real-time fraud detection system.
- Enhance explainability using SHAP or LIME for model interpretation.

---

## Contributors
- **Md Ubaidur Rahman**  
  MSc Data Science and Analytics
  University of Westminster
- **Supervisor: Yongchao Huang**
  Senior Research Associate
  University of Cambridge, 
  Lecturer in CS
  University of Westminster,
  Assistant Professor in Computing Science
  University of Aberdeen

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
