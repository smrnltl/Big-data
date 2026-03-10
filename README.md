# Bank Customer Churn Analysis

**Module:** ST7082CEM : Big Data Management and Data Visualisation
**Student:** Smaran Luitel | **Student ID:** 250087

## Overview

A complete data science pipeline for analyzing bank customer churn using **Apache Spark (PySpark)**. The project covers data preprocessing, exploratory analysis, supervised learning (classification), unsupervised learning (clustering), and regression modeling on a 10,000-record customer dataset.

## Project Structure

```
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data loading, EDA, feature engineering
│   ├── 02_classification.ipynb     # Logistic Regression & Random Forest
│   ├── 03_clustering.ipynb         # K-Means customer segmentation
│   └── 04_regression.ipynb         # Linear Regression on CreditScore
├── dataset/
│   └── Customer-Churn-Records.csv  # Source data (10,000 records × 18 features)
├── exports/                        # Generated CSVs for Tableau dashboards
└── Bigdata-customers.twb           # Tableau workbook
```

## Dataset

**Source:** Bank Customer Churn Records (10,000 customers)

| Feature | Description |
|---|---|
| CreditScore | Credit bureau score (350–850) |
| Geography | Country (France, Spain, Germany) |
| Gender | Male / Female |
| Age | Customer age (18–92) |
| Tenure | Years as customer (0–10) |
| Balance | Account balance (€) |
| NumOfProducts | Products owned (1–4) |
| HasCrCard | Credit card holder (0/1) |
| IsActiveMember | Active member (0/1) |
| EstimatedSalary | Annual salary estimate |
| Exited | **Target** : Churned (1) or Retained (0) |
| SatisfactionScore | Satisfaction rating (1–5) |
| CardType | Card tier (Diamond, Gold, Silver, Platinum) |
| PointEarned | Loyalty points earned |

**Note:** The `Complain` column was excluded from all models due to target leakage (r = 0.996 with `Exited`).

## Methodology & Results

### 1. Preprocessing & EDA

- No missing values or duplicates in the dataset
- **Churn rate:** 20.38% (class imbalance ratio 3.91:1)
- Key churn drivers identified: NumOfProducts (3–4 products to 80–100% churn), Age 50–59 (56% churn), Germany (32.4% churn)
- Feature engineering: one-hot encoding for categorical variables, standard scaling for numerical features

### 2. Classification : Churn Prediction

| Model | AUC-ROC | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.765 | 72.11% | 0.744 |
| **Random Forest** | **0.863** | **86.88%** | **0.853** |

- Random Forest outperforms across all metrics
- Top features: Age (29.6%), NumOfProducts (20.9%), Balance (7.5%)
- Class weighting applied to handle imbalance in Logistic Regression

### 3. Clustering : Customer Segmentation (K-Means, k=4)

| Cluster | Label | Size | Churn Rate |
|---|---|---|---|
| 0 | Satisfied Savers | 21.1% | 23.6% |
| 1 | Multi-Product Loyalists | 31.0% | **12.4%** (lowest) |
| 2 | High-Value Dissatisfied | 25.3% | **24.6%** (highest) |
| 3 | Loyal High-Earners | 22.6% | 23.5% |

- Optimal k selected via elbow method and silhouette analysis
- Cluster 2 (high balance, low satisfaction) identified as priority for retention efforts

### 4. Regression : CreditScore Prediction

| Metric | Value |
|---|---|
| RMSE | 97.15 |
| MAE | 78.59 |
| R² | ≈ 0.00 |

- Near-zero R² is expected : credit scores depend on external bureau data not available in this dataset
- Demonstrates regression methodology and honest model evaluation

## Technology Stack

- **Apache Spark 4.1.1** (PySpark) : distributed data processing & ML pipelines
- **Python 3.14** : Spark driver
- **Jupyter Notebooks** : interactive analysis
- **Tableau Desktop** : interactive dashboards and visualization
- **Pandas** : data export

## How to Run

1. **Prerequisites:** Apache Spark, Python with PySpark, Jupyter Notebook
2. **Windows setup:** Set `HADOOP_HOME` to a directory containing `winutils.exe`
3. **Run notebooks in order:**
   ```
   01_preprocessing.ipynb to 02_classification.ipynb to 03_clustering.ipynb to 04_regression.ipynb
   ```
4. **Tableau dashboards:** Open `Bigdata-customers.twb`

## Key Findings

- **Random Forest is the recommended model** for churn prediction (AUC 0.863)
- **Age and NumOfProducts** are the strongest churn predictors
- **Germany-based customers** have nearly double the churn rate of France/Spain
- **Four actionable customer segments** enable targeted retention strategies
- **Multi-product customers** (Cluster 1) show significantly lower churn : supports cross-selling as a retention strategy
