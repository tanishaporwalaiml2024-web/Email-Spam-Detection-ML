# 💳 Credit Card Fraud Detection using Machine Learning

A machine learning project for detecting fraudulent credit card
transactions using **Random Forest Classification** and **SMOTE
(Synthetic Minority Oversampling Technique)** to address severe class
imbalance.

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where the
model must distinguish between:

-   **Class 0:** Legitimate transaction
-   **Class 1:** Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions
representing only a small fraction of all transactions. This project
compares a baseline Random Forest model with a Random Forest model
trained after applying SMOTE to the training data.

## 🎯 Objectives

-   Explore and understand the transaction dataset.
-   Check data quality and missing values.
-   Separate features and target variable.
-   Split the data into training and testing sets.
-   Train a Random Forest classifier.
-   Handle class imbalance using SMOTE.
-   Evaluate the model using fraud-focused metrics.
-   Visualize performance using a confusion matrix, ROC curve, and
    Precision-Recall curve.

## 📊 Dataset

The dataset used in this project contains:

  Property                      Value
  ------------------------- ---------
  Total transactions          284,807
  Total features/columns           31
  Legitimate transactions     284,315
  Fraudulent transactions         492
  Missing values                    0

### Features

The dataset contains:

-   `Time`
-   `V1` to `V28`
-   `Amount`
-   `Class` --- target variable

`Class = 0` represents a legitimate transaction and `Class = 1`
represents fraud.

> **Note:** The notebook uses the local file `dataset/creditcard.csv`.
> The dataset is included in this repository using **Git LFS** because
> of its large file size.

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Imbalanced-learn
-   Jupyter Notebook
-   Git & Git LFS

## 🔄 Machine Learning Workflow

``` text
Load Dataset
     ↓
Data Exploration
     ↓
Check Missing Values
     ↓
Separate Features & Target
     ↓
Train-Test Split
     ↓
Baseline Random Forest
     ↓
Apply SMOTE on Training Data
     ↓
Train Random Forest
     ↓
Make Predictions
     ↓
Evaluate Model
     ↓
Confusion Matrix / ROC Curve / PR Curve
```

## ⚖️ Handling Class Imbalance with SMOTE

The original training data contains a very small number of fraud cases
compared with legitimate transactions.

SMOTE is applied **only to the training data** so that the test set
remains representative of unseen data.

### Class distribution

**Before SMOTE**

``` text
Class 0: 227,451
Class 1:     394
```

**After SMOTE**

``` text
Class 0: 227,451
Class 1: 227,451
```

This creates a balanced training dataset and helps the classifier learn
the minority fraud class more effectively.

## 🤖 Model

The main classifier used is:

**Random Forest Classifier**

Final model configuration:

``` python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Random Forest combines multiple decision trees to make a more robust
classification decision.

## 📈 Results

The final SMOTE-based Random Forest model achieved the following results
on the test set:

  Metric         Score
  ----------- --------
  Accuracy      99.95%
  Precision     87.37%
  Recall        84.69%
  F1-Score      86.01%
  ROC-AUC       98.49%

### Why these metrics matter

**Accuracy:** Overall percentage of correctly classified transactions.

**Precision:** Of the transactions predicted as fraud, how many were
actually fraudulent.

**Recall:** Of all actual fraud cases, how many were successfully
detected.

**F1-Score:** Harmonic mean of precision and recall, useful when both
false positives and false negatives matter.

**ROC-AUC:** Measures how well the model separates fraudulent and
legitimate transactions across classification thresholds.

For fraud detection, **precision, recall, F1-score, and ROC-AUC are more
informative than accuracy alone**, because the dataset is highly
imbalanced.

## 🔍 Baseline Model

Before applying SMOTE, a Random Forest model with 50 trees was
evaluated.

### Baseline results

-   Accuracy: **99.95%**
-   Confusion Matrix:

``` text
[[56862     2]
 [   24    74]]
```

The SMOTE-based model was then trained with a larger Random Forest
configuration and evaluated using additional fraud-focused metrics.

## 📉 Visualizations

The notebook includes:

-   Confusion Matrix
-   ROC Curve
-   Precision-Recall Curve

These visualizations help analyze the model beyond simple accuracy.

## 📁 Project Structure

``` text
Credit-Card-Fraud-Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
├── requirements.txt
└── .gitattributes
```

## 🚀 How to Run the Project

### 1. Clone the repository

``` bash
git clone https://github.com/tanishaporwalaiml2024-web/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Open the notebook

``` bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

Or open the notebook directly in **VS Code** or **JupyterLab**.

### 4. Run the cells

Make sure the dataset is available at:

``` text
dataset/creditcard.csv
```

## 💡 Key Learning Outcomes

Through this project, I practiced:

-   Exploratory Data Analysis
-   Binary Classification
-   Train-Test Splitting
-   Random Forest
-   Handling Imbalanced Data
-   SMOTE
-   Confusion Matrix
-   Precision, Recall and F1-Score
-   ROC-AUC
-   Precision-Recall Curve
-   Model evaluation and interpretation
-   Git and Git LFS for large datasets

## ⚠️ Important Note

This project is intended for **educational and portfolio purposes**. The
evaluation results are based on the provided dataset and train-test
split and should not be interpreted as evidence of production readiness.

## 👩‍💻 Author

**Tanisha Porwal**

B.Tech Artificial Intelligence & Machine Learning

------------------------------------------------------------------------

⭐ If you find this project useful, feel free to explore the notebook
and the machine learning workflow.
