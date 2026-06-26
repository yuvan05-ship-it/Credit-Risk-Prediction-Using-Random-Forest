# Credit Risk Prediction Using Random Forest

A machine learning project that predicts whether a bank customer
will default on a loan within the next 2 years, using real-world
financial data from 150,000 US bank customers.

## 📌 Problem Statement
Banks lose crores every year from loan defaults.
This project builds a classification model to identify
high-risk borrowers before approving loans — helping banks
make smarter lending decisions.

## 📊 Dataset
- Source: Give Me Some Credit — Kaggle Competition
- Rows: 150,000 customers
- Features: 10 financial attributes
- Target: SeriousDlqin2yrs (1 = defaulted, 0 = did not)

| Column | Description |
|--------|-------------|
| RevolvingUtilizationOfUnsecuredLines | Credit card usage ratio |
| age | Borrower age |
| NumberOfTimes90DaysLate | Times payment was 90+ days late |
| DebtRatio | Monthly debt ÷ monthly income |
| MonthlyIncome | Monthly income |
| NumberOfOpenCreditLinesAndLoans | Total active loans |
| NumberOfDependents | Family members supported |

## 🔍 Challenges Solved
| Challenge | Solution |
|-----------|----------|
| Class imbalance (93% vs 7%) | SMOTE oversampling |
| Missing values (20% in MonthlyIncome) | Median imputation |
| Outliers (age=0, DebtRatio=3000) | Percentile capping |

## 📈 Results
| Metric | Score |
|--------|-------|
| Accuracy | 0.84+ |
| Precision | 0.72+ |
| Recall | 0.76+ |
| F1 Score | 0.74+ |
| AUC | 0.85+ |

## 🔑 Key Learnings
- High accuracy on imbalanced data is misleading
- AUC and Recall matter more than Accuracy for default prediction
- SMOTE improved minority class detection significantly
- NumberOfTimes90DaysLate and RevolvingUtilization
  are the strongest default predictors

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## 📁 Project Structure
├── credit_risk_prediction.ipynb   # main notebook
├── cs-training.csv                # dataset
└── README.md

## 🚀 How to Run
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook credit_risk_prediction.ipynb

## 📬 Connect
- GitHub: github.com/yuvan05-ship-it
- LinkedIn: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/yuvanraj05/)
