# Early Detection of Loan Default Risk for SMEs & Youth Entrepreneurs
1. **Project Overview**
   
Small and Medium Enterprises (SMEs) and youth entrepreneurs in emerging markets often face limited access to affordable credit due to high perceived default risk. Traditional credit assessment methods are manual, slow, subjective, and difficult to scale.
This project develops a supervised machine learning framework for early loan default risk detection, enabling lenders to:

    Reduce non-performing loans (NPLs)
    Identify high-risk borrowers early
    Enable proactive intervention strategies
    Promote financial inclusion through fairer, data-driven credit decisions

2.  **Data Understanding**
   
 2.1 **Data Source**
 
The dataset consists of loan-level data collected from Micromart Africa’s digital lending platform, focusing on SME and youth entrepreneur borrowers.

 2.2 **Data Overview**

Total records: 277,159 loans

Problem type: Binary classification

Target variable: is_default (1 = Default, 0 = Non-default)

2.3 **Feature Categories**

Borrower Attributes: Age, Gender, Account Age

Loan Characteristics: Principal Amount, Interest, Loan Term, Product Type, Number of Installments

Repayment Behavior: Total paid, Unpaid Balance, Days Late, Number of Late Payments, Number of on-time payments

The dataset combines demographic, transactional, and behavioral data, making it suitable for supervised credit risk modeling.


3. **Data Cleaning & Preprocessing**
   
3.1 **Missing Values**

Extreme cases with high levels of missing data were dropped to reduce noise and improve model reliability.
We dropped unique IDs used for tracking and do not provide predictive value for loan repayment behavior

3.2 **Duplicates**
Our data had no duplicated values


3.3 **Exploratory Data Analysis(EDA)**

3.3.1 **Univariate analysis**
This was conducted to understand feature distributions.
<img width="927" height="390" alt="image" src="https://github.com/user-attachments/assets/823948c0-ff5e-4bbf-a394-d2bb4e15b544" />


Bivariate analysis was used to assess relationships between features and loan default risk.

3.3 **Feature Engineering**

We identified the Top most important features using absolute coefficient values from the logistic regression baseline model.

Top Predictive Features

   Num_ontime_payments-More on-time payments reduce default risk   
   Num_late_payments-Higher late payments increase default risk   
   Late_payment_rate-Critical early warning signal   
   Has_payment_history-Existence of repayment history lowers uncertainty   
   Num_penalties-Indicates financial distress  

**Key Insights**

Payment behavior is the strongest predictor of loan default
Certain loan products (e.g., Product IDs 30150, 30154, 30111) exhibit higher risk
Seasonality effects were observed, with December loans showing elevated default risk

4. **Exploratory Data Analysis (EDA)**
Target Variable Distribution
      Non-default: 80.21%
      Default: 19.79%
   
**Interpretation**
The dataset shows moderate class imbalance (≈80:20), which is manageable without aggressive rebalancing.
A naïve model predicting all loans as non-default would achieve 80.21% accuracy, setting a strong baseline to outperform.

Modeling Implications

Focus on Recall, especially for the default class

Use Precision, F1-score, ROC-AUC, and Confusion Matrix for comprehensive evaluation

Stratified data splitting is recommended

5**Modeling Approach**

**Baseline Model**

Algorithm: Logistic Regression with ElasticNet Regularization
Solver: saga

**Best Hyperparameters**

C = 1

l1_ratio = 0.7

Penalty = ElasticNet

**Test Set Performance**

Recall: 92.53%
Precision: 97.64%
F1-score: 95.01%
Accuracy: 98.07%

**Business Interpretation**

The model correctly identifies over 92% of defaulters
Very low false alarm rate (~2.4%)
Minimal overfitting observed
Strong and reliable baseline for early risk detection

6. **Advanced Models**
   
Several advanced models were evaluated, and the best-performing model was LightGBM

**LightGBM was best because......**
   It gave the Highest Recall: 95.84%
   Detects 240 more defaulters per 10,000 loans compared to Random Forest
   Best F1-score: 97.25%
   Highest Accuracy: 98.92%
   Fast training & memory-efficient
   Handles complex non-linear patterns
   Built-in regularization reduces overfitting risk
   Native support for categorical features

Conclusion: LightGBM provides the best balance between risk detection, reliability, and scalability.












