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

Target Variable: is_default

<img width="860" height="354" alt="Univariate" src="https://github.com/user-attachments/assets/55fe6d10-7f36-40e9-9427-583bf645b9a5" />

**Interpretation**

-The dataset exhibited moderate class Imbalance so our Evaluation matrix included

        ◦ Precision & Recall (especially for the minority "Default"              class)
        ◦ F1-Score (balance between precision and recall)
        ◦ ROC-AUC (overall discrimination ability)
        ◦ Confusion Matrix (visualize trade-offs)

**Recommendation**

-Data splitting-We used stratified sampling (train/test/validation) to maintain the ∼80:20 class ratio.

Model Selection-Algorithms that handle model imbalance well:
        ◦ Random Forest (class weighting)
        ◦ XGBoost (scale_pos_weight parameter)
        ◦ Logistic Regression (class_weight='balanced')


3.3.2 **Bivariate analysis**

This was used to assess relationships between features and loan default risk.

<img width="878" height="394" alt="Bivariate" src="https://github.com/user-attachments/assets/5616195c-2c2d-4cee-8a1e-050866568939" />

**Findings**

1. Default Rate by Quarter (Seasonal Pattern)
         • Q1: 19.23%
         • Q2: 20.17%
         • Q3: 20.73% (peak)
         • Q4: 18.85% (lowest)
This implies that Q3 carries the highest risk, possibly due to seasonal economic factors or lending cycles.

2. Default Rate by Gender (Minimal Difference)
    • Gender 1.0: 19.51% default
    • Gender 2.0: 19.88% default
    • The difference is negligible (<0.4 percentage points), indicating gender is not a strong differentiator of default risk.
    
3. Default Rate by Loan Type (Significant Difference)
    • New Loans: 25.29% default rate
    • Existing Loans: 17.99% default rate
    • Difference: 7.3 percentage points
    
New loans are significantly riskier, which aligns with typical lending patterns even in the industry where new borrowers have unproven creditworthiness.

4. Top 10 Products by Default Rate
    • Product 30149 has 100% default rate but only 2 loans (small            sample).
    • Other high-risk products have default rates ranging from 40% to         83%.
    • These high-risk products represent specific segments that may          need product-level risk review.

**Payment Behaviour Analysis**

<img width="870" height="502" alt="PBA" src="https://github.com/user-attachments/assets/d4a29aaf-aec7-4e3d-b96b-00e9b02ed155" />

**Interpretations and Recommendations**

1. Default is Preceded by Clear Delinquency Patterns
Defaulters show classic delinquency patterns:
    • Chronic lateness (avg 222 days late)
    • Maximum lateness extending over a year
    • Penalty accumulation
    • Reduced payment amounts and frequency
2. Actionable Early Warning Signals
Monitor these thresholds for early intervention:
    • avg_days_late > 30 days
    • max_days_late > 90 days
    • Any penalties incurred
    • num_ontime_payments below 3
    • avg_installment_paid dropping >20%
Recommendations:
    1. Key features for default prediction models are
        max_days_late, avg_days_late, num_ontime_payments, penalty_rate
        
    2. Create a composite "delinquency score" combining:
        ◦ Days late metrics
        ◦ Penalty indicators
        ◦ Payment frequency
        
    3. Segment customers by payment behavior:
        ◦ "At-risk": High days late + penalties
        ◦ "Watchlist": Reduced payment amounts/frequency
        ◦ "Low-risk": Consistent on-time payments, no penalties

**Loan Characteristics Analysis**

Interpretation and Recommendation

1. Loan Amount vs. Default
         ◦ Small loan amounts have more defaulters
         ◦ Large loan amounts have less defaulters
   implying that small loans carry disproportionate risk.

2. Interest Rate vs. Default-both very high and very low interest rates are associated with default

3. Loan Term Days vs. Default

      Defaulters tend to have shorter loan terms compared to non-            defaulters this is because they require larger periodic                payments, less time for financial recovery and may divert the          purpose

 **Customer Demographics Analysis**

   1. Age

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












