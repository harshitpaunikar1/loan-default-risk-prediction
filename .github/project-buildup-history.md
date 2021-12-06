# Project Buildup History: Loan Default Risk Prediction

- Repository: `loan-default-risk-prediction`
- Category: `data_science`
- Subtype: `prediction`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2021-09-06 - Day 3: Data cleaning pass

- Task summary: Returned to Loan Default Risk Prediction after a couple weeks away. The dataset had some columns with very high missing rates that I had flagged earlier but not acted on. Today decided which to impute and which to drop — anything above 40 percent missing was dropped, the rest got median or mode fill depending on type. Also found that one of the binary target columns had a few rows encoded as strings instead of integers which was silently coercing the whole column to object dtype. Fixed that and reran the dtype checks.
- Deliverable: Missing value strategy applied, dtype issue in target column resolved.
## 2021-09-13 - Day 4: EDA and correlation pass

- Task summary: Deep EDA session on the loan dataset today. Went through each feature group systematically — loan amount distribution, term breakdown, employment length spread, and home ownership categories. The most interesting finding was that annual income had a huge number of outliers that were making the histograms unreadable. Capped at the 99th percentile for visualization purposes and noted it for the feature prep stage. Also looked at default rate by loan grade which showed a very clean monotonic relationship — that feature is probably going to be one of the most predictive.
- Deliverable: EDA complete with annotations. Loan grade likely top predictor.
## 2021-09-13 - Day 4: EDA and correlation pass

- Task summary: Follow-up in the afternoon: the correlation heatmap was not rendering properly for the full feature set because too many columns were still in string format. Fixed the type casting block and regenerated it.
- Deliverable: Heatmap rendering properly. No major surprises in the correlation structure.
## 2021-09-13 - Day 4: EDA and correlation pass

- Task summary: Late session: added a class balance check for the target variable. About 20 percent default rate which is workable without aggressive resampling. Noted it in the project doc.
- Deliverable: Class distribution documented. Mild imbalance, manageable.
## 2021-11-01 - Day 5: Feature engineering

- Task summary: Back to the loan default project for a feature engineering session. The raw dataset has a date column for loan issuance that had not been used at all yet. Extracted year, month, and a quarter indicator, then also computed how many months ago the loan was issued relative to the dataset snapshot date. That temporal distance feature turned out to correlate decently with default probability, which makes sense since recent loans have less repayment history. Also looked at interaction between loan amount and annual income as a debt-to-income proxy.
- Deliverable: Temporal features added. DTI proxy computed. Feature set expanded from 14 to 18.
## 2021-12-06 - Day 6: Model training

- Task summary: Trained the first proper model on the loan default dataset today. Started with logistic regression as the baseline and got a reasonable AUC. The recall on the default class was still low so tried increasing the class weight parameter. Also ran a quick Random Forest to see how far the tree-based approach could push the AUC without much tuning. The gap between the two was meaningful — about 0.07 AUC — which justified moving to the tree-based approach for the main model.
- Deliverable: Random Forest meaningfully outperforms logistic baseline. Continuing with RF as primary model.
