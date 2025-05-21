# Colombia Subsidy ML Prediction (GEIH)

A machine learning project leveraging Colombia’s GEIH household survey to build and evaluate predictive models for identifying subsidy-eligible households. By combining thorough exploratory analysis with advanced imbalance-handling techniques, we aim to optimize resource allocation and help reduce socioeconomic inequality.

## Notebooks

1. **Exploratory Analysis**  
   A detailed walkthrough of data cleaning, descriptive statistics and visualizations. Here we:
   - Quantify subsidy coverage and its distribution across income, social class, geography and demographic groups  
   - Contrast pre- and post-subsidy income distributions to measure impact on inequality  
   - Identify which GEIH variables (household size, income, education, labor formality, etc.) most strongly correlate with subsidy receipt  

2. **Advanced Modeling**  
   In this notebook, we explore and implement a variety of intermediate-to-advanced machine learning techniques to tackle a highly imbalanced binary classification problem. Our primary goal is to boost the model’s ability to correctly identify instances of the minority class (Class 1) without sacrificing overall performance. Key methods include:  
   - Hybrid resampling (SMOTEENN, Borderline-SMOTE, ADASYN)  
   - Cost-sensitive algorithms and class-weight tuning  
   - Threshold optimization via Precision–Recall curves  
   - Ensemble strategies (Balanced Random Forest, EasyEnsemble, stacking)  

## Key Highlights

- **Impact Measurement:** Cash transfers more than double mean income and nearly quadruple median income for beneficiary households, drastically narrowing class gaps.  
- **Feature Importance:** Combined housing cost, household size, declared income, employment formality and education emerge as the top predictors.  
- **Targeting Gaps:** While education, contract type and geography are well accounted for, infrastructure deficits and ethnic vulnerability remain under-weighted in current subsidy rules.  
- **Modeling Challenges:** Severe class imbalance (only ~8 % recipients) makes it hard for standard learners to detect beneficiaries—advanced imbalance-handling is essential to raise precision without losing recall.

---

Building on these insights can guide policymakers in refining eligibility criteria, expanding coverage and designing multidimensional safety nets that reach everyone who needs them most.  

