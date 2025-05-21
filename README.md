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
   - Hybrid resampling (SMOTEENN, Hard-Negative Mining)  
   - Cost-sensitive algorithms and class-weight tuning  
   - Threshold optimization via Precision–Recall curves
   - Two Model Cascade (coarse filter + fine-tuning classifier)
   - Ensemble and Boosting strategies (Balanced Random Forest, XGBoost)
   - One Class Models (SVM, Isoletion Forest)  

## Key Highlights

- **Dramatic Income Improvements & Persistent Coverage Gap**  
  Cash transfers more than double mean income (from ~COP 1 M to ~COP 2.7 M) and quadruple median income (from COP 700 k to COP 3 M), compressing the distribution and eliminating extreme low-income outliers. Yet only 8.6 % of individuals receive a subsidy, highlighting a need to broaden coverage.

- **Top Predictive Features for Targeting**  
  The most informative GEIH variables are combined housing cost, household size, declared monthly income, labor‐market formality (contract type, pension contributions) and educational attainment. Embedding these dimensions into eligibility rules markedly improves identification of vulnerable households.

- **Cascading ML Pipeline for Balanced Performance**  
  A two-stage approach—first a high-recall XGBoost filter, then a precision-focused Random Forest with tuned probability thresholds—meets recall targets (≥ 70 %) while containing false positives, demonstrating that pipeline design outweighs any single algorithm.

- **Data Scarcity Limits Supervised Precision**  
  Even with SMOTEENN, Borderline-SMOTE, class-weighted losses, polynomial/PCA feature engineering and exhaustive hyperparameter tuning, supervised cascades plateau at ~20–25 % precision on the subsidy class, constrained by the rarity and overlap of positive examples.

- **Anomaly Detection Excels on High Precision**  
  Framing subsidy recipients as anomalies (Isolation Forest, One-Class SVM) achieves > 90 % precision—at the expense of lower recall—offering a complementary strategy when high confidence in positive predictions is paramount.

- **Modular Architecture Enables Policy-Driven Trade-Offs**  
  Each stage (filter, refiner, anomaly detector) can be independently adjusted to favor recall or precision, allowing policymakers to recalibrate the system according to changing objectives or resource constraints.


## Prerequisites

The following Python packages (and their dependencies) are required:

```bash
pip install pandas numpy matplotlib sklearn imblearn xgboost random
```

## Data Description

The project relies in on folder and one file:

- **`data/CSV.rar/`**  
  Contains the original May 2024 GEIH survey tables in CSV format:  
  - `generales.csv`  
  - `laborales.csv`  
  - `hogar.csv`  
  - `subsidios.csv`  
  - `fuerza_trabajo.csv`  
  - `desempleados.csv`  

- **`data/Base_Modelo.csv/`**  
  Contains the consolidated and preprocessed dataset used in the **Advanced Modeling** notebook (Script 2). This file merges, cleans and transforms the key variables needed for training and evaluating imbalance-aware classification models. 

## How to Run

Simply open the notebooks in order—first the **Exploratory Analysis** (`subsidy analysis.ipynb`), then the **Advanced Modeling** (`Full Maching Learning Modeling.ipynb`)—and execute each cell sequentially. All code chunks are fully documented for ease of follow-through

---

Building on these insights can guide policymakers in refining eligibility criteria, expanding coverage and designing multidimensional safety nets that reach everyone who needs them most.  

## Contributing

Contributions are welcome! Please open issues or submit pull requests at  
https://github.com/pablo-reyes8

## License

This project is licensed under the Apache License 2.0.  
