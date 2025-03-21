# Taxis_DataAnalysis_MLClassificationModels

## This notebook analyzes a taxi dataset by performing data cleaning, preprocessing, visualization, and machine learning modeling.

*************************************
1️⃣ **Data Cleaning & Preprocessing**

✅ Handles missing values using SimpleImputer.

✅ Detects and removes duplicate values.

✅ Encodes categorical features using LabelEncoder.

✅ Applies TF-IDF & SVD on text-based features (pickup_zone, dropoff_zone).

✅ Uses Power and Quantile transformations to normalize numerical features.
************************************************
2️⃣ **Data Visualization**

✅ Heatmaps to check missing values.

✅ Box plots to understand numerical feature distributions.

✅ Donut & Bar charts for categorical data insights.
****************************************************
3️⃣ **Machine Learning Models**

✅ Compares LightGBM, CatBoost, and RandomForest classifiers.

✅ Uses Cross-Validation with ROC AUC as an evaluation metric.

✅ Prints model performance results to identify the best classifier for predicting payment type in the taxi dataset.
*********************************************
4️⃣ **Key Libraries Used**

- pandas, numpy → Data processing.
  
- matplotlib, seaborn, plotly → Data visualization.
  
- SimpleImputer, LabelEncoder → Data preprocessing.
  
- TF-IDF, SVD → Text feature transformation.
  
- PowerTransformer, QuantileTransformer → Feature scaling.
- 
- LightGBM, CatBoost, RandomForest → Machine learning models.
********************************************************
🔹 **Summary**
✔ Cleans and preprocesses the taxi dataset.
✔ Visualizes key insights with different plots.
✔ Compares ML models to predict payment type.

**********************************************************
NOTE:
**CatBoost performs best likely because it handles categorical features and complex relationships efficiently, reducing overfitting.**

**LightGBM performs competitively but might need more careful hyperparameter tuning to reach the highest performance.**

**RandomForest shows the lowest ROC AUC among the three, possibly due to its more basic averaging approach and inability to capture intricate feature interactions as effectively as the boosting algorithms.**

**Each model's performance can also be influenced by the specifics of your dataset (e.g., noise level, feature distribution, presence of outliers), so fine-tuning and feature engineering play significant roles.**
