# 🏠 Real Estate Price Prediction

https://realestatepriceprediction-beige.vercel.app/

An end-to-end Machine Learning project that predicts California house prices using advanced regression models.  
The project includes complete EDA, preprocessing, model comparison, hyperparameter tuning, and deployment using Flask API.

---

## 📌 Project Overview

This project uses the California Housing dataset to predict the median house value based on various housing and demographic features.

The final model selected was **HistGradientBoostingRegressor**, which outperformed other models after cross-validation and hyperparameter tuning.

---

## 📊 Dataset Information

- Total Records: 20,640
- Features: 9 input features
- Target Variable: `median_house_value`
- Categorical Feature: `ocean_proximity`
- Missing Values: Present in `total_bedrooms` (handled using median imputation)

### Features Used:

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity

---

## 🔎 Exploratory Data Analysis (EDA)

✔ Checked missing values  
✔ Checked duplicates  
✔ Distribution plots  
✔ Boxplots for outlier detection  
✔ Correlation heatmap  
✔ Residual analysis  

### Key Insights:

- `median_income` has the strongest correlation with house price (0.68)
- Some price values are capped at 500001
- Tree-based models perform significantly better than linear models

---

## 🤖 Model Comparison

| Model | CV RMSE | CV R2 |
|-------|---------|-------|
| Linear Regression | 68,604 | 0.648 |
| Ridge | 68,595 | 0.648 |
| Lasso | 68,603 | 0.648 |
| Random Forest | 49,458 | 0.817 |
| **HistGradientBoosting** | **48,094** | **0.827** |

---

## 🎯 Final Model (After Hyperparameter Tuning)

**Model:** HistGradientBoostingRegressor  

### Best Parameters:
- learning_rate: 0.1  
- max_leaf_nodes: 63  
- min_samples_leaf: 20  
- l2_regularization: 0.1  

### Final Performance:

**Train Performance**
- RMSE: 35,917
- R² Score: 0.903

**Test Performance**
- RMSE: 46,667
- R² Score: 0.834

The model shows good generalization with no severe overfitting.

---

## 🛠 Preprocessing Pipeline

The project uses a Scikit-Learn Pipeline with:

- Median Imputation (Numerical)
- Standard Scaling
- Most Frequent Imputation (Categorical)
- OneHot Encoding
- HistGradientBoosting Regressor

Everything is wrapped inside a single pipeline for production readiness.

