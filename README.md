# Mushroom Classification Project

## Overview

This project focuses on the classification of mushrooms into poisonous or edible categories based on various features. The dataset used (`mushrooms 2.csv`) contains categorical and numerical attributes of mushrooms.

## Project Structure

- **Data Preprocessing**
  - **Missing Data Handling**:
    - Missing values in categorical features were imputed using the most frequent value.
  - **Categorical Data Encoding**:
    - Label encoding was applied to transform categorical features into numerical values.
  - **Normalization**:
    - Numerical features were normalized using Min-Max scaling to a range of [0, 1].

- **Exploratory Data Analysis (EDA)**
  - **Visualization**:
    - **Bar Plots**: Displayed the count of poisonous and edible mushrooms for each categorical feature.
    - **Pie Charts**: Illustrated the distribution of mushroom types across categorical features.

- **Feature Engineering**
  - **Correlation Analysis**:
    - **Kendall and Pearson Correlation**: Investigated correlations between numerical features to identify relationships and redundancies.
  - **Feature Selection**:
    - Dropped the 'veil-type' feature due to its constant value, which did not contribute to classification.

- **Model Building**
  - Implemented the following machine learning models:
    - Logistic Regression
    - Gaussian Naive Bayes
    - K-Nearest Neighbors
    - Decision Tree

- **Model Evaluation**
  - **Performance Metrics**:
    - Evaluated models using accuracy score and confusion matrix to assess classification performance.

## Conclusion

This project demonstrates the process of exploring, preparing, and modeling mushroom data for classification tasks. It provides insights into feature importance, model performance, and steps involved in building a predictive model.


