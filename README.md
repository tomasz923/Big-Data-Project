## Income Classification based on US Census Data

This repository contains the final project for the "Big Data" course at Warsaw School of Economics. The project was developed collaboratively by me and two other colleagues. It was developed via Databricks.

Here is the link to the notebook: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1370453633106322/95111070522763/7710213420369909/latest.html

### Dataset

The project utilizes census information from the UCI Machine Learning Repository. The dataset consists of individual records and includes various demographic variables. The main objective of this study is to classify whether an individual earns more or less than $50,000 per year based on these demographic attributes.

Link to the data: https://archive.ics.uci.edu/ml/datasets/adult

### Project Objective

The primary goal of this project is to accurately predict the income level of workers in the United States by analyzing demographic variables. The project encompasses several stages, including data preprocessing, exploratory data analysis, and the development of machine learning models for income classification.

### Methodology

The project follows a well-defined methodology to achieve the desired objective. Here is an overview of the main steps involved:

1. **Data Preprocessing and Exploration:** The dataset is thoroughly cleaned and prepared for analysis. Missing values are handled, and appropriate transformations are applied to ensure data quality. Exploratory data analysis techniques are employed to gain insights into the distribution and relationships among the features and target variables.

2. **Descriptive Analysis:** A comprehensive descriptive analysis is conducted to understand the characteristics and patterns present in the dataset. This analysis provides valuable insights into the factors that may influence an individual's income level.

3. **Machine Learning Models:** Two machine learning models, namely logistic regression and classification tree, are utilized for the income level classification task. These models are trained on the preprocessed dataset, which includes predictors such as age, work class, years of education, marital status, occupation, relationship status, race, sex, capital gain, capital loss, hours per week worked, and native country.

4. **Model Evaluation:** The performance of the developed models is evaluated using two key metrics: Area Under ROC (AUC) and accuracy. The AUC metric is selected as the primary performance measure for model evaluation. Additionally, a thorough Cross-Validation approach is employed for hyperparameter tuning to enhance the model's predictive capabilities.

### Repository Structure

The repository is structured as follows:

- **Data:** This directory contains the raw and preprocessed datasets used in the project.
- **Notebook:** This directory includes Jupyter notebook that outline the step-by-step process of the project, from data preprocessing to model development and evaluation. Yet, it is recommended to use the link provided at the top. It includes the evaluation metrics, visualizations, and reports generated during the project.


### Conclusion

This project presents an in-depth analysis of census data to classify the income level of workers in the United States. By implementing effective data preprocessing techniques and leveraging machine learning models, we aim to accurately predict whether an individual earns more or less than $50,000 per year based on their demographic attributes.

For further details, please refer to the Databricks notebook and files in this repository.

If you have any questions or feedback, please feel free to reach out.
