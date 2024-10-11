# Mental Health Disorder Analysis

This project explores global trends in mental health disorders from 1990 to 2017 using data analysis and machine learning models, such as Linear Regression and K-Nearest Neighbors (KNN). The analysis aims to predict future trends in mental health disorders such as depression, anxiety, and schizophrenia using data obtained from [Kaggle](https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder).

## Project Structure
```
mental_health_disorder_analysis/
│
├── CIT694_文献综述_Essay.pdf          # Graduate research essay
├── Mental_health_Depression_disorder_Data.csv  # Dataset on global mental health trends
├── multivariate_linear_regression.py   # Multivariate Linear Regression code
├── knn_regression_year_vs_depression.py  # KNN regression model to predict depression
└── README.md                          # This README file
```

Dataset
- The dataset used in this project was sourced from Kaggle, and it contains information on mental health disorders such as depression, anxiety, bipolar disorder, and more. The data is grouped by year, region, age group, and disorder type, including the prevalence rate and associated healthcare costs.
- Link to dataset: Kaggle Mental Health Disorder Dataset

Installation and Setup
- To run this project, you will need to install the following Python packages: pip install pandas numpy scikit-learn matplotlib seaborn

Usage
- The project includes several Python scripts for different analysis approaches:
```
multivariate_linear_regression.py: Uses multivariate linear regression to predict mental health disorder percentages.
knn_regression_year_vs_depression.py: Implements KNN regression to model depression disorder trends over time.
```
Results
- The project showcases visualizations, including correlation matrices and trend forecasts, to highlight the impact of various mental health disorders on different populations.
