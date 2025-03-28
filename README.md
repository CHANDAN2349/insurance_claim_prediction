Thought for 2 seconds### Data-Driven Claim Severity Prediction for Insurance Risk Management

## Problem Statement

An insurance company aims to improve its underwriting process by better predicting the severity of claims. Using historical data including claim amounts, policyholder demographics, vehicle details, and previous claim history, the goal is to develop a predictive model that can accurately estimate claim costs to support pricing decisions and risk management.

## Approach

1. **Data Preprocessing**: Handled missing values, special characters, and standardized features
2. **Exploratory Data Analysis**: Analyzed target variable distribution, claim components, and relationships between features
3. **Feature Engineering**: Created new features like vehicle age, customer tenure, incident season, and claim ratios
4. **Model Building**: Trained and compared multiple regression models (Linear, Ridge, Random Forest, Gradient Boosting, XGBoost)
5. **Model Fine-Tuning**: Optimized hyperparameters to reduce overfitting while maintaining predictive power
6. **Interactive Prediction**: Developed a user-friendly interface for real-time predictions


## Results

- Gradient Boosting emerged as the best model with a testing R² of 0.7333
- Successfully reduced overfitting from 0.0858 to 0.052 (difference between training and testing R²)
- Identified key predictive features including incident severity, vehicle age, and policy details
- Deployed an interactive Streamlit application with policy lookup and prediction capabilities


## Streamlit Application Features

The deployed application includes:

1. **Home Page**: Introduction and data upload functionality
2. **Data Exploration**: Visualizations and statistical analysis
3. **Model Training & Evaluation**: Model selection, training, and comparison
4. **Prediction Dashboard**: Policy lookup and manual entry options for claim prediction


## How to Run

1. Install requirements:


```plaintext
pip install -r requirements.txt
```

2. Run the Streamlit app:


```plaintext
streamlit run app.py
```

## Author

**Chandan N**

College: Christ University

Email: [chandan.n@msds.christuniversity.in](mailto:chandan.n@msds.christuniversity.in)

LinkedIn: [https://www.linkedin.com/in/chandan2349/](https://www.linkedin.com/in/chandan2349/)

GitHub: [https://github.com/CHANDAN2349](https://github.com/CHANDAN2349)

## Future Work

- Incorporate additional external data sources
- Implement more advanced ensemble techniques
- Develop a real-time monitoring system for model performance
- Extend the application to include fraud detection capabilities
