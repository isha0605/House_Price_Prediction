<b>House Price Prediction - AI/ML Project<b>

This project involves building a machine learning model to predict house prices based on various features. The dataset is sourced from the Kaggle competition "House Prices - Advanced Regression Techniques". The goal is to develop a highly accurate predictive model using advanced regression techniques.

📌 Kaggle Competition
Dataset: House Prices - Advanced Regression Techniques

Model Performance: 87.16% (R-squared score)
📌 Libraries Used
Python Libraries:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

XGBoost

📝 Data Analysis and Preprocessing
Loading Data:

The dataset is loaded from CSV files.

Exploratory Data Analysis (EDA):

Visualizations (histograms, box plots, heatmaps) help identify patterns and missing values.

Preprocessing Steps:

Handling missing values (imputation/dropping columns).

Encoding categorical variables using one-hot encoding.

Standardizing numerical features for better model performance.

🤖 Model Training & Selection
Evaluated multiple regression models:

Linear Regression, SVR, SGDRegressor, KNeighborsRegressor

DecisionTreeRegressor, RandomForestRegressor

GradientBoostingRegressor, XGBRegressor, MLPRegressor

GradientBoostingRegressor was selected based on superior performance.

Cross-validation was used for model evaluation.

🎯 Model Evaluation & Prediction
The trained model is used to predict house prices on the test dataset.

Predictions are saved in submission.csv.

The model is stored as gbr.pkl for future use.

🚀 Future Enhancements
Hyperparameter tuning for better model accuracy.

Feature selection techniques to improve model performance.

Deployment of the model using Flask or FastAPI.
