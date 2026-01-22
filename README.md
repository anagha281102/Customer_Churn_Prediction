Customer Churn Prediction Project

This project aims to predict customer churn for a telecommunications company using machine learning. 
By identifying customers likely to cancel their service, businesses can take proactive measures to improve retention.

Project Overview
The project follows a complete machine learning workflow:
* Data Cleaning: Handling data type conversions for TotalCharges and addressing missing values.
* Exploratory Data Analysis (EDA): Visualizing relationships between features like contract type, tenure, and churn.
* Preprocessing: Using Label Encoding for categorical variables.
* Handling Imbalance: Using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
* Modeling: Training and evaluating different classification models.
* Serialization: Saving the trained model and encoders using Pickle for future use.
  
Tech Stack
* Language: Python
* Libraries: Pandas, NumPy, Matplotlib, Seaborn
* Machine Learning: Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* Model Saving: Pickle
  
Dataset
The dataset used is the Telco Customer Churn dataset. It includes:
* Customer Demographics: Gender, Senior Citizen status, Partners, and Dependents.
* Service Information: Phone service, Multiple lines, Internet service (DSL/Fiber optic), and Security/Support add-ons.
* Account Information: Tenure, Contract type (Month-to-month, One year, Two year), Payment method, Monthly Charges, and Total Charges.
  
Models Implemented
I evaluated the following models to find the best fit for this classification task:
1. Decision Tree Classifier
2. Random Forest Classifier
3. XGBoost Classifier
  
How to Use
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed (pandas, sklearn, xgboost, imblearn).
3. Open the Jupyter Notebook Customer_Churn_Prediction.ipynb.
4. Run the cells to see the analysis, model training, and results.
   
Saving and Loading Models
To ensure the project is production-ready, I used the Pickle library to save the trained model and the encoders. This allows anyone to load the "trained" state of the project without needing to re-run the entire training process.

Example of loading the model:

import pickle
# Load the saved model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

    
Results
The RandomForest model, combined with SMOTE for data balancing, provided the most reliable predictions for identifying potential churn customers.
