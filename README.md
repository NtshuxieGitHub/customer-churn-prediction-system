# Customer Churn Modelling and Prediction System

## 📊 Project Overview

This project involves developing a **Customer Churn Prediction System** using machine learning. The aim is to model and predict customer churn behavior based on various features from the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## 🧰 Tools and Libraries

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – data manipulation
  - `sklearn` – ML modelling, preprocessing, and evaluation
  - `imblearn` – imbalance handling
  - `xgboost` – gradient boosting classifier
  - `seaborn`, `matplotlib` – visualizations
  - `pickle` – saving encoders and models

---

## 🧪 Workflow Steps

1. **Data Exploration & Understanding**  
   Load and examine raw data for completeness and distribution.
2. **Data Cleaning**  
   Handle missing values, standardize formats.

3. **EDA (Exploratory Data Analysis)**  
   Analyze trends and patterns; generate and save plots.

4. **Data Processing**

   - Encode categorical variables
   - Save encoders using `.pkl`
   - Handle class imbalance
   - Split features (X) and target (y)
   - Create train-test split

5. **Model Training (Baseline)**  
   Train models with default parameters:

   - Decision Tree
   - Random Forest
   - XGBoost

6. **Hyperparameter Tuning**

   - Use `GridSearchCV` to optimize models
   - Evaluate performance with cross-validation based on accuracy

7. **Model Saving and Evaluation**

   - Fit the best model on training data
   - Save the trained model with `.pkl`
   - Evaluate on test data

8. **Automation & Deployment**
   - Implement the model training system (`modelling_system.py`)
   - Implement the prediction interface (`predictive_system.py`)
   - Test the system with fresh unseen data

---

## 📁 Folder Structure

![Folder Structure](/folder_structure.jpg)

---

## 📤 Outputs

- **Plots**: Saved in `outputs/plots/`
- **Models**: Trained models saved as `.pkl` in `outputs/models/`
- **Logs**: Output logs from training and prediction steps

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```
2. Install dependencies:

   pip install -r requirements.txt

3. Configure settings:

   Edit configurations in config/config.py and update config/\_init\_\_.py

4. Run the model training system:

   07_modelling_testing.ipynb

5.Run the predictive system:

    08_prediction_system.ipynb
