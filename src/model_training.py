# Import dependencies
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def default_param_training(X_train: pd.DataFrame, Y_train: pd.DataFrame):
    """
    Train models using default parameters

    Args:
        X_train (pd.DataFrame): The training data.
        Y_train (pd.DataFrame): The target data.

    Raises:
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Create a dictionary of models with a random state of 42
        mdls = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42)
        }

        # Initialize a dictionary to store cross validation results
        cross_val_scores = {} 

        # Loop through each model
        for mdl_name, mdl in mdls.items():
            # Perform 5-fold cross validation for the model:

            # Let user know model is being trained
            print(f"Training {mdl_name} using default parameters...")

            # Train the model using 5-fold cross validation
            scores = cross_val_score(mdl, X_train, Y_train, cv=5, scoring="accuracy")

            # Store the cross validation results
            cross_val_scores[mdl_name] = scores

            # Print the cross validation accuracy score
            print(f"{mdl_name} cross-validation accuracy score: {np.mean(scores):.2f}")
            print("-"*50)
    except Exception as e:
        print(f"Failed to train models using default parameters : {e}")