# Import dependencies
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle as pkl
import joblib
import os


def default_param_training(X_train: pd.DataFrame, Y_train: pd.DataFrame) -> dict:
    """
    Train models using default parameters

    Args:
        X_train (pd.DataFrame): The training data.
        Y_train (pd.DataFrame): The target data.

    Raises:
        - Any errors encountered

    Returns:
        dict: A dictionary containing the cross validation results.
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

        return cross_val_scores
    except Exception as e:
        print(f"Failed to train models using default parameters : {e}")

def tuning_and_best_model_selection(X_train: pd.DataFrame, Y_train: pd.DataFrame) -> tuple:
    """
    Perform hyperparameter tuning to improve the scores of the models, s
    ave the models, then select the best model

    Args:
        X_train (pd.DataFrame): The training data.
        Y_train (pd.DataFrame): The target data.

    Raises:
        - Any errors encountered

    Returns:
        object: The best model.
    """
    try:
        # Define models and their parameter grids
        param_grids = {
            "Decision Tree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                    "min_child_weight": [0.01, 0.1]
                }
            }
        }

        # Initialize best score, model, model name and grid search
        results = {}
        best_score = 0
        best_model = None
        best_model_name = ""

        # Loop through each model
        for model_name, model_grid in param_grids.items():
            # Let user know that model is being trained using hyperparameter tuning
            print(f"Training {model_name} using hyperparameter tuning...")

            # Perform grid search with 5-fold cross validation
            grid_search = GridSearchCV(model_grid["model"], model_grid["params"], cv=5, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train, Y_train)

            # Store the cross validation results
            results[model_name] = {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_
            }

            # Update best score, model, model name and grid search if necessary
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_model_name = model_name

        # Save the best model
        with open(f"../outputs/models/{best_model_name.replace(' ', '_')}_best_model.pkl", "wb") as f:
            pkl.dump(best_model, f)
        
        # Let user know what the best model is and its score
        print(f"The best model is {best_model_name} with a score of {best_score:.2f}")
        print("-"*50)

        return results, best_model
    except Exception as e:
        print(f"Failed to train models using hyperparameter tuning : {e}")