# Import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from config import TARGET_COLUMN

def split_data(data: pd.DataFrame) -> tuple:
    """
    Splits the data into predictors and target data

    Args:
        data (pd.DataFrame): The dataset to be split.

    Raises:
        Exception: If an error occurs.

    Returns:
        tuple: A tuple containing the train and test sets.
    """
    try:
        # Split data into predictors(X) and target data(Y)
        X = data.drop(columns=[TARGET_COLUMN])
        Y = data[TARGET_COLUMN]

        return X, Y
    except Exception as e:
        print(f"Failed to split data : {e}")

def split_train_test_data(X: pd.DataFrame, Y: pd.DataFrame) -> tuple:
    """
    Splits the data into train and test sets

    Args:
        X (pd.DataFrame): The predictor data.
        Y (pd.DataFrame): The target data.

    Raises:
        Exception: If an error occurs.

    Returns:
        tuple: A tuple containing the train 
        and test sets of the predictors and target.
    """
    try:
        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
            )

        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        print(f"Failed to split data : {e}")