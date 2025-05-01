# Import dependencies
import pandas as pd
import pickle as pkl
from src.load_model import load_model
from config import DATA_PATH
import sys
import os

def run_predictive_system() -> str:
    """
    Runs the predictive system.

    Args:
        None

    Raises:
        - Any errors encountered

    Returns:
        - prediction (str): The prediction of the model
    """
    # Get the directory where this function's file lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the encoders.pkl relative to this script
    encoder_path = os.path.join(current_dir, "outputs", "models", "encoders.pkl")

    try:
        # Load the trained model and feature columns
        mdl, features = load_model()

        # Load data
        df = pd.read_csv(DATA_PATH)

        # Encode data
        with open(encoder_path, "rb") as f:
            encoders = pkl.load(f)

        for column in features:
            if column in encoders:
                df[column] = encoders[column].transform(df[column])

        # Make a prediction
        prediction = mdl.predict(df)

        if prediction == 0:
            p = "Customer will not churn."
        else:
            p = "Customer will churn."

        # Get the probability of the prediction
        proba = mdl.predict_proba(df)

        return p, proba
    except Exception as e:
        print(f"Failed to run predictive system : {e}")