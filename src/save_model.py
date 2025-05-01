# import dependencies
import pandas as pd
import pickle as pkl

def save_trained_mdl(mdl: object, X: pd.DataFrame) -> None:
    """
    Save the trained model

    Args:
        mdl (object): The model.
        X (pd.DataFrame): The feature columns.

    Raises:
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Save the trained model as a pickle file
        mdl_data = {"model": mdl, "feature_columns": X.columns.tolist()}

        with open('../outputs/models/trained_customer_churn_mdl.pkl', 'wb') as f:
            pkl.dump(mdl_data, f)

    except Exception as e:
        print(f"Failed to save model data : {e}")