# Import dependencies
import pickle as pkl

def load_model() -> tuple:
    """
    Load the trained model

    Args:
        None

    Raises:
        - Any errors encountered

    Returns:
        tuple: A tuple containing the model and feature columns
    """
    try:
        # Load the trained model from a pickle file
        with open('../outputs/models/trained_customer_churn_mdl.pkl', 'rb') as f:
            model_data = pkl.load(f)

        # Extract the model and feature columns
        model = model_data['model']
        feature_columns = model_data['feature_columns']

        return model, feature_columns
    except Exception as e:
        print(f"Failed to load model data : {e}")