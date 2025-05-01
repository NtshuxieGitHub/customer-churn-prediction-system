# Import dependencies
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(X_test: pd.DataFrame, Y_test: pd.DataFrame, mdl: object) -> tuple:
    """
    Evaluate the model

    Args:
        X_test (pd.DataFrame): The test data.
        Y_test (pd.DataFrame): The target data.
        mdl (object): The model.

    Raises:
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Evaluate the model on the test data
        Y_test_pred = mdl.predict(X_test)

        # Calculate accuracy score
        accuracy = accuracy_score(Y_test, Y_test_pred)

        # Calculate confusion matrix
        cm = confusion_matrix(Y_test, Y_test_pred)

        # Calculate classification report
        cr = classification_report(Y_test, Y_test_pred)

        # Print outputs
        print("Accuracy Score:\n", accuracy)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cr)

        return Y_test_pred, accuracy, cm, cr
        
    except Exception as e:
        print(f"Failed to evaluate model : {e}")