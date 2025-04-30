# Import dependencies
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from config import TARGET_COLUMN

def encode_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs label encoding.

    Args:
        data (pd.DataFrame): The dataset to be processed.

    Raises:
        Exception: If an error occurs.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    try:
        # Encode the target column
        data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"No": 0, "Yes": 1})

        # Create a list of categorical (object) columns
        object_cols = data.select_dtypes(include=['object']).columns.tolist()

        if os.path.exists("../outputs/models/encoders.pkl"):
            # If the encoders.pkl file exists, load the encoders
            with open("../outputs/models/encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
            
            # Loop through each object column and encode
            for column in object_cols:
                if column in encoders:
                    # If the column is in the encoder, encode object column
                    data[column] = encoders[column].transform(data[column])
                else:
                    raise ValueError(f"Column {column} not found in encoders")
                
        else:
            # Initialize disctionary to save encoders
            encoders = {}

            # Loop through each object column
            for column in object_cols:
                # Encode object column
                label_encoder = LabelEncoder()
                data[column] = label_encoder.fit_transform(data[column])
                encoders[column] = label_encoder

            # Save the encoders
            with open("../outputs/models/encoders.pkl", "wb") as f:
                pickle.dump(encoders, f)

        return data
    except Exception as e:
        print(f"Failed to process data : {e}")