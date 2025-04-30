# Import dependencies
import pandas as pd
from config import DATA_TYPE_CHANGE_COLUMN

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the customer dataset by removing unnecessary 
    columns, getting rid of null values, converting data 
    types to the desired types, removing duplicate data, etc.

    Args: 
        data (pd.DataFrame): The dataset to be cleaned.

    Raises:
        Exception: If an error occurs

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    try:
        # Drop the customerID column as it isn't required for modelling
        data = data.drop(columns=['customerID'])

        # Replace " " with "0" in the TotalCharges column
        data['TotalCharges'] = data['TotalCharges'].replace(' ', '0.0')

        # Convert the TotalCharges column to a float
        data[DATA_TYPE_CHANGE_COLUMN] = data[DATA_TYPE_CHANGE_COLUMN].astype(float)

        return data
    except Exception as e:
        print(f"Data cleaning failed : {e}")