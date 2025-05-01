# Import dependencies
import pandas as pd
from config import NUMERICAL_COLUMNS, DATA_TYPE_CHANGE_COLUMN, DATA_PATH

def collect_data() -> tuple:
    """
    Load the customer churn dataset into a 
    pandas dataframe and explore its shape, 
    structure and data types

    Raise:
        - Any errors encountered

    Returns:
        - data (pd.DataFrame): The dataset
        - first_five (pd.DataFrame): The first 5 rows of the dataset
        - data_shape (tuple): The shape of the dataset
        - data_info (object): The info of the dataset
        - data_summary (object): The summary statistics of the dataset
        - unique_values_dict (dict): The unique values in each column
        - filtered_data (pd.DataFrame): The filtered dataset
        - height_filtered_data (int): The height of the filtered dataset
    """
    try:
        # Load data
        data = pd.read_csv(DATA_PATH)

        # View the first 5 rows of the data and all columns
        pd.set_option('display.max_columns', None)
        first_five = data.head()
        
        # Get the shape of the dataset
        data_shape = data.shape

        # Get data info: missing data, data types, etc.
        data_info = data.info(verbose=True)

        # Get summary statistics of the dataset
        data_summary = data.describe()

        # Initialize empty dictionary
        unique_values_dict = {}

        # Get the unique values in each column
        for column in data.columns:
            if column not in NUMERICAL_COLUMNS:
                # If the column is in the list

                # get unique values
                unique_values = data[column].unique()

                # Append unique_values_dict
                unique_values_dict[column] = unique_values

        # Filter dataframe using missing TotalCharges values " "
        missing_total_charges = data[data[DATA_TYPE_CHANGE_COLUMN] == " "]
        height_missing_total_charges = missing_total_charges.shape[0]

        # Cehck class distribution of target column (Churn)
        churn_distribution = data['Churn'].value_counts()

        return data, first_five, data_shape, data_info, data_summary, unique_values_dict, missing_total_charges, height_missing_total_charges, churn_distribution

    except Exception as e:
        print(f"Data collection and understanding failed : {e}")
