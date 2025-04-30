# Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data: pd.DataFrame, column_name: str):
    """
    Plots a histogram of the specified column.

    Args:
        data (pd.DataFrame): The dataset to be plotted.
        column_name (str): The name of the column to be plotted.

    Raises: 
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Plot histogram
        plt.figure(figsize=(5, 3))
        sns.histplot(data[column_name], kde=True)
        plt.title(f"Distribution of {column_name}")

        # Calculate mean and median for the column
        col_mean = data[column_name].mean()
        col_median = data[column_name].median()

        # Add vertical lines for mean and median
        plt.axvline(col_mean, color='red', linestyle='--', label="Mean")
        plt.axvline(col_median, color='green', linestyle='-', label="Median")

        # Add a legend on the plot
        plt.legend()

        # Save the plot
        plt.savefig(f"../outputs/plots/{column_name}_histogram.png")

        # Display the plot
        plt.show()
    except Exception as e:
        print(f"Failed to plot histogram : {e}")

def plot_boxplot(data: pd.DataFrame, column_name: str):
    """
    Plots a boxplot of the specified column.

    Args:
        data (pd.DataFrame): The dataset to be plotted.
        column_name (str): The name of the column to be plotted.

    Raises: 
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Plot boxplot
        plt.figure(figsize=(5, 3))
        sns.boxplot(y=data[column_name])
        plt.title(f"Distribution of {column_name}")
        plt.ylabel(column_name)

        # Save the plot
        plt.savefig(f"../outputs/plots/{column_name}_boxplot.png")

        # Display the plot
        plt.show()
    except Exception as e:
        print(f"Failed to plot a boxplot : {e}")

def plot_heatmap(data: pd.DataFrame, columns: list):
    """
    Plots a heatmap of the specified columns.

    Args:
        data (pd.DataFrame): The dataset to be plotted.
        columns (list): The names of the columns to be plotted.

    Raises: 
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Plot heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(data[columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")

        # Save the plot
        plt.savefig("../outputs/plots/correlation_heatmap.png")

        # Display the plot
        plt.show()
    except Exception as e:
        print(f"Failed to plot a heatmap : {e}")

def plot_countplot(data: pd.DataFrame):
    """
    Plots a countplot of the specified column.

    Args:
        data (pd.DataFrame): The dataset to be plotted.

    Raises: 
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Create a list of categorical (object) columns
        object_cols = data.select_dtypes(include=['object']).columns.tolist()
        object_cols = ["SeniorCitizen"] + object_cols

        # Loop through each object column
        for column in object_cols:
            # Plot countplot
            plt.figure(figsize=(10, 3))
            sns.countplot(x=data[column])
            plt.title(f"Count Plot of {column}")

            # Save the plot
            plt.savefig(f"../outputs/plots/{column}_countplot.png")

            # Display the plot
            plt.show()
        
    except Exception as e:
        print(f"Failed to plot a Count Plot : {e}")