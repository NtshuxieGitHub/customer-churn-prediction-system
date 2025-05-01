# Import dependencies
import sys
import os

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import functions
from src.data_collection import collect_data
from src.data_cleaning import clean_data
from src.eda import plot_histogram, plot_boxplot, plot_heatmap, plot_countplot
from src.data_processing import encode_labels, handle_target_imbalance
from src.train_test_split import split_data, split_train_test_data
from src.model_training import default_param_training, tuning_and_best_model_selection
from src.model_evaluation import evaluate_model
from src.save_model import save_trained_mdl

# Import config
from config import NUMERICAL_COLUMNS

def run_modelling_system():
    """
    Runs the modelling system.

    Args:
        None

    Raises:
        - Any errors encountered

    Returns:
        None
    """
    try:
        # Run functions
        data, _, _, _, _, _, _, _, _ = collect_data()
        data_cleaned = clean_data(data)

        # Plot histograms and boxplots
        for col in NUMERICAL_COLUMNS:
            plot_histogram(data_cleaned, col)
            plot_boxplot(data_cleaned, col)

        # Plot heatmap
        plot_heatmap(data_cleaned, NUMERICAL_COLUMNS)

        # Plot count plots for object features
        plot_countplot(data_cleaned)

        # Process data: encode, split X, Y, train test, imbalance handling
        encoded_data = encode_labels(data_cleaned)
        X, Y = split_data(encoded_data)
        X_train, X_test, Y_train, Y_test = split_train_test_data(X, Y)
        X_train, Y_train = handle_target_imbalance(X_train, Y_train)

        # Model training, tuning and best model selection
        _ = default_param_training(X_train, Y_train)
        _, best_model = tuning_and_best_model_selection(X_train, Y_train)
        best_model = best_model.fit(X_train, Y_train)

        # Model evaluation
        _, _, _, _ = evaluate_model(X_test, Y_test, best_model)

        # Save trained model
        save_trained_mdl(best_model, X)
    except Exception as e:
        print(f"Failed to run predictive system : {e}")