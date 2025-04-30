# Import dependencies
import sys
import os

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import functions
from src.data_collection import collect_data
from src.data_cleaning import clean_data
from src.eda import plot_histogram, plot_boxplot, plot_heatmap, plot_countplot

# Import config
from config import NUMERICAL_COLUMNS

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

