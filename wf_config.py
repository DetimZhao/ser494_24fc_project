# Contains contants/defaults and utility functions for logging and other common tasks in the workflow.

import sys
import os
import logging

# Set PYTHONPATH if not already set 

# Define the root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Add the project root directory to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# If this does not work, try the following if on UNIX:
# export PYTHONPATH=.

#### CONSTANTS/DEFAULTS ####

# CONSTANTS FOR FOLDER PATHS
# Project default folder structure
DATA_GEN_FOLDER = 'data_gen/'
DATA_ORIGINAL_FOLDER = 'data_original/'
DATA_PROCESSED_FOLDER = 'data_processed/'
VISUALIZATIONS_FOLDER = 'visuals/'
MODELS_FOLDER = 'models/'
EVALUATION_FOLDER = 'evaluation/'

# Additional folders 
VECTORIZERS_FOLDER = f'{DATA_PROCESSED_FOLDER}vectorizers/' # Folder for storing vectorizers (insde data_processed)
VECTORIZED_RESULTS_FOLDER = f'{DATA_PROCESSED_FOLDER}vectorized_results/' # Folder for storing vectorized results (inside data_processed)

# CONSTANTS FOR DATA FILES
STEAM_STORE_DATA = f'{DATA_ORIGINAL_FOLDER}steam-games.csv' # Original store data
STEAM_STORE_DATA_CLEANED = f'{DATA_PROCESSED_FOLDER}steam-store-data-cleaned.csv' # Cleaned store data

IDS_LIST_FILE = f'{DATA_GEN_FOLDER}idlist.txt' # File containing list of game IDs to fetch reviews for

STEAM_REVIEWS_DATA = f'{DATA_ORIGINAL_FOLDER}combined_reviews.csv' # Combined all Steam reviews data
STEAM_REVIEWS_DATA_CLEANED = f'{DATA_PROCESSED_FOLDER}steam-reviews-data-cleaned.csv' # Cleaned reviews data

STEAM_REVIEWS_DATA_BERT_EMBEDDINGS_NPY = f'{DATA_PROCESSED_FOLDER}bert_embeddings.npy' # BERT embeddings for reviews

COMBINED_CLUSTERING_STEAM_DATASET = f'{DATA_PROCESSED_FOLDER}combined-clustering-steam-dataset.csv' # Combined dataset for clustering

# Constants for ml training and testing datasets
FEATURES_SCALER_PICKLE = f'{DATA_PROCESSED_FOLDER}features_scaler.pkl' # File for scaler used in standardizing features
TRAIN_FEATURES_NPY = f'{DATA_PROCESSED_FOLDER}train_features.npy'
TEST_FEATURES_NPY = f'{DATA_PROCESSED_FOLDER}test_features.npy'
TRAIN_IDS_CSV = f'{DATA_PROCESSED_FOLDER}train_ids.csv'
TEST_IDS_CSV = f'{DATA_PROCESSED_FOLDER}test_ids.csv'
TRAIN_CLUSTERS_CSV = f'{DATA_PROCESSED_FOLDER}train_clusters.csv'
TEST_CLUSTERS_CSV = f'{DATA_PROCESSED_FOLDER}test_clusters.csv'


# CONSTANTS FOR OUTPUT FILES
CORRELATIONS_TEXT_FILE = f'{DATA_PROCESSED_FOLDER}correlations.txt'
STORE_DATA_SUMMARY_TEXT_FILE = f'{DATA_PROCESSED_FOLDER}store_data_summary.txt'
REVIEWS_DATA_SUMMARY_TEXT_FILE = f'{DATA_PROCESSED_FOLDER}reviews_data_summary.txt'
EVALUATION_SUMMARY_TEXT_FILE = 'summary.txt'

# CONSTANTS FOR VISUALIZATION FILES
CORRELATIONS_HEATMAP = f'{VISUALIZATIONS_FOLDER}heatmap_correlations.png'


#### LOGGING ####

# Define logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_section(title):
    '''
    Logs a section header or footer with a given title, enclosed in a pattern for readability.
    
    Args:
        title (str): The title of the section to display within the pattern.
    '''
    title = title.upper() # Convert title to uppercase for emphasis
    border = '#' * (len(title) + 10)
    print(f'\n{border}')
    print(f'#### {title} ####')
    print(f'{border}\n')

#### CREATE FOLDERS IF THEY DO NOT EXIST ####
directories = [
    DATA_GEN_FOLDER,
    DATA_ORIGINAL_FOLDER,
    DATA_PROCESSED_FOLDER,
    VISUALIZATIONS_FOLDER,
    MODELS_FOLDER,
    EVALUATION_FOLDER,
    VECTORIZERS_FOLDER,
    VECTORIZED_RESULTS_FOLDER
]
for directory in directories:
    # Create directory if it does not exist
    os.makedirs(directory, exist_ok=True)
