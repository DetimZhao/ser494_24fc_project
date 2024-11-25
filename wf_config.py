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
DATA_GEN_FOLDER = 'data_gen/'
DATA_ORIGINAL_FOLDER = 'data_original/'
DATA_PROCESSED_FOLDER = 'data_processed/'
VISUALIZATIONS_FOLDER = 'visuals/'


# CONSTANTS FOR DATA FILES
STEAM_STORE_DATA = f'{DATA_ORIGINAL_FOLDER}steam-games.csv'
STEAM_STORE_DATA_CLEANED = f'{DATA_PROCESSED_FOLDER}steam-store-data-cleaned.csv'
IDS_LIST_FILE = f'{DATA_GEN_FOLDER}idlist.txt'
STEAM_REVIEWS_DATA = f'{DATA_ORIGINAL_FOLDER}combined_reviews.csv'
STEAM_REVIEWS_DATA_CLEANED = f'{DATA_PROCESSED_FOLDER}steam-reviews-data-cleaned.csv'


# CONSTANTS FOR OUTPUT FILES
CORRELATIONS_TEXT_FILE = f'{DATA_PROCESSED_FOLDER}correlations.txt'
STORE_DATA_SUMMARY_TEXT_FILE = f'{DATA_PROCESSED_FOLDER}store_data_summary.txt'
REVIEWS_DATA_SUMMARY_TEXT_FILE = f'{DATA_PROCESSED_FOLDER}reviews_data_summary.txt'


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
