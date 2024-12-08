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

# CONSTANT LIST FOR COMPUTATION OF MATRIX. Column names of combined dataset features for custom distance computation
COL_NAMES_DIST_FUNC_INPUT = [
        'original_price_INR',
        'percentage_of_original_price',
        'discounted_price_INR',
        'dlc_available',
        'age_restricted',
        'awards_count',
        'overall_positive_review_percentage',
        'overall_review_count',
        'overall_review_encoded',
        'mean_compound',
        'mean_positive',
        'mean_negative',
        'mean_neutral',
        'mean_engagement_ratio',
        'mean_playtime_percentile_review',
        'mean_playtime_percentile_total',
        'mean_votes_up',
        'mean_votes_funny',
        'mean_weighted_vote_score',
        'median_playtime_at_review',
        'mean_review_length',
        'mean_word_count',
        'genres_tfidf_access',
        'genres_tfidf_action',
        'genres_tfidf_adventure',
        'genres_tfidf_animation',
        'genres_tfidf_audio',
        'genres_tfidf_casual',
        'genres_tfidf_design',
        'genres_tfidf_development',
        'genres_tfidf_early',
        'genres_tfidf_education',
        'genres_tfidf_free',
        'genres_tfidf_game',
        'genres_tfidf_illustration',
        'genres_tfidf_indie',
        'genres_tfidf_massively',
        'genres_tfidf_modeling',
        'genres_tfidf_movie',
        'genres_tfidf_multiplayer',
        'genres_tfidf_play',
        'genres_tfidf_production',
        'genres_tfidf_publishing',
        'genres_tfidf_racing',
        'genres_tfidf_rpg',
        'genres_tfidf_simulation',
        'genres_tfidf_software',
        'genres_tfidf_sports',
        'genres_tfidf_strategy',
        'genres_tfidf_to',
        'genres_tfidf_training',
        'genres_tfidf_utilities',
        'genres_tfidf_video',
        'genres_tfidf_web',
        'categories_tfidf_about',
        'categories_tfidf_achievements',
        'categories_tfidf_anti',
        'categories_tfidf_app',
        'categories_tfidf_available',
        'categories_tfidf_captions',
        'categories_tfidf_cards',
        'categories_tfidf_cheat',
        'categories_tfidf_cloud',
        'categories_tfidf_co',
        'categories_tfidf_collectibles',
        'categories_tfidf_commentary',
        'categories_tfidf_controller',
        'categories_tfidf_cross',
        'categories_tfidf_editor',
        'categories_tfidf_enabled',
        'categories_tfidf_family',
        'categories_tfidf_features',
        'categories_tfidf_game',
        'categories_tfidf_hdr',
        'categories_tfidf_hl2',
        'categories_tfidf_in',
        'categories_tfidf_includes',
        'categories_tfidf_is',
        'categories_tfidf_lan',
        'categories_tfidf_leaderboards',
        'categories_tfidf_learning',
        'categories_tfidf_level',
        'categories_tfidf_limited',
        'categories_tfidf_mmo',
        'categories_tfidf_mods',
        'categories_tfidf_multiplayer',
        'categories_tfidf_notifications',
        'categories_tfidf_on',
        'categories_tfidf_online',
        'categories_tfidf_only',
        'categories_tfidf_op',
        'categories_tfidf_phone',
        'categories_tfidf_platform',
        'categories_tfidf_play',
        'categories_tfidf_player',
        'categories_tfidf_profile',
        'categories_tfidf_purchases',
        'categories_tfidf_pvp',
        'categories_tfidf_remote',
        'categories_tfidf_require',
        'categories_tfidf_screen',
        'categories_tfidf_sdk',
        'categories_tfidf_shared',
        'categories_tfidf_sharing',
        'categories_tfidf_single',
        'categories_tfidf_source',
        'categories_tfidf_split',
        'categories_tfidf_stats',
        'categories_tfidf_steam',
        'categories_tfidf_steamvr',
        'categories_tfidf_support',
        'categories_tfidf_supported',
        'categories_tfidf_tablet',
        'categories_tfidf_this',
        'categories_tfidf_together',
        'categories_tfidf_tracked',
        'categories_tfidf_trading',
        'categories_tfidf_turn',
        'categories_tfidf_tv',
        'categories_tfidf_valve',
        'categories_tfidf_vr',
        'categories_tfidf_workshop'
    ] 


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
