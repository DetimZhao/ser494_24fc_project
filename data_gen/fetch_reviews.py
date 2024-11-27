# Fetch reviews from Steam API using the steamreviews library
# steamreviews saves the review data to the data/ directory as JSON files
# steamreviews expects the input and outputs to be in root directory, so we temporarily change the working directory for this script
# Steam reviews were last fetched on 2024-11-05

# Import libraries
import os
import time
import atexit
import logging

import steamreviews

import wf_config as config

def set_working_directory():
    """ 
    Sets the working directory to the directory where this script is located.
    This ensures that relative file paths within the script work as expected.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logging.info(f"Changed working directory to: {script_dir}")


def reset_working_directory(original_dir):
    """
    Resets the working directory back to the original directory before script execution.

    Args:
        original_dir (str): The original directory path to reset the working directory to.
    """
    os.chdir(original_dir)
    logging.info(f"Reset working directory to: {original_dir}")


def fetch_reviews(day_range=180, verbose=False):
    """
    Fetch reviews from Steam API limited to a specified number of days.

    This function retrieves user reviews for a specified app ID from the Steam platform. 
    It allows filtering by language, helpfulness (default), and a time range to limit data volume.

    Args:
        day_range (int): The number of past days to consider for fetching reviews. 
                         Reviews older than this range will be excluded. Default is 180 (6 months) days.

    Notes:
        - The Steam API sorts reviews by helpfulness when the 'filter' parameter is set to 'all'.
        - Reviews are limited to the English language.
        - More filters can be found in the source code of the `steamreviews` library: https://github.com/woctezuma/download-steam-reviews/blob/master/steamreviews/download_reviews.py
            - filters from Steamworks API documentation: https://partner.steamgames.com/doc/store/getreviews
    """

    # Define request parameters
    request_params = {
        'filter': 'all',        # Consider all reviews 
        'language': 'english',  # Fetch reviews in English
        'day_range': day_range  # Consider reviews from the last day_range days
    }

    # Call the batch download function with logging around the cooldown
    while True:
        try:
            logging.info("Starting review download batch...")
            
            # Attempt batch download
            steamreviews.download_reviews_for_app_id_batch(
                chosen_request_params=request_params,
                verbose=verbose
            )
            
            logging.info("Completed review download batch.")
            break  # Exit loop if batch download completes successfully

        except Exception as e:
            # Log any issues with API limits or other exceptions
            logging.warning(f"An error occurred: {e}")
            logging.info("Entering cooldown period...")

            # Implement a cooldown to avoid hitting API limits
            cooldown_duration = steamreviews.get_steam_api_rate_limits()["cooldown"]
            time.sleep(cooldown_duration)
            logging.info("Resuming review download after cooldown.")


def log_execution_time(start_time):
    """Logs the total execution time."""
    elapsed_time = time.time() - start_time
    print(f"Total elasped time of fetching reviews: {elapsed_time:.2f} seconds.")


def reviews_already_fetched(data_folder, data_gen_folder, processed_file_prefix="idprocessed_on_"):
    """
    Checks if review data has already been fetched by verifying the existence
    of JSON review files and a processed indicator file in the data folder.

    Args:
        data_folder (str): Path to the folder containing the JSON review files.
        processed_file_prefix (str): Prefix of the processed indicator file.

    Returns:
        bool: True if both JSON files and a processed indicator file exist, False otherwise.
    """
    # Check for the existence of JSON files in the data folder
    json_files_exist = any(f.endswith('.json') for f in os.listdir(data_folder))
    if not json_files_exist:
        logging.info(f"No JSON review files found in {data_folder}.")
        return False

    # Check for the existence of a processed indicator file
    processed_files_exist = any(f.startswith(processed_file_prefix) for f in os.listdir(data_gen_folder))
    if not processed_files_exist:
        logging.info(f"No processed indicator file found in {data_gen_folder}.")
        return False

    # Both JSON files and the processed indicator file exist
    print(".JSON files and processed indicator file found. Skipping fetching process.")
    return True


def main():
    config.log_section("SETUP TO FETCH REVIEWS")

    output_data = os.path.join(config.DATA_GEN_FOLDER, "data")  # Folder with review JSONs
    data_gen_folder = config.DATA_GEN_FOLDER  # Folder with processed indicator file

    # Check if reviews have already been fetched
    if reviews_already_fetched(output_data, data_gen_folder):
        print("Reviews already fetched. Skipping fetching process.")
        return  # Exit early    

    # Record the start time within main
    start_time = time.time()

    # Register the log_execution_time to run upon script exit
    # atexit.register(log_execution_time, start_time) # Uncomment this line to log the execution time

    # Store the original directory
    original_directory = os.getcwd()
    
    # Change to script's directory
    set_working_directory()

    day_range = 180  # Number of days to consider for fetching reviews (defualt: 180 days)

    logging.info(f"Fetching reviews for the past {day_range} days...")

    try:
        config.log_section("FETCHING REVIEWS FROM STEAM API...")

        # Fetch reviews with the specified day range
        fetch_reviews(day_range=day_range)

        # Log completion of the fetching process
        config.log_section("FETCHING PROCESS DONE")
    finally:
        # Reset the directory back to original
        reset_working_directory(original_directory)

    logging.info("COMPLETE FETCHING PROCESS DONE.")

if __name__ == '__main__':
    main()