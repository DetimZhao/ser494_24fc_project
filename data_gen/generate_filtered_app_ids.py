# Generates a list of app/game IDs based on minimum review count and save a random sample. 
# This list of games is used for fetching reviews from the Steam API.

import math

import pandas as pd

import wf_config as config



def filter_app_ids(data_file, output_file, sample_size=100, random_state=42):
    """
    Filters app IDs based on minimum review count and saves a random sample. 
    Dynmically selects the minimum review count based on the median value. Read below for more details.

    For a balanced approach, we use the the median value of overall_review_count. 
    It is generally more robust since we have large variations in review counts (indicated by a high mean). 
    This helps avoid excessive bias toward highly popular games while selecting games with a reasonable level of interaction.

    Args:
    - data_file (str): Path to the cleaned dataset.
    - output_file (str): Path to save the filtered and sampled app IDs.
    - min_reviews (int): Minimum review count threshold.
    - sample_size (int): Number of app IDs to sample.
    - random_state (int): Seed for reproducibility.
    """
    # Load data
    df = pd.read_csv(data_file)
    
    # Dynamically calculate the median of 'overall_review_count'
    min_reviews = int(df['overall_review_count'].median())
    print(f"Using median review count as threshold: {min_reviews}")
    
    # Filter games with at least `min_reviews`
    filtered_df = df[df['overall_review_count'] >= min_reviews]
    
    # Randomly sample `sample_size` games from filtered DataFrame
    if len(filtered_df) > sample_size:
        sampled_df = filtered_df.sample(n=sample_size, random_state=random_state)
    else:
        print("WARNING: Sample size larger than available filtered games.")
        sampled_df = filtered_df
    
    # Save app IDs to a text file
    with open(output_file, 'w') as f:
        for app_id in sampled_df['app_id']:
            f.write(f"{app_id}\n")
    print(f"\nFILTERING COMPLETE. Filtered app IDs with at least {min_reviews} reviews saved to: {output_file}\n")



def calculate_sample_size(confidence_level=0.95, margin_of_error=0.05, proportion=0.5):
    """
    Calculate the sample size needed for a given confidence level and margin of error.

    Parameters:
    - population_size (int): Total population size.
    - confidence_level (float): Confidence level for the sample (e.g., 0.95 for 95% confidence).
    - margin_of_error (float): Margin of error (e.g., 0.05 for 5% margin of error).
    - proportion (float): Estimated proportion of the population (default 0.5 for maximum variability).

    Returns:
    - int: Sample size needed.
    """
     
    # Z-scores for common confidence levels
    z_scores = {
        0.90: 1.645,    
        0.95: 1.96,
        0.99: 2.576
    }
    
    # Get Z-score based on confidence level
    if confidence_level not in z_scores:
        raise ValueError("ERROR: Value not supported. Supported confidence levels are 0.90, 0.95, and 0.99.")
    
    Z = z_scores[confidence_level]
    
    # Calculate sample size for a large population
    numerator = (Z**2) * proportion * (1 - proportion)
    denominator = margin_of_error**2
    sample_size = numerator / denominator
    
    # Round up to ensure sufficient sample size
    return math.ceil(sample_size)




if __name__ == "__main__":

    config.log_section("FILTERING APP IDs BASED ON MINIMUM REVIEW COUNT...")

    # Default path to the cleaned dataset in this project is found in 'data_processed/steam-store-data-cleaned.csv'
    input_data_path = config.STEAM_STORE_DATA_CLEANED

    # Default values for confidence level and margin of error
    confidence_level = 0.95
    margin_of_error = 0.05
    
    # Calculate sample size based on the dataset with the default 95% confidence level and 5% margin of error
    sample_size = calculate_sample_size(confidence_level=confidence_level, margin_of_error=margin_of_error) 
    print(f"Recommended sample size with {confidence_level*100}% confidence level and {margin_of_error*100}% margin of error: {sample_size}")

    output_file_name = config.IDS_LIST_FILE # steamreviews expects this exact filename for input

    # Filter app IDs based on minimum review count and save a random sample
    filter_app_ids(
        data_file=input_data_path,      # Path to the dataset.
        output_file=output_file_name,   # Output file to save filtered IDs for fetching reviews
        sample_size=sample_size,        # Number of app IDs to sample, calculated based on data
        random_state=47                 # Seed for reproducibility (42 is common but I use 47 in my project)
    )


