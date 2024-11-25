# Save review data from JSON file to CSV
# Format CSV based on Steamworks API documentation: https://partner.steamgames.com/doc/store/getreviews

# Import libraries
import os
import json

import pandas as pd

import wf_config as config

config.log_section("CONVERT ALL REVIEW JSONs TO ONE CSV")

def convert_reviews_to_csv(input_folder, output_csv):
    """
    Combines all review JSON files into a single CSV file using pandas.

    Args:
        input_folder (str): Path to the folder containing the JSON review files.
        output_csv (str): Path to save the combined CSV file.

    Notes:
        - JSON files must follow the naming pattern 'review_<app_id>.json'.
        - The output CSV will include all relevant fields, including flattened 'author' fields and the app_id.
    """
    all_reviews = []  # List to store all review data
    
    for file_name in os.listdir(input_folder):
        if file_name.startswith("review_") and file_name.endswith(".json"):
            app_id = file_name.split("_")[1].split(".")[0]  # Extract the app ID as a string
            file_path = os.path.join(input_folder, file_name)
            
            # Load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process reviews
            for review_id, review_data in data.get('reviews', {}).items():
                # Flatten the 'author' data into the main dictionary
                author_data = review_data.pop('author', {})
                review_data.update(author_data)
                
                # Add the 'app_id' and 'review_id' for context
                review_data['app_id'] = app_id 
                review_data['review_id'] = review_id

                # Append to the list
                all_reviews.append(review_data)

    # Convert to DataFrame
    if all_reviews:
        df = pd.DataFrame(all_reviews)

        # Reorder columns for better readability
        preferred_order = [
            'app_id',
            'review_id',
            'steamid',
            'language',
            'review',
            'timestamp_created',
            'timestamp_updated',
            'timestamp_dev_responded',
            'developer_response',
            'voted_up',
            'votes_up',
            'votes_funny',
            'weighted_vote_score',
            'comment_count',
            'steam_purchase',
            'received_for_free',
            'written_during_early_access',
            'hidden_in_steam_china',
            'steam_china_location',
            'num_games_owned',
            'num_reviews',
            'playtime_forever',
            'playtime_last_two_weeks',
            'playtime_at_review',
            'last_played',
            'primarily_steam_deck',
            'deck_playtime_at_review'
        ]

        # Align DataFrame columns to preferred order
        df = df[[col for col in preferred_order if col in df.columns]]

        # Save to CSV
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Combined reviews saved to {output_csv}")
    else:
        print("No reviews found. CSV not created.")

    
def main():
    # Define input folder and output CSV paths
    input_folder = os.path.join(config.DATA_GEN_FOLDER, "data")  # Folder with review JSONs
    output_csv = os.path.join(config.STEAM_REVIEWS_DATA)  # Output is named 'combined_reviews.csv'
    
    # Convert reviews to CSV
    convert_reviews_to_csv(input_folder, output_csv)

if __name__ == "__main__":
    main()