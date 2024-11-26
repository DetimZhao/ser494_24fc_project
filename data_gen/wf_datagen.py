# The main file that calls the necessary functions to generate the Steam reviews data.

from data_gen import generate_filtered_app_ids, fetch_reviews, review_json_to_csv

def main():
    # Get Steam review data (commented out as it takes a long time to run) and because data exists
    generate_filtered_app_ids.main() # Generate filtered app ids for the reviews to fetch
    fetch_reviews.main() # Fetch reviews for the filtered app ids
    review_json_to_csv.main() # Convert JSON to CSV

if __name__ == '__main__':
    main()