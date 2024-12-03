# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import wf_config as config

# %%
# LOAD CLEANED DATA AND PREPARE FOR VISUALIZATION

def load_and_prepare_store_data():
    # Load the cleaned data
    data_cleaned = pd.read_csv('data_processed/steam-store-data-cleaned.csv', parse_dates=['release_date'])

    # Make a dictionary to map original column names to nicer names for visualization
    column_name_mapping = {
        'game_title': 'Game Title',
        'release_date': 'Release Date',
        'genres': 'Genres',
        'categories': 'Categories',
        'developer': 'Developer',
        'publisher': 'Publisher',
        'original_price_INR': 'Original Price (INR)',
        'percentage_of_original_price': 'Percentage of Original Price',
        'discounted_price_INR': 'Discounted Price (INR)',
        'dlc_available': 'DLC Available',
        'age_restrcited': 'Age Restrcited',
        'store_game_description': 'Store Game Description',
        'win_support': 'Windows Support',
        'mac_support': 'Mac Support',
        'linux_support': 'Linux Support',
        'awards_count': 'Awards Count',
        'overall_review': 'Overall Review',
        'overall_positive_review_percentage': 'Positive Review Percentage',
        'overall_review_count': 'Review Count'
    }

    # Use the rename method to apply the new names
    data_visualization = data_cleaned.rename(columns=column_name_mapping)

    # Filter out rows with -1 (indicating no reviews) in Positive Review Percentage column
    data_visualization = data_visualization[data_visualization['Positive Review Percentage'] != -1]

    # Only select quantitative columns for visualization
    quantitative_columns = ['Original Price (INR)', 'Discounted Price (INR)', 
                            'Awards Count', 'Positive Review Percentage', 'Review Count']

    # Only select qualitative columns for visualization
    qualitative_columns = ['Game Title', 'Genres', 'Categories', 'Developer', 'Publisher', 'Overall Review']

    return data_visualization, quantitative_columns, qualitative_columns



def load_and_prepare_reviews_data():
    # Load the cleaned reviews data
    reviews_data_path = config.STEAM_REVIEWS_DATA_CLEANED
    data_cleaned = pd.read_csv(reviews_data_path, parse_dates=['timestamp_created', 'timestamp_updated', 'last_played'])

    # Make a dictionary to map original column names to nicer names for visualization
    column_name_mapping = {
        'app_id': 'App ID',
        'review_id': 'Review ID',
        'review': 'Review Text',
        'timestamp_created': 'Timestamp Created',
        'timestamp_updated': 'Timestamp Updated',
        'voted_up': 'Voted Up',
        'votes_up': 'Votes Up',
        'votes_funny': 'Votes Funny',
        'weighted_vote_score': 'Weighted Vote Score',
        'comment_count': 'Comment Count',
        'steam_purchase': 'Steam Purchase',
        'received_for_free': 'Received for Free',
        'num_games_owned': 'Number of Games Owned',
        'num_reviews': 'Number of Reviews',
        'playtime_forever': 'Playtime Forever',
        'playtime_last_two_weeks': 'Playtime Last Two Weeks',
        'playtime_at_review': 'Playtime at Review',
        'last_played': 'Last Played',
        'review_age_days': 'Review Age (Days)',
        'updated_review_age_days': 'Updated Review Age (Days)',
        'last_played_days': 'Last Played (Days)',
        'engagement_ratio': 'Engagement Ratio',
        'playtime_percentile_review': 'Playtime Percentile at Review',
        'playtime_percentile_total': 'Playtime Percentile Total',
        'review_length': 'Review Length',
        'word_count': 'Word Count',
        'compound': 'Compound Sentiment',
        'positive': 'Positive Sentiment',
        'negative': 'Negative Sentiment',
        'neutral': 'Neutral Sentiment'
    }

    # Use the rename method to apply the new names
    data_visualization = data_cleaned.rename(columns=column_name_mapping)

    # Only select quantitative columns for visualization
    quantitative_columns = [
        'Votes Up',
        'Weighted Vote Score', 
        'Review Age (Days)', 
        'Engagement Ratio', 
        'Playtime Percentile at Review', 
        'Compound Sentiment'
    ]

    # Only select qualitative columns for visualization
    qualitative_columns = [
        'Review Text', 
        'Voted Up', 
        'Steam Purchase', 
        'Received for Free'
    ]

    return data_visualization, quantitative_columns, qualitative_columns


def load_and_prepare_combined_data():
    # Load the combined clustering dataset
    data = pd.read_csv(config.COMBINED_CLUSTERING_STEAM_DATASET)

    # Only select quantitative columns for visualization
    quantitative_columns = [
        'mean_playtime_percentile_review', 
        'mean_playtime_percentile_total',
        'mean_votes_up', 
        'mean_votes_funny',
        'median_playtime_at_review', 
        'mean_review_length', 
        'mean_word_count'
    ]

    # Only select qualitative columns for visualization
    qualitative_columns = [
        'overall_review', 
    ]

    return data, quantitative_columns, qualitative_columns


# %%
# CREATE CORRELATION MATRIX AND SAVE TO FILE

def create_and_save_correlation_matrix(data_visualization, quantitative_columns):
    # TODO - Fix the function to take in either the store or review data 

    # Calculate the correlation matrix
    correlation_matrix = data_visualization[quantitative_columns].corr().round(2)

    # Display the correlation matrix
    formatted_corr_matrix = correlation_matrix.to_string() 

    # File path and filename to save to
    corr_matrix_file_name = 'correlations.txt'
    corr_matrix_file_path = f'data_processed/{corr_matrix_file_name}'

    # Save the correlation matrix to a file
    try:
        with open(corr_matrix_file_path, 'w') as f:
            f.write(formatted_corr_matrix)
        print(f"Correlation Matrix saved successfully to path: {corr_matrix_file_path}\n")
    except Exception as e:
        print(f"Error saving Correlation Matrix to path: {corr_matrix_file_path}\n")
        print(e)

    # File path and filename to save to
    corr_matrix_heatmap_file_name = 'correlations.png'
    corr_matrix_heatmap_file_path = f'{config.VISUALIZATIONS_FOLDER}{corr_matrix_heatmap_file_name}'

    # Save the correlation matrix heatmap to a file
    # Create a heatmap to visualize the correlations (half matrix)
    plt.figure(figsize=(12, 10))  
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)  
    plt.title("Correlation Matrix (Quantitative Features)")  
    plt.savefig(corr_matrix_heatmap_file_path)
    # plt.show()
    plt.close('all')


# %%
# CREATE SCATTER PLOTS FOR QUANTITATIVE COLUMNS

# Create scatter plots for all pairs of quantitative features 
def create_and_save_scatterplot(data_visualization, quantitative_columns):
    # TODO - make axis log scale when necessary
    for i in range(len(quantitative_columns)):
        for j in range(i + 1, len(quantitative_columns)):
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=data_visualization[quantitative_columns[i]], 
                            y=data_visualization[quantitative_columns[j]],
                            alpha=0.6, s=50)
            
            plt.title(f"{quantitative_columns[i]} vs {quantitative_columns[j]}", fontsize=14)
            plt.xlabel(quantitative_columns[i], fontsize=12)
            plt.ylabel(quantitative_columns[j], fontsize=12)

            # Log scale for Original Price and Discounted Price since they are rupees with a high value range
            # Apply log scale to X axis if 'Original Price (INR)' or 'Discounted Price (INR)'  is on X axis
            if quantitative_columns[i] in ['Original Price (INR)', 'Discounted Price (INR)']:
                plt.xscale('log')
            
            # Apply log scale to Y axis if 'Original Price (INR)' or 'Discounted Price (INR)' is on Y axis
            if quantitative_columns[j] in ['Original Price (INR)', 'Discounted Price (INR)']:
                plt.yscale('log')

            plt.savefig(f'{config.VISUALIZATIONS_FOLDER}scatterplot_{quantitative_columns[i]} vs {quantitative_columns[j]}.png')
            # plt.show()
            plt.close()
            print(f"Saved scatter plot: scatterplot_{quantitative_columns[i]} vs {quantitative_columns[j]}.png\n")


# %%
# CREATE BAR GRAPHS FOR QUALITATIVE COLUMNS

# Create bar graphs for each qualitative feature (limiting large categories to the top 20)
def create_and_save_bar_graphs(data_visualization, qualitative_columns):
    # Set a threshold for large categorical columns (like Game Title)
    max_categories = 20

    # Create bar graph for each qualitative feature
    for col in qualitative_columns:
        plt.figure(figsize=(12, 8))
        
        # Check if the column has more than max_categories unique values
        if data_visualization[col].nunique() > max_categories:
            # For large columns, only plot the top max_categories
            top_categories = data_visualization[col].value_counts().nlargest(max_categories).index
            sns.countplot(y=data_visualization[col], order=top_categories)
            plt.title(f"Top {max_categories} {col}", fontsize=14)
        else:
            # For smaller columns (like Overall Review), plot all categories
            sns.countplot(y=data_visualization[col], order=data_visualization[col].value_counts().index)
            plt.title(f"Distribution of {col}", fontsize=14)
        
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(col, fontsize=12)
        # plt.show()
        plt.close()
        print(f"Saved bar graph: bar_graph_{col}.png\n")



# %%
# CREATE HISTOGRAMS FOR QUALITATIVE COLUMNS

def create_and_save_histograms(data_visualization):
    # Plot the distribution of 'Overall Positive Review Percentage' as a histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(data=data_visualization, x='Positive Review Percentage', bins=20, kde=True)

    plt.title('Distribution of Overall Positive Review Percentage', fontsize=14)
    plt.xlabel('Overall Positive Review Percentage (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.close()
    print("Saved histogram: histogram_positive_review_percentage.png\n")
    # plt.show()


def inspect_outliers(data, columns_to_check):
    """
    Inspect potential outliers in numeric columns using summary statistics and visualizations.

    Args:
        data (pd.DataFrame): The dataset.
        columns_to_check (list): List of column names to inspect.

    Returns:
        None: Displays the summary and visualizations.
    """
    for col in columns_to_check:
        print(f"Summary statistics for '{col}':\n")
        print(data[col].describe())
        print("\n" + "-"*50 + "\n")
        
        # Plot histogram
        plt.figure(figsize=(10, 5))
        plt.hist(data[col], bins=30, edgecolor='k', alpha=0.7)
        plt.title(f"Histogram for {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(f'{config.VISUALIZATIONS_FOLDER}histogram_inspect_outliers_{col}.png')
        print(f"Saved histogram for {col} to: {config.VISUALIZATIONS_FOLDER}histogram_inspect_outliers_{col}.png\n")
        # plt.show()

        # Plot boxplot
        plt.figure(figsize=(10, 5))
        plt.boxplot(data[col].dropna(), vert=False, patch_artist=True)
        plt.title(f"Boxplot for {col}")
        plt.xlabel(col)
        plt.savefig(f'{config.VISUALIZATIONS_FOLDER}boxplot_inspect_outliers_{col}.png')
        print(f"Saved boxplot for {col} to: {config.VISUALIZATIONS_FOLDER}boxplot_inspect_outliers_{col}.png\n")
        # plt.show()


def check_for_outliers():
    """
    Check for outliers in the specified columns of the dataset.
    It is used for testing and checking purposes to ensure that 
    no significant outliers are missed in the dataset.
    """
    
    # Check the combined clustering dataset for outliers since we could have missed them
    data = pd.read_csv(config.COMBINED_CLUSTERING_STEAM_DATASET)    # Columns to check
    columns_to_check = [
        'mean_playtime_percentile_review', 
        'mean_playtime_percentile_total',
        'mean_votes_up', 
        'mean_votes_funny',
        'median_playtime_at_review', 
        'mean_review_length', 
        'mean_word_count'
    ]
    inspect_outliers(data, columns_to_check)
    # Insight from the above analysis (before log transformation btw):
    # - The playtime columns show a bell-shaped distribution with some outliers at the higher percentiles.
    # - Both vote distributions are highly skewed.
    # - The playtime_at_review column is highly skewed with some extreme outliers (e.g., values > 30,000).
    # - The review and word count distributions are slightly skewed with a few outliers.
    # As a result from the above analysis, we will apply log transformations to the skewed columns.


def run_all_data_visualizations(load_and_prepare_data_func):
    # Load and prepare the data using the provided function
    data_visualization, quantitative_columns, qualitative_columns = load_and_prepare_data_func()
    
    # Create and save visualizations
    create_and_save_correlation_matrix(data_visualization, quantitative_columns)  
    create_and_save_scatterplot(data_visualization, quantitative_columns) 
    create_and_save_bar_graphs(data_visualization, qualitative_columns) 
    create_and_save_histograms(data_visualization) 
    plt.close('all') # Cleanup


def plot_review_creation_histogram():
    """
    Plot a histogram of review creation timestamps of the original reviews dataset.

    Args:
        data (pd.DataFrame): The original review data containing `timestamp_created`.
        cutoff_date (str): The cutoff date as a string (e.g., '2024-05-25').
    """

    # Load the original reviews data
    original_reviews_data = pd.read_csv(f'{config.DATA_ORIGINAL_FOLDER}combined_reviews.csv')
    data = original_reviews_data.copy()
    cutoff_date = "2024-05-25"

    # Ensure `timestamp_created` is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp_created']):
        data['timestamp_created'] = pd.to_datetime(data['timestamp_created'], unit='s')

    # Plot the histogram using seaborn
    print(f"Checking for reviews after {cutoff_date}...")
    plt.figure(figsize=(10, 6))
    num_bins = int(np.sqrt(len(data['timestamp_created'])))
    print(f"Number of bins: {num_bins}")
    print(f"Number of Reviews before {cutoff_date}: {len(data[data['timestamp_created'] <= cutoff_date])} out of {len(data)}")
    print(f"Number of Reviews after {cutoff_date}: {len(data[data['timestamp_created'] > cutoff_date])} out of {len(data)}")
    print(f"Percentage of Reviews after {cutoff_date}: {len(data[data['timestamp_created'] > cutoff_date]) / len(data) * 100:.2f}%")
    sns.histplot(data['timestamp_created'], bins=num_bins, color='blue')
    plt.axvline(pd.Timestamp(cutoff_date), color='red', linestyle='--', label=f'Cutoff Date: {cutoff_date}')
    plt.title('Histogram of Review Creation Dates')
    plt.xlabel('Review Creation Date')
    plt.ylabel('Number of Reviews')
    plt.legend()
    plt.savefig(f'{config.VISUALIZATIONS_FOLDER}histogram_review_creation_dates.png')
    print(f"Saved histogram of Review Creation Dates to: {config.VISUALIZATIONS_FOLDER}histogram_review_creation_dates.png\n")
    # plt.show()


def main():
    config.log_section("DATA VISUALIZATION")

    # run_all_data_visualizations(load_and_prepare_store_data) # Visualize store data

    # run_all_data_visualizations(load_and_prepare_reviews_data) # Visualize reviews data

    # Combined data visualizations are not necessary, but it is good to have them for reference
    # check_for_outliers() # Check for outliers in the combined dataset
    # plot_review_creation_histogram() # Plot a histogram of review creation timestamps

    # run_all_data_visualizations(load_and_prepare_combined_data) # Visualize combined clustering data



# %%

if __name__ == '__main__':
    main()