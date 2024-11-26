# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import wf_config as config

# %%
# LOAD CLEANED DATA AND PREPARE FOR VISUALIZATION

def load_and_prepare_data():
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


# %%
# CREATE CORRELATION MATRIX AND SAVE TO FILE

def create_and_save_correlation_matrix(data_visualization, quantitative_columns):
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
        print(f"Correlation Matrix saved successfully to path: {corr_matrix_file_path}")
    except Exception as e:
        print(f"Error saving Correlation Matrix to path: {corr_matrix_file_path}")
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


# %%
# CREATE SCATTER PLOTS FOR QUANTITATIVE COLUMNS

# Create scatter plots for all pairs of quantitative features 
def create_and_save_scatterplot(data_visualization, quantitative_columns):
    for i in range(len(quantitative_columns)):
        for j in range(i + 1, len(quantitative_columns)):
            plt.figure(figsize=(10, 6))
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
            print(f"Saved scatter plot: scatterplot_{quantitative_columns[i]} vs {quantitative_columns[j]}.png\n")


# %%
# CREATE BAR GRAPHS FOR QUALITATIVE COLUMNS

# Create bar graphs for each qualitative feature (limiting large categories to the top 20)
def create_and_save_bar_graphs(data_visualization, qualitative_columns):
    # Set a threshold for large categorical columns (like Game Title)
    max_categories = 20

    # Create bar graph for each qualitative feature
    for col in qualitative_columns:
        plt.figure(figsize=(10, 6))
        
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
        plt.savefig(f'{config.VISUALIZATIONS_FOLDER}bar_graph_{col}.png')
        print(f"Saved bar graph: bar_graph_{col}.png\n")



# %%
# CREATE HISTOGRAMS FOR QUALITATIVE COLUMNS

def create_and_save_histograms(data_visualization):
    # Plot the distribution of 'Overall Positive Review Percentage' as a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_visualization, x='Positive Review Percentage', bins=20, kde=True)

    plt.title('Distribution of Overall Positive Review Percentage', fontsize=14)
    plt.xlabel('Overall Positive Review Percentage (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(f'{config.VISUALIZATIONS_FOLDER}histogram_positive_review_percentage.png')
    # plt.show()
    print("Saved histogram: histogram_positive_review_percentage.png\n")


def main():
    config.log_section("DATA VISUALIZATION")
    data_visualization, quantitative_columns, qualitative_columns = load_and_prepare_data()
    create_and_save_correlation_matrix(data_visualization, quantitative_columns)  
    create_and_save_scatterplot(data_visualization, quantitative_columns) 
    create_and_save_bar_graphs(data_visualization, qualitative_columns) 
    create_and_save_histograms(data_visualization) 

# %%

if __name__ == '__main__':
    main()