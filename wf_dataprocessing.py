# Import libraries
import re
import ssl
import pickle
import os

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


import wf_config as config

config.log_section("DOWNLOAD FOR DATA CLEANING")
# Suppress SSL certificate verification warnings from: https://stackoverflow.com/a/50406704/15193980
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
print("Downloading NLTK resources... if not already downloaded.")
# Download NLTK resources if not already done
nltk.download('stopwords')                   # Download stopwords for text processing
nltk.download('punkt')                       # Download tokenizer models
nltk.download('wordnet')                     # Download WordNet corpus for lemmatization
nltk.download('omw-1.4')                     # Download Open Multilingual Wordnet
nltk.download('averaged_perceptron_tagger')  # Download POS tagger model
nltk.download('vader_lexicon')               # Download VADER lexicon for sentiment analysis

# Define stopwords list
stop_words = set(stopwords.words('english'))

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()



def vectorize_text_features(data, text_column, vectorizer=None, max_features=50, save_vectorizer_to=None, save_vectorized_results_to=None):
    """
    Vectorizes text features using TfidfVectorizer.

    Args:
        data (pd.DataFrame): The dataset containing the text column to vectorize.
        text_column (str): The column name containing text data (e.g., 'genres').
        vectorizer (TfidfVectorizer): Optional pre-configured TfidfVectorizer.
        max_features (int): Maximum number of features to retain.
        save_vectorizer_to (str): Directory path to save the vectorizer as a pickle file.
        save_vectorized_results_to (str): Directory path to save the vectorized results as a .npy file.

    Returns:
        pd.DataFrame: DataFrame with vectorized features optionally added.
        TfidfVectorizer: The vectorizer used (for saving/reuse).
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform the text column
    tfidf_matrix = vectorizer.fit_transform(data[text_column].fillna(''))

    # Convert to DataFrame
    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Add prefix to avoid column name clashes
    tfidf_features = tfidf_features.add_prefix(f"{text_column}_tfidf_")

    # Reset index to align with the original dataset
    tfidf_features.index = data.index

    # Concatenate vectorized features with original dataset (optional)
    # data = pd.concat([data, tfidf_features], axis=1) # Uncomment to add vectorized features to the cleaned data

    # Save the vectorizer for reuse
    if save_vectorizer_to:
        if not os.path.exists(save_vectorizer_to):
            os.makedirs(save_vectorizer_to, exist_ok=True)
        vectorizer_path = os.path.join(save_vectorizer_to, f"{text_column}_vectorizer.pkl")
        save_object_to_pickle(vectorizer, vectorizer_path)

    # Save the vectorized results for future analysis
    if save_vectorized_results_to:
        if not os.path.exists(save_vectorized_results_to):
            os.makedirs(save_vectorized_results_to, exist_ok=True)
        results_path = os.path.join(save_vectorized_results_to, f"{text_column}_vectorized_results.npy")
        np.save(results_path, tfidf_matrix.toarray())
        print(f"Vectorized results saved to {results_path}")

    return data, vectorizer



def clean_steam_store_data():
    """
    Clean the Steam Store dataset by:
        - Renaming columns for clarity.
        - Converting 'release_date' to datetime dtype.
        - Converting 'original_price' and 'discounted_price' to float dtype.
        - Converting 'discount_percentage' to float dtype.
        - Dropping rows with missing values in key columns: developer, publisher, genres, categories.
        - Dropping rows with missing values in content descriptor tags.
        - Dropping rows with missing values in recent review columns.
        - Dropping rows with missing values in original and discounted prices.
        - Imputing missing values in overall review columns.
        - Dropping rows with missing about_description.
        - Dropping rows with missing OS support columns.
        - TODO: Add more cleaning steps as needed.
        - Saving the cleaned data to a new CSV file.

    Returns:
        - pd.DataFrame: The cleaned Steam Store dataset.
    """
    config.log_section("CLEAN STEAM STORE DATA")
    # %% 
    # LOAD DATA AND INITIAL CHECK OF THE DATASET

    # Load original data
    data = pd.read_csv(config.STEAM_STORE_DATA)

    # Check the first few rows of the dataset
    # data.head()
    # Insights:
    # So the price is in rupees, and the discount is a negative integer with a percentage symbol. 
    # Pandas could be reading some of these as NaN because of their formatting.



    # %% 
    # CHECK DATA TYPES AND MISSING VALUES  

    # Check data types and summary stats
    # print(f"Shape of data: {data.shape}\n") # (41975, 20)
    # print(f"Summary of stats:\n {data.describe()}\n")

    # data.info()
    # RangeIndex: 42497 entries, 0 to 42496
    # Data columns (total 24 columns)

    # Check missing values
    # data.isnull().sum()



    # %%
    # SAMPLE DATA FOR EDA/BASIC QUESTIONS (RECORDS)

    # Random state value for reproducibility
    random_state_value = 47
    # Select two records from the dataset
    sample_record1 = data.iloc[[168]] # Overwatch 2 row (Just chose this for fun. Also found out that 24305 for random_state will give Overwatch 2)
    sample_record2 = data.sample(random_state=random_state_value) # Sample random record with random_state for reproducibility

    # Display the sample records for EDA/BASIC QUESTIONS
    display_sample_records = pd.concat([sample_record1, sample_record2]) # Concatenate the two sample records for pleasant row display
    display_sample_records # Display the sample records when in interactive mode



    # %%
    # START CLEANING DATA BY MAKING A COPY AND RENAMING COLUMNS FOR CLARITY

    data_cleaned = data.copy() # Create a copy of the original data to clean

    # Rename 'title' column to 'game_title' for clarity
    data_cleaned.rename(columns={'title': 'game_title'}, inplace=True)

    # Rename 'age_rating' to 'age_restricted' for clarity
    data_cleaned.rename(columns={'age_rating': 'age_restricted'}, inplace=True)

    # Rename content_descriptor to content_descriptor_tags for clarity
    data_cleaned.rename(columns={'content_descriptor': 'content_descriptor_tags'}, inplace=True)

    # Rename 'about_description' to 'store_game_description' for clarity
    data_cleaned.rename(columns={'about_description': 'store_game_description'}, inplace=True)

    # Rename awards to awards_count 
    data_cleaned.rename(columns={'awards': 'awards_count'}, inplace=True)

    # Rename overall_review_% to overall_positive_review_percentage for clarity
    data_cleaned.rename(columns={'overall_review_%': 'overall_positive_review_percentage'}, inplace=True)

    # Rename recent_review_% to recent_positive_review_percentage for clarity
    data_cleaned.rename(columns={'recent_review_%': 'recent_positive_review_percentage'}, inplace=True)



    # %%
    # DATA CLEANING CONT: CONVERT RELEASE_DATE TO DATETIME

    # Convert 'release_date' to datetime dtype for future analysis
    data_cleaned['release_date'] = pd.to_datetime(data_cleaned['release_date'], errors='coerce')
    # errors='coerce' tells to set any invalid dates as NaT (Not a Time) instead of giving an error
    # - Note: When saving to CSV, it will lose the datetime64 format and will need to be explicitly parsed.
    #   - With pandas, parse_dates=['release_date'] can be used when loading to parse the string back into datetime.

    # data_cleaned['release_date'].head() # Check the first few rows of the release_date column for changes



    # %%
    # DATA CLEANING CONT: CONVERT AND CLARIFY ORIGINAL PRICE AND DISCOUNTED PRICE TO FLOAT

    # Assumptions: 
    # - If the discounted price is 0, then the original price is also 0 (free game).
    # - If the original price is NaN, then the original price is equal to the discounted price.
    #     - Since the dataset doesn't contain timeseries data, the prices are likely reflective of when the scraping took place. 
    #     If there was no sale for a particular game, the discounted_price would simply match the original_price.

    # Convert 'original_price' and 'discounted_price' to float dtype, removing '₹' (rupee symbol) and commas
    data_cleaned['original_price'] = data['original_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
    data_cleaned['discounted_price'] = data['discounted_price'].replace({'₹': '', ',': '', 'Free': '0'}, regex=True).astype(float)

    # Handle missing 'original_price' where 'discounted_price' is 0 (Free games)
    data_cleaned.loc[data_cleaned['discounted_price'] == 0, 'original_price'] = 0.00

    # If 'original_price' is NaN but 'discounted_price' has a value, set 'original_price' equal to 'discounted_price'
    data_cleaned.loc[data_cleaned['original_price'].isna() & data_cleaned['discounted_price'].notna(), 'original_price'] = data_cleaned['discounted_price']

    # Rename columns to indicate rupees (INR) for clarity
    data_cleaned.rename(columns={'original_price': 'original_price_INR'}, inplace=True)
    data_cleaned.rename(columns={'discounted_price': 'discounted_price_INR'}, inplace=True)

    data_cleaned # Check changes
    # data_cleaned.isnull().sum() # Check missing values so far



    # %%
    # DATA CLEANING CONT: CONVERT AND CLARIFY DISCOUNT PERCENTAGE TO FLOAT

    # Convert 'discount_percentage' to represent the percentage of the original price after discount
    data_cleaned['discount_percentage'] = data_cleaned['discount_percentage'].replace({'%': '', '-': ''}, regex=True).astype(float)

    # Convert the discount percentage to the remaining price percentage after the discount by subtracting from 100
    data_cleaned['discount_percentage'] = 100 - data_cleaned['discount_percentage']

    # Rename column to percentage_of_original_price for clarity
    data_cleaned.rename(columns={'discount_percentage': 'percentage_of_original_price'}, inplace=True)

    # Replace NaNs in the discount column with 100 (meaning no discount, it is the same as the original price)
    data_cleaned['percentage_of_original_price'] = data_cleaned['percentage_of_original_price'].fillna(100)

    data_cleaned # Check changes
    # data_cleaned.isnull().sum() # Check missing values so far



    # %%
    # PAUSE DATA CLEANING: CHECK FOR MISSING VALUES IN KEY COLUMNS: developer, publisher

    # Insights (missing_developer):
    # - The Witcher 3: Wild Hunt - Complete Edition: has genres, categories 
    #   - missing developer, pubslisher, and release_date
    # - Middle-earth™: The Shadow Bundle: missing genres, categories, developer, publisher
    # - Batman: Arkham Collection: missing genres, categories, developer, publisher

    # Insights (missing_release_date):
    # - Grand Theft Auto IV: Complete Edition: has genre, categories, developer, publisher
    #   - missing release_date
    # - Dawn of War Franchise Pack: missing description, genres, categories, developer, publisher, release_date
    # - Fallout Classic Collection: missing description, genres, categories, developer, publisher, release_date

    # Assumptions from the insights:
    # - All of these games are some edition, bundle, or collection of a base game or franchise.
    # - Unclear why they are missing developer and publisher, which can be imputed from the base game or franchise, 
    #   BUT likely uncessary for analysis since these are just editions/packs/collections of base games/franchises.



    # %%
    # DATA CLEANING CONT: DROP ROWS WITH MISSING DEVELOPER, PUBLISHER, GENRES, AND CATEGORIES

    # Because all the missing values are less than 1%, we can drop these rows without losing much info.
    # Drop rows where developer, publisher, genres, or categories are missing
    data_cleaned = data_cleaned.dropna(subset=['developer', 'publisher', 'genres', 'categories', 'release_date'])

    # Check cleaned dataset's shape to see how many rows remain 
    # data_cleaned.shape # (42199, 24) where 42199 rows remain after dropping missing values




    # %% 
    # PAUSE DATA CLEANING: CHECK MISSING VALUES IN CONTENT DESCRIPTOR TAGS

    # Assumption and Insights:
    # - 94.39% of the data is missing content descriptor tags, so we can drop this column.
    # - Upon googling, content descriptors (tags) were introduced in 2023, so it makes sense that most of the data is missing this info.



    # %%
    # PAUSE DATA CLEANING: CHECK PERCENTAGES OF MISSING VALUES IN RECENT REVIEW COLUMNS

    # Assumption:
    # - 87.00% of data is missing recent review data, likely because recent reviews only happen for popular games, recent games, or games with recent updates.
    #   - This would make sense, as there are so many games on Steam, and not all of them are popular or have recent updates.
    # - Since the data is missing for most of the dataset, we can drop these columns.



    # %%
    # DATA CLEANING CONT: DROP RECENT REVIEW COLUMNS AND CONTENT DESCRIPTOR TAGS

    # Drop the recent review columns due to the high percentage of missing values
    data_cleaned = data_cleaned.drop(columns=['content_descriptor_tags', 'recent_review', 'recent_positive_review_percentage', 'recent_review_count'])

    # Check the updated data to ensure the columns are dropped
    # data_cleaned.info()
    # data_cleaned.isnull().sum() # Check missing values so far



    # %%
    # PAUSE DATA CLEANING: CHECK MISSING VALUES IN ORIGINAL PRICE AND DISCOUNTED PRICE

    # Assumption and Insights:
    # - BioShock™, DEATH STRANDING DIRECTOR'S CUT, Call of Duty® are all not free games, and they are from different years.
    # - Both original and discounted prices are missing for these games, so the previous assumption that discounted price of 0 (indicating a free game) is not valid here.
    # - In the context of the dataset, it is possible these games may not have been available for purchase on the Indian Steam store at the time of scraping.
    # - It's also possible the data is missing due to scraping issues, special editions/versions, licensing issues, or other reasons.
    # Again, because the missing values are less than 1%, we can drop these rows without losing much info.



    # %%
    # DATA CLEANING CONT: DROP ROWS WITH MISSING ORIGINAL AND DISCOUNTED PRICES

    # Drop rows where original and discounted prices are missing
    data_cleaned = data_cleaned.dropna(subset=['original_price_INR', 'discounted_price_INR'])

    # Check cleaned dataset's shape to see how many rows remain 
    # data_cleaned.shape # (41979, 20)



    # %%
    # PAUSE DATA CLEANING: CHECK MISSING VALUES IN OVERALL REVIEW COLUMNS

    # Assumption and Insights:
    # - Recently Released Games: Games released around May 2024 haven’t had enough time to accumulate reviews (data was scraped in May 2024).
    # - Niche Games: Older obscure games with very few or no reviews, likely due to their limited audience.
    # - Will impute missing values for overall review count with 0 to indicate no reviews yet.
    #   - Same will be done for overall review percentage and overall review count. 
    #   - If these imputed values affect an analysis or distribution, we can still filter them out later based on conditions.



    # %%
    # DATA CLEANING CONT: IMPUTE MISSING VALUES IN OVERALL REVIEW COLUMNS

    # Impute 'overall_review' with 'No Reviews' and 'overall_review_count' with 0
    data_cleaned['overall_review'] = data_cleaned['overall_review'].fillna('No Reviews')
    data_cleaned['overall_review_count'] = data_cleaned['overall_review_count'].fillna(0)

    # For 'overall_positive_review_percentage', fill with -1 (placeholder/special case) to indicate no reviews
    # We do not use 0 since that could imply 0% positive (very negative) reviews, which is different from no reviews
    data_cleaned['overall_positive_review_percentage'] = data_cleaned['overall_positive_review_percentage'].fillna(-1)

    # Check the updated data to ensure the columns are imputed
    imputed_overall_reviews_head =  data_cleaned.loc[data_cleaned['overall_review_count'] == 0].head() # Check the imputed values of the head
    imputed_overall_reviews_tail = data_cleaned.loc[data_cleaned['overall_review_count'] == 0].tail() # Check the imputed values of the tail

    imputed_overall_reviews_check = pd.concat([imputed_overall_reviews_head, imputed_overall_reviews_tail]) # Concatenate head and tail for display
    imputed_overall_reviews_check.loc[:, ['game_title', 'store_game_description', 'release_date', 'overall_review', 'overall_positive_review_percentage', 'overall_review_count']]



    # %%
    # PAUSE DATA CLEANING: CHECK MISSING VALUES IN ABOUT_DESCRIPTION

    # Assumption and Insights:
    # - These four games are seemingly missing completely at random, and there are only 4 vaules so dropping them won't affect the dataset.



    # %%
    # DATA CLEANING CONT.: DROP ROWS WITH MISSING ABOUT_DESCRIPTION AND OS SUPPORT COLUMNS

    # Drop rows where 'store_game_description' is missing
    data_cleaned = data_cleaned.dropna(subset=['store_game_description'])
    
    # Drop OS support columns (win_support, mac_support, linux_support) since they are not needed for analysis
    data_cleaned = data_cleaned.drop(columns=['win_support', 'mac_support', 'linux_support'])



    # FINAL DATA CLEAING: VECTORIZE TEXT FEATURES (GENRES, CATEGORIES)
    vectorizer_save_path = config.VECTORIZERS_FOLDER
    vectorized_results_save_path = config.VECTORIZED_RESULTS_FOLDER
    
    # Vectorize 'genres' and 'categories'
    data_cleaned, genres_vectorizer = vectorize_text_features(
        data_cleaned, 
        'genres', 
        max_features=100, 
        save_vectorizer_to=vectorizer_save_path, 
        save_vectorized_results_to=vectorized_results_save_path
    )
    
    data_cleaned, categories_vectorizer = vectorize_text_features(
        data_cleaned, 
        'categories', 
        max_features=100, 
        save_vectorizer_to=vectorizer_save_path, 
        save_vectorized_results_to=vectorized_results_save_path
    )

    # data_cleaned.info()

    # %% 
    # CHECK AFTER ALL DATA CLEANING (FINAL CHECK)
    # - This cell has been ran multiple times to check the data after each cleaning step
    # - This cell is also what drove some of the assumptions and decisions made in the cleaning process
    # - This cell is also for checking the data after all cleaning steps are dones

    # Final check on the cleaned data
    # data_cleaned.shape
    # print(f"Shape of data after cleaning: {data_cleaned.shape}\n") # (41916, 17)
    # data_cleaned.info()
    # data_cleaned.isnull().sum()


    # %% 
    # SAVE CLEANED DATA TO A NEW CSV FILE

    # Save the cleaned data to a new CSV file
    cleaned_data_filename = config.STEAM_STORE_DATA_CLEANED
    save_to_csv(data_cleaned, cleaned_data_filename)
    # %%

    return data_cleaned



def generate_summary_stats(input_data, output_file, quantitative_cols, qualitative_cols):
    # Load original data
    data = pd.read_csv(input_data)

    # Calculate summary statistics for the selected columns
    quantitative_stats = data[quantitative_cols].describe().round(2)
    # print(quantitative_stats, "\n")
    
    # File path and filename to save to
    stats_file_path = output_file
    
    # Save the summary stats to a file
    try:
        with open(stats_file_path, 'w') as f:
            f.write(f"Quantitative Stats:\n{quantitative_stats}\n\n")
            for column in qualitative_cols:
                category_counts = data[column].value_counts()
                most_frequent = category_counts[category_counts == category_counts.max()].index.tolist()
                least_frequent = category_counts[category_counts == category_counts.min()].index.tolist()[:10]  # Limit to 10 for brevity
                f.write(f"Column Name: {column}: \nMost Frequent: {most_frequent}\nLeast Frequent: {least_frequent}\nNumber of Categories: {category_counts.count()}\n\n")
        print(f"\nSummary Stats saved successfully to path: {stats_file_path}")
    except Exception as e:
        print(f"\nError saving Summary Stats to path: {stats_file_path}")
        print(e)



def filter_reviews_by_game(input_data):
    """
    Filters reviews based on game-specific thresholds for weighted_vote_score.

    Args:
        data (pd.DataFrame): The reviews dataset containing 'app_id' and 'weighted_vote_score'.

    Returns:
        pd.DataFrame: Filtered dataset with only "helpful" reviews.

    Notes:
        - The weighted_vote_score is a metric from the Steam API that determines the helpfulness of a review.
        - The threshold for filtering is the 75th percentile (Q3) of the weighted_vote_score for each game.
            - This is based on the distribution of the weighted_vote_score of the overall dataset where Q3/75% is 0.5.
    """
    # print(input_data['weighted_vote_score'].describe())
    # Function to calculate Q3 (75th percentile) for each game based on weighted_vote_score and their app_id
    def calculate_threshold(group):
        q3 = group['weighted_vote_score'].quantile(0.75)  # Q3 for the game
        return group[group['weighted_vote_score'] >= q3]  # Keep reviews above Q3

    # Apply the function to each game's reviews
    filtered_data = (
        input_data.groupby('app_id', group_keys=False, sort=False)
        .apply(calculate_threshold)
        .reset_index(drop=True)  # Reset index after applying
    )
    return filtered_data  



def clean_review_content(text, min_meaningful_words=2):
    """
    Cleans individual review text by:
    - Handling NaNs if any
    - Removing Steam markup tags
    - Removing non-ASCII characters
    - Removing punctuation
    - Removing excessive whitespace
    - Converting text to lowercase
    - Tokenizing text and removing stopwords
    - Ensuring a minimum number of meaningful words are present in the text.

    Args:
        text (str): The review content.
        min_meaningful_words (int): Minimum number of meaningful words required for the text to be considered valid. Defaults to 2.

    Returns:
        str: Cleaned review text.

    Notes:
        - This function is applied to the 'review' column in the reviews dataset.
        - Steam has its own markup tags, which are removed here, info: https://steamcommunity.com/comment/Guide/formattinghelp
    """
    if pd.isnull(text):  # Handle potential NaNs
        return text
    text = re.sub(r'\[.*?\]', '', text)         # Remove Steam markup tags (e.g., [b], [i], [url], [img])
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # Remove excessive whitespace
    text = text.lower()                         # Convert to lowercase

    # Tokenize and remove stopwords
    words = word_tokenize(text)
    meaningful_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Return text if it meets the minimum meaningful word threshold, otherwise None
    if len(meaningful_words) >= min_meaningful_words:
        return text
    return None



def get_wordnet_pos(tag):
    """
    Maps NLTK POS tags to WordNet POS tags for lemmatization.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no match



def lemmatize_review_with_pos(text):
    """
    Lemmatizes text using POS tagging for better accuracy.
    
    Args:
        text (str): The review content.

    Returns:
        str: Lemmatized review text.
    """
    if pd.isnull(text):  # Handle NaNs
        return text
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatize each word with its POS tag
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags if word.isalpha()
    ]
    
    # Join tokens back into a single string
    return ' '.join(lemmatized_tokens)



def inspect_reviews(data_temp, data_cleaned, num_samples=5):
    """
    Randomly samples reviews from the dataset and prints the original and cleaned (lemmatized) reviews.

    Args:
        data_temp (pd.DataFrame): Temporary dataset containing 'original_review' for inspection.
        data_cleaned (pd.DataFrame): Dataset containing the final cleaned and lemmatized reviews.
        num_samples (int): Number of random samples to inspect. Default is 5.

    Notes:
        - Assumes 'original_review' exists in `data_temp` and 'review' exists in `data_cleaned`.
    """
    # Randomly sample reviews
    sample_indices = data_temp.sample(n=num_samples, random_state=47).index
    
    for idx in sample_indices:
        print(f"Original Review (app_id: {data_temp.loc[idx, 'app_id']}):")
        print(data_temp.loc[idx, 'original_review'])
        print("\nCleaned/Lemmatized Review:")
        print(data_cleaned.loc[idx, 'review'])
        print("\n" + "-" * 80 + "\n")



def extract_sentiment_scores(text):
    """
    Extracts sentiment scores (compound, positive, negative, neutral) from text using VADER.

    Args:
        text (str): The text to analyze.

    Returns:
        pd.Series: A series containing the sentiment scores.
    """
    scores = sia.polarity_scores(text)
    return pd.Series({
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    })



def save_to_csv(input_data, output_file):
    """
    Save the cleaned data to a new CSV file.

    Args:
        dainput_datata (pd.DataFrame): The cleaned dataset.
        output_file (str): Path and filename of the cleaned dataset.
    """
    # Save the cleaned data to a new CSV file
    try:
        input_data.to_csv(output_file, index=False)
        print(f"Data saved successfully to path: {output_file}")

    except Exception as e:
        print(f"Error saving data to path: {output_file}")
        print(e)



def save_object_to_pickle(obj, file_path):
    """
    Save an object to a pickle file.

    Args:
        obj (Any): The object to be pickled. This can be any serializable Python object.
        file_path (str): Path to save the pickle file.

    Returns:
        None
    """

    # Save the object
    try: 
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Pickle file (type: {type(obj).__name__}) saved at: {file_path}")
    except Exception as e:
        print(f"Error saving pickle file at: {file_path}. Exception: {e}")
        print(e)




def clean_steam_reviews_data():
    """
    Clean the Steam reviews dataset by:
        - Dropping irrelevant columns that are not needed for analysis.
        - Removing rows with missing review text and playtime.
        - Converting epoch timestamps to datetime format.
        - Adding derived features such as age of review and playtime.
        - Calculating engagement metrics and playtime percentiles.
        - Filtering reviews by weighted vote score.
        - Cleaning the review text and removing empty reviews.
        - Lemmatizing the review text to standardize word forms.
        - Adding sentiment scores to the reviews using sentiment analysis.
        - Saving the cleaned and processed data to a new CSV file.

    Returns:
        - pd.DataFrame: The cleaned and processed reviews dataset.
    """
    config.log_section("CLEAN STEAM REVIEWS DATA")
    # %%
    # LOAD DATA AND INITIAL CHECK OF THE DATASET
    data = pd.read_csv(config.STEAM_REVIEWS_DATA)

    # START DATA CLEANING 
    def transform_reviews_data(input_data):
        """
        Perform initial/basic data cleaning and feature engineering on the reviews dataset.

        Args:
            data (pd.DataFrame): The reviews dataset.

        Returns:
            pd.DataFrame: The cleaned and transformed reviews dataset.
        """

        # START DATA CLEANING BY DROP UNIMPORTANT COLUMNS
        data_cleaned = input_data.copy()  # Create a copy of the original data to clean

        # Drop columns that are not needed for analysis
        irrelevant_cols = [
            'steamid',                      # User ID of each steam reviewer
            'language',                     # Only English reviews anyways
            'timestamp_dev_responded',      # Missing and rarely useful
            'developer_response',           # Only present in a small subset
            'primarily_steam_deck',         # Steam Deck-specific
            'deck_playtime_at_review',      # Not relevant 
            'written_during_early_access',  # Not relevant
        ]
        data_cleaned.drop(columns=irrelevant_cols, inplace=True)

        # Old dataset had these columns, so drop them if they exist
        if 'steam_china_location' in data_cleaned.columns:
            data_cleaned.drop(columns=['steam_china_location'], inplace=True)
        if 'hidden_in_steam_china' in data_cleaned.columns:
            data_cleaned.drop(columns=['hidden_in_steam_china'], inplace=True)

        # DROP ROWS WITH MISSING REVIEW TEXT AND PLAYTIME
        data_cleaned = data_cleaned.dropna(subset=['review'])  # only 479 out of 136511 missing
        data_cleaned = data_cleaned.dropna(subset=['playtime_at_review'])  # only 52 out of 136511 missing


        # CONVERT EPOCH TIMESTAMP TO DATETIME
        data_cleaned['timestamp_created'] = pd.to_datetime(data_cleaned['timestamp_created'], unit='s')
        data_cleaned['timestamp_updated'] = pd.to_datetime(data_cleaned['timestamp_updated'], unit='s')
        data_cleaned['last_played'] = pd.to_datetime(data_cleaned['last_played'], unit='s')


        # FEATURE ENGINEERING: ADD DERIVED FEATURES, SUCH AS AGE OF REVIEW AND PLAYTIME
        now = pd.Timestamp.now()
        data_cleaned['review_age_days'] = (now - data_cleaned['timestamp_created']).dt.days
        data_cleaned['updated_review_age_days'] = (now - data_cleaned['timestamp_updated']).dt.days
        data_cleaned['last_played_days'] = (now - data_cleaned['last_played']).dt.days


        # DROP timestamp_created, timestamp_updated, last_played COLUMNS 
        data_cleaned.drop(columns=['timestamp_created', 'timestamp_updated', 'last_played'])


        # FEATURE ENGINEERING: ENGAGMENT METRICS, PLAYTIME PERCENTILE, USER EXPERIENCE METRICS

        # Engagement Metrics
        # - Engagement Ratio: votes_up / (votes_up + votes_funny) 
        # - It helps differentiate reviews that are purely informative from those that are entertaining
        data_cleaned['engagement_ratio'] = data_cleaned['votes_up'] / (
            data_cleaned['votes_up'] + data_cleaned['votes_funny'] + 1e-5  # Avoid division by zero
        )
        
        # Playtime Percentile
        # - Rank the playtime_forever values and calculate the percentile
        # - This helps understand how much a user has played a game compared to others
        # - High playtime percentiles may suggest users with deeper experience in the game

        # Playtime percentiles for user experience at the time of review. 
        # - Holistic measure, stable metric, but could introduce bias. 
        data_cleaned['playtime_percentile_review'] = data_cleaned.groupby('app_id')['playtime_at_review'].rank(pct=True)
        
        # Playtime percentiles for total engagement
        # - Contextual metric, Sentiment correlation, Temporal consistency, but less comphehensive.
        data_cleaned['playtime_percentile_total'] = data_cleaned.groupby('app_id')['playtime_forever'].rank(pct=True)

        # Check playtime percentiles for a specific game
        # sample_game = data_cleaned[data_cleaned['app_id'] == data_cleaned['app_id'].iloc[0]]
        # print(sample_game[['playtime_at_review', 'playtime_percentile_review', 'playtime_forever', 'playtime_percentile_total']].head())

        # Print head to verify
        # print(data_cleaned[['engagement_ratio', 'playtime_percentile']].head())

        return data_cleaned



    # %%
    # APPLY BASIC TRANSFORMATIONS AND FEATURE ENGINEERING TO THE ENTIRE REVIEWS DATA
    data_cleaned = transform_reviews_data(data)
    # print(f"\nData shape after initial cleaning: {data_cleaned.shape}\n") # (119827, 24)



    # %%
    # CONT. DATA CLEANING: FILTER DATA BY WEIGHTED VOTE SCORE
    data_cleaned = filter_reviews_by_game(data_cleaned) 
    # print(f"Data shape after filtering by weighted vote score: {data_cleaned.shape}\n") # (76279, 24)



    # %%
    # CONT. DATA CLEANING: CLEAN REVIEW TEXT, REMOVE EMPTY REVIEWS, AND KEEP ONLY MEANINGFUL CONTENT
    
    # Clean the 'review' text
    data_cleaned['review'] = data_cleaned['review'].apply(clean_review_content)
    
    data_cleaned = data_cleaned[data_cleaned['review'].str.strip() != ''] # Remove empty strings
    data_cleaned = data_cleaned.dropna(subset=['review'])
    # print(f"Data shape after cleaning review content: {data_cleaned.shape}\n") # (61709, 24)



    # CONT. DATA CLEANING: LEMMATIZE REVIEW

    # Make a copy of the data to keep the original review text
    data_temp = data_cleaned.copy()
    data_temp['original_review'] = data_temp['review']

    # Apply the lemmatization function to 'review'
    print("\nLemmatizing review content... THIS MAY TAKE A WHILE, PLEASE WAIT...\n")
    data_cleaned['review'] = data_cleaned['review'].apply(lemmatize_review_with_pos)
    print("Lemmatization complete!\n")

    # inspect_reviews(data_temp, data_cleaned) # Inspect random samples of the lemmtized reviews
    # print(f"Data shape after lemmatizing review content: {data_cleaned.shape}\n") # (61709, 24)



    # FEATURE ENGINEERING: ADD REVIEW LENGTH AND WORD COUNT
    # - Helps understand the length and complexity of reviews
    data_cleaned['review_length'] = data_cleaned['review'].str.len()
    data_cleaned['word_count'] = data_cleaned['review'].str.split().apply(len)
    # print(f"Data shape after adding review length and word count: {data_cleaned.shape}\n") # (61709, 26)

    # FEATURE ENGINEERING: ADD SENTIMENT SCORES TO THE REVIEWS

    # Apply sentiment analysis to reviews
    print("\nApplying sentiment analysis to reviews... THIS MAY TAKE A WHILE, PLEASE WAIT...\n")
    sentiment_scores = data_cleaned['review'].apply(extract_sentiment_scores)
    data_cleaned = pd.concat([data_cleaned, sentiment_scores], axis=1)
    print("Sentiment analysis complete!\n")

    # Inspect sentiment distribution
    # print(data_cleaned[['compound', 'positive', 'negative', 'neutral']].describe())
    # print(data_cleaned.head())
    # print(f"Data shape after extracting sentiment scores: {data_cleaned.shape}\n") # (61709, 30



    # CHECK FOR EMPTY REVIEWS AFTER CLEANING

    # empty_reviews_count = data_cleaned['review'].isnull().sum() + (data_cleaned['review'] == '').sum()
    # if empty_reviews_count > 0:
    #     print(f"There are {empty_reviews_count} empty reviews remaining. Further cleaning required.")
    # else:
    #     print("No empty reviews found. Dataset is clean.")


    # FINAL CHECK ON THE CLEANED DATA

    # data_cleaned.shape
    # print(f"Shape of data after cleaning: {data_cleaned.shape}\n") 
    # data_cleaned.info()
    # data_cleaned.isnull().sum()

    # SAVE CLEANED DATA TO A NEW CSV FILE
    cleaned_data_filename = config.STEAM_REVIEWS_DATA_CLEANED
    save_to_csv(data_cleaned, cleaned_data_filename)
    # %%

    return data_cleaned



def generate_all_summary_stats():
    config.log_section("GENERATE SUMMARY STATS")
    store_data_quant_cols = ['awards', 'overall_review_%', 'overall_review_count']
    store_data_qual_cols = ['genres', 'developer']
    generate_summary_stats(config.STEAM_STORE_DATA, config.STORE_DATA_SUMMARY_TEXT_FILE, store_data_quant_cols, store_data_qual_cols)

    review_data_quant_cols = ['votes_up', 'votes_funny', 'weighted_vote_score']
    review_data_qual_cols = ['language', 'steam_purchase']
    generate_summary_stats(config.STEAM_REVIEWS_DATA, config.REVIEWS_DATA_SUMMARY_TEXT_FILE, review_data_quant_cols, review_data_qual_cols)



def main():
    # Generate summary statistics for both datasets
    generate_all_summary_stats() # for sake of workflow, just run this function first

    # Clean the Steam Store data
    # clean_steam_store_data()

    # Clean the Steam Reviews data
    # clean_steam_reviews_data()

    # TODO - CLEAN UP THIS FILE TO BE MORE READABLE AND ORGANIZED (FUNCTIONS, COMMENTS, ETC.)

if __name__ == '__main__':
    main()