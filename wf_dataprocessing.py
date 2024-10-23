# Import libraries
import pandas as pd



# %% 
# LOAD DATA AND INITIAL CHECK OF THE DATASET

# Load original data
data = pd.read_csv('data_original/steam-games.csv')

# Check the first few rows of the dataset
data.head()
# Insights:
# So the price is in rupees, and the discount is a negative integer with a percentage symbol. 
# Pandas could be reading some of these as NaN because of their formatting.



# %% 
# CHECK DATA TYPES AND MISSING VALUES  

# Check data types and summary stats
print(f"Shape of data: {data.shape}\n") # (41975, 20)
print(f"Summary of stats:\n {data.describe()}\n")

data.info()
# RangeIndex: 42497 entries, 0 to 42496
# Data columns (total 24 columns)

# Check missing values
data.isnull().sum()



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
# DATA CLEANING CONT: CONVERT RELEASE_DATE TO DATETIME, AGE_RESTRICTED TO BOOLEAN

# Convert 'release_date' to datetime dtype for future analysis
data_cleaned['release_date'] = pd.to_datetime(data_cleaned['release_date'], errors='coerce')
# errors='coerce' tells to set any invalid dates as NaT (Not a Time) instead of giving an error
# - Note: When saving to CSV, it will lose the datetime64 format and will need to be explicitly parsed.
#   - With pandas, parse_dates=['release_date'] can be used when loading to parse the string back into datetime.

# Convert 'age_restricted' to boolean dtype for future analysis, convert 0/1 to True/False
data_cleaned['age_restricted'] = data_cleaned['age_restricted'].apply(lambda x: True if x == 1 else False)

# data_cleaned['release_date'].head() # Check the first few rows of the release_date column for changes
# data_cleaned['age_restricted'].head() # Check the first few rows of the age_restricted column for changes   




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

# Check for patterns in missing values in 'developer' and 'release_date'
missing_developer = data_cleaned[data_cleaned['developer'].isnull()]
missing_release_date = data_cleaned[data_cleaned['release_date'].isnull()]

# View a sample of missing developer and release_date rows to check for patterns
print("##################### MISSING DEVELOPER ######################") # for readability
print(missing_developer[['game_title', 'store_game_description', 'genres', 'categories', 'publisher']].head())
print("\n##################### MISSING RELEASE DATE ######################") # for readability
print(missing_release_date[['game_title', 'store_game_description', 'genres', 'categories', 'developer']].head())

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
# PAUSE DATA CLEANING: CHECK PERCENTAGES OF MISSING VALUES IN KEY COLUMNS: developer, publisher, genres, categories
 
# Find/Calculate the percentage of missing values for developer and publisher 
missing_developer_percentage = data_cleaned['developer'].isnull().mean() * 100
missing_publisher_percentage = data_cleaned['publisher'].isnull().mean() * 100
print("\n############### MISSING PERCENTAGES OF DEVELOPER AND PUBLISHER ###############") # for readability
print(f"Missing Developer: {missing_developer_percentage:.2f}%") # 0.45% of dataset
print(f"Missing Publisher: {missing_publisher_percentage:.2f}%") # 0.50% of dataset

# Find/Calculate the percentage of missing values for genres and categories
missing_genres_percentage = data_cleaned['genres'].isnull().mean() * 100
missing_categories_percentage = data_cleaned['categories'].isnull().mean() * 100
print("\n############### MISSING PERCENTAGES OF GENRES AND CATEGORIES ###############") # for readability
print(f"Missing Genres: {missing_genres_percentage:.2f}%") # 0.20% of dataset
print(f"Missing Categories: {missing_categories_percentage:.2f}%") # 0.11% of dataset

# Find/Calculate the percentage of missing values for release date
missing_release_date_percentage = data_cleaned['release_date'].isnull().mean() * 100
print("\n############### MISSING PERCENT OF RELEASE DATE ###############") # for readability
print(f"Missing Genres: {missing_release_date_percentage:.2f}%") # 0.13% of dataset



# %%
# DATA CLEANING CONT: DROP ROWS WITH MISSING DEVELOPER, PUBLISHER, GENRES, AND CATEGORIES

# Because all the missing values are less than 1%, we can drop these rows without losing much info.
# Drop rows where developer, publisher, genres, or categories are missing
data_cleaned = data_cleaned.dropna(subset=['developer', 'publisher', 'genres', 'categories', 'release_date'])

# Check cleaned dataset's shape to see how many rows remain 
data_cleaned.shape # (42199, 24) where 42199 rows remain after dropping missing values




# %% 
# PAUSE DATA CLEANING: CHECK MISSING VALUES IN CONTENT DESCRIPTOR TAGS

# Check missing values in content descriptor tags
missing_content_descriptor_tags = data_cleaned['content_descriptor_tags'].isnull().mean() * 100

print("\n############### MISSING PERCENT OF CONTENT DESCRIPTOR TAGS ###############") # for readability
print(f"Missing Content Descriptor Tags: {missing_content_descriptor_tags:.2f}%") # 94.39% of dataset

# Assumption and Insights:
# - 94.39% of the data is missing content descriptor tags, so we can drop this column.
# - Upon googling, content descriptors (tags) were introduced in 2023, so it makes sense that most of the data is missing this info.



# %%
# PAUSE DATA CLEANING: CHECK PERCENTAGES OF MISSING VALUES IN RECENT REVIEW COLUMNS

# Find/Calculate the percentage of missing values for recent review columns
# Different naming of variables to avoid confusion with the original data columns
percentage_missing_recent_review = data_cleaned['recent_review'].isnull().mean() * 100
percentage_missing_recent_review_positive_percentage = data_cleaned['recent_positive_review_percentage'].isnull().mean() * 100
percentage_missing_recent_review_count = data_cleaned['recent_review_count'].isnull().mean() * 100

print("\n############### MISSING PERCENTAGES OF REVIEW COLUMNS ###############") # for readability
print(f"Missing Recent Review: {percentage_missing_recent_review:.2f}%") # 87.00% of dataset
print(f"Missing Recent Review Positive Percentage: {percentage_missing_recent_review_positive_percentage:.2f}%") # 87.00% of dataset
print(f"Missing Recent Review Count: {percentage_missing_recent_review_count:.2f}%") # 87.00% of dataset

# Assumption:
# - 87.00% of data is missing recent review data, likely because recent reviews only happen for popular games, recent games, or games with recent updates.
#   - This would make sense, as there are so many games on Steam, and not all of them are popular or have recent updates.
# - Since the data is missing for most of the dataset, we can drop these columns.



# %%
# DATA CLEANING CONT: DROP RECENT REVIEW COLUMNS AND CONTENT DESCRIPTOR TAGS

# Drop the recent review columns due to the high percentage of missing values
data_cleaned = data_cleaned.drop(columns=['content_descriptor_tags', 'recent_review', 'recent_positive_review_percentage', 'recent_review_count'])

# Check the updated data to ensure the columns are dropped
data_cleaned.info()
data_cleaned.isnull().sum() # Check missing values so far



# %%
# PAUSE DATA CLEANING: CHECK MISSING VALUES IN ORIGINAL PRICE AND DISCOUNTED PRICE

# Original price and discounted price have the same number of missing values
# Check again if missing original is related to free games
missing_price_data = data_cleaned[data_cleaned['original_price_INR'].isnull()]
print(missing_price_data[['game_title', 'dlc_available', 'store_game_description', 'discounted_price_INR', 'release_date']].head()) 

# Assumption and Insights:
# - BioShock™, DEATH STRANDING DIRECTOR'S CUT, Call of Duty® are all not free games, and they are from different years.
# - Both original and discounted prices are missing for these games, so the previous assumption that discounted price of 0 (indicating a free game) is not valid here.
# - In the context of the dataset, it is possible these games may not have been available for purchase on the Indian Steam store at the time of scraping.
# - It's also possible the data is missing due to scraping issues, special editions/versions, licensing issues, or other reasons.

# Find/Calculate the percentage of missing values in original and discounted prices after initial cleaning of these columns
print("\n#### MISSING PERCENTAGES OF ORIGINAL AND DISCOUNTED PRICES AFTER INITIAL CLEANING ####") # for readability
missing_original_price_percentage = data_cleaned['original_price_INR'].isnull().mean() * 100
missing_discounted_price_percentage = data_cleaned['discounted_price_INR'].isnull().mean() * 100
print(f"Missing Original Price: {missing_original_price_percentage:.2f}%") # 0.52% of dataset
print(f"Missing Discounted Price: {missing_discounted_price_percentage:.2f}%") # 0.52% of dataset

# Again, because the missing values are less than 1%, we can drop these rows without losing much info.



# %%
# DATA CLEANING CONT: DROP ROWS WITH MISSING ORIGINAL AND DISCOUNTED PRICES

# Drop rows where original and discounted prices are missing
data_cleaned = data_cleaned.dropna(subset=['original_price_INR', 'discounted_price_INR'])

# Check cleaned dataset's shape to see how many rows remain 
data_cleaned.shape # (41979, 20)



# %%
# PAUSE DATA CLEANING: CHECK MISSING VALUES IN OVERALL REVIEW COLUMNS

# Investigate missing overall reviews columns since they have the same number of missing values
missing_review_data = data_cleaned[data_cleaned['overall_review'].isnull()]
print("#### HEAD OF MISSING OVERALL REVIEW DATA ####") # for readability
print(missing_review_data[['game_title', 'store_game_description', 'original_price_INR', 'release_date', 'overall_review_count']].head())
print("\n#### TAIL OF MISSING OVERALL REVIEW DATA ####") # for readability
print(missing_review_data[['game_title', 'store_game_description', 'original_price_INR', 'release_date', 'overall_review_count']].tail()) 

print("\n#### MISSING OVERALL REVIEW PERCENTAGES ####") # for readability 
missing_overall_review_percentage = data_cleaned['overall_review'].isnull().mean() * 100
missing_overall_review_count_percentag = data_cleaned['overall_review_count'].isnull().mean() * 100
print(f"Missing Overall Review: {missing_overall_review_percentage:.2f}%") # 5.57%% of dataset   
print(f"Missing Overall Review Count: {missing_overall_review_count_percentag:.2f}%") # 5.57% of dataset

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

# Check missing values in 'store_game_description' (only 4/four missing values from checking missing values at this point)
missing_description_data = data_cleaned[data_cleaned['store_game_description'].isnull()]
missing_description_data.loc[:, :]

# Assumption and Insights:
# - These four games are seemingly missing completely at random, and there are only 4 vaules so dropping them won't affect the dataset.



# %%
# FINAL DATA CLEANING: DROP ROWS WITH MISSING ABOUT_DESCRIPTION

# Drop rows where 'store_game_description' is missing
data_cleaned = data_cleaned.dropna(subset=['store_game_description'])

# Check cleaned dataset's shape to see how many rows remain 
# data_cleaned.shape # (41975, 20)



# %% 
# CHECK AFTER ALL DATA CLEANING (FINAL CHECK)
# - This cell has been ran multiple times to check the data after each cleaning step
# - This cell is also what drove some of the assumptions and decisions made in the cleaning process
# - This cell is also for checking the data after all cleaning steps are dones

# Final check on the cleaned data
# data_cleaned.shape
print(f"Shape of data after cleaning: {data_cleaned.shape}\n") # (41975, 20)
data_cleaned.info()
data_cleaned.isnull().sum()



# %% 
# SAVE CLEANED DATA TO A NEW CSV FILE

# File path and filename to save to
cleaned_csv_file_name = 'steam-store-data-cleaned.csv'
cleaned_csv_file_path = f'data_processed/{cleaned_csv_file_name}'

# Save the cleaned data to a new CSV file
try:
    data_cleaned.to_csv(cleaned_csv_file_path, index=False)
    print(f"Data saved successfully to path: {cleaned_csv_file_path}")

except Exception as e:
    print(f"Error saving data to path: {cleaned_csv_file_path}")
    print(e)

# %%
