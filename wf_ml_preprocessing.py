# Prepares data specifically for ML tasks.
# Generates BERT embeddings, combines them with store data, and prepares for clustering.
# My personal device is an m1 mac, so I will be utilizing tensorflow-metal: https://developer.apple.com/metal/tensorflow-plugin/

# Import libraries
import os
import logging
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder

import wf_config as config 


def compute_bert_embeddings(texts, tokenizer, model, max_length=512, batch_size=32, save_path=None):
    """
    Compute BERT embeddings for a list of text reviews.

    Args:
        texts (list[str]): List of text reviews to embed.
        tokenizer (BertTokenizer): Pre-trained BERT tokenizer.
        model (TFBertModel): Pre-trained BERT model.
        max_length (int): Maximum sequence length for BERT inputs. Default is 512.
        batch_size (int): Number of texts processed in each batch. Default is 32.

    Returns:
        np.ndarray: Array of BERT embeddings (shape: [num_texts, embedding_dim]).
    """
    embeddings = []

    # Check if checkpoint exists
    if save_path and os.path.exists(save_path):
        logging.info(f"Resuming from existing embeddings checkpoint: {save_path}")
        embeddings = list(np.load(save_path, allow_pickle=True))
        start_batch = len(embeddings) // batch_size
    else:
        start_batch = 0

    # Calculate total batches
    num_batches = len(texts) // batch_size + int(len(texts) % batch_size != 0)

    for i in range(start_batch, num_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors="tf")
        outputs = model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
        batch_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
        embeddings.append(batch_embeddings)
        logging.info(f"Processed batch {i + 1}/{num_batches}")

        # Save progress after each batch
        if save_path:
            np.save(save_path, np.vstack(embeddings))

    return np.vstack(embeddings)


def extract_bert_embeddings():
    config.log_section("BERT EMBEDDING EXTRACTION")
    
    # Save embeddings as a .npy file
    embeddings_path = config.STEAM_REVIEWS_DATA_BERT_EMBEDDINGS_NPY

    # Check if the embeddings file already exists
    if os.path.exists(embeddings_path):
        print(f"BERT embeddings already exist at {embeddings_path}. Skipping computation.")
        return  # Exit early

    # Load cleaned reviews dataset
    data_path = config.STEAM_REVIEWS_DATA_CLEANED
    data = pd.read_csv(data_path)
    logging.info(f"Loaded dataset from {data_path} with shape: {data.shape}")

    # Load pre-trained BERT model and tokenizer
    logging.info("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    # Compute BERT embeddings
    logging.info("Computing BERT embeddings...")
    texts = data['review'].tolist()
    save_checkpoint_path = f"{config.DATA_PROCESSED_FOLDER}bert_embeddings_checkpoint.npy"  # Temporary file for checkpointing
    embeddings = compute_bert_embeddings(texts, tokenizer, model, save_path=save_checkpoint_path)

    np.save(embeddings_path, embeddings) # Save embeddings to file
    logging.info(f"BERT embeddings saved to: {embeddings_path}")

    # Clean up checkpoint file after successful run
    if os.path.exists(save_checkpoint_path):
        os.remove(save_checkpoint_path)
        logging.info(f"Removed temporary checkpoint file: {save_checkpoint_path}")

    logging.info("BERT embedding extraction complete.")



def combine_bert_with_steam_data(include_vectorized_features=False, aggregation_method='mean'):
    """
    Combines BERT embeddings, sentiment scores, review features, and store metadata
    for clustering, with optional inclusion of vectorized genres and categories.

    Args:
        include_vectorized_features (bool): Whether to include vectorized genres and categories.
        aggregation_method (str): Aggregation method for numeric features (e.g., "mean", "median", "sum"). Default is "mean".
    """

    def validate_agg_method(method):
        """
        Validate and return aggregation function based on method. 
        """
        method = method.lower()
        if method not in ['mean', 'median', 'sum']:
            raise ValueError(f"Unsupported aggregation method: {method}")
        return method
        
    agg_method = validate_agg_method(aggregation_method)
    logging.info(f"Combining data using aggregation method: {aggregation_method}")

    # Load embeddings, reviews, and store data
    reviews_data = pd.read_csv(config.STEAM_REVIEWS_DATA_CLEANED)
    store_data = pd.read_csv(config.STEAM_STORE_DATA_CLEANED)

    # Use median for skewed features
    playtime_agg_method = 'median'

    # Define the aggregation dictionary dynamically
    aggregation_dict = {
        'compound'                  : agg_method,
        'positive'                  : agg_method,
        'negative'                  : agg_method,
        'neutral'                   : agg_method,
        'engagement_ratio'          : agg_method,
        'playtime_percentile_review': agg_method,
        'playtime_percentile_total' : agg_method,
        'votes_up'                  : agg_method,
        'votes_funny'               : agg_method,
        'weighted_vote_score'       : agg_method,
        'playtime_at_review'        : playtime_agg_method, # Use median for playtime
        'review_length'             : agg_method,
        'word_count'                : agg_method,
    }
    
    # Aggregate reviews by app_id
    aggregated_reviews = reviews_data.groupby('app_id').agg(aggregation_dict).reset_index()

    # Rename columns based on their specific aggregation method
    renamed_columns = {
        key: (f"{playtime_agg_method}_{key}" if key == 'playtime_at_review'
              else f"{agg_method}_{key}")
        for key in aggregation_dict.keys()
    }
    aggregated_reviews.rename(columns=renamed_columns, inplace=True)

    # Apply log transformations to skewed columns
    skewed_columns = ['mean_votes_up', 'mean_votes_funny', 'median_playtime_at_review']
    for col in skewed_columns:
        if col in aggregated_reviews:
            aggregated_reviews[col] = np.log1p(aggregated_reviews[col])  # log1p handles log(0)

    store_data = encode_overall_review(store_data) # Encode overall_review column

    # Drop irrelevant columns from store data
    columns_to_drop = [
        'store_game_description',  # Purely descriptive
        'game_title',              # Identifier but not useful for clustering
        'developer',               # Not useful for clustering
        'publisher',               # Not useful for clustering
        'release_date',            # Not useful for clustering
        'genres',                  # Already vectorized
        'categories',              # Already vectorized
        'overall_review',          # Encoded as 'overall_review_encoded'
    ]
    store_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')  # Drop columns if they exist

    # Merge aggregated reviews with store data
    combined_data = pd.merge(store_data, aggregated_reviews, on='app_id', how='inner')

    # Optional: Add vectorized genres and categories
    if include_vectorized_features:
        genres_vectorized = np.load(os.path.join(config.VECTORIZED_RESULTS_FOLDER, 'genres_vectorized_results.npy'))
        categories_vectorized = np.load(os.path.join(config.VECTORIZED_RESULTS_FOLDER, 'categories_vectorized_results.npy'))
    
        # Load the original store_data to align with app_id
        original_store_data = pd.read_csv(config.STEAM_STORE_DATA_CLEANED)
    
        # Dynamically create feature names for genres
        genres_vectorizer_path = os.path.join(config.VECTORIZERS_FOLDER, 'genres_vectorizer.pkl')
        with open(genres_vectorizer_path, 'rb') as f:
            genres_vectorizer = pickle.load(f)
        genres_feature_names = [f"genres_tfidf_{term}" for term in genres_vectorizer.get_feature_names_out()]
        aligned_genres = pd.DataFrame(
            genres_vectorized,
            index=original_store_data['app_id'],  # Use app_id as the index
            columns=genres_feature_names
        )
    
        # Dynamically create feature names for categories
        categories_vectorizer_path = os.path.join(config.VECTORIZERS_FOLDER, 'categories_vectorizer.pkl')
        with open(categories_vectorizer_path, 'rb') as f:
            categories_vectorizer = pickle.load(f)
        categories_feature_names = [f"categories_tfidf_{term}" for term in categories_vectorizer.get_feature_names_out()]
        aligned_categories = pd.DataFrame(
            categories_vectorized,
            index=original_store_data['app_id'],  # Use app_id as the index
            columns=categories_feature_names
        )
    
        # Merge aligned genres and categories with combined_data
        combined_data = combined_data.set_index('app_id')  # Set app_id as the index
        if all(app_id in aligned_genres.index for app_id in combined_data.index):
            combined_data = combined_data.join(aligned_genres, how='inner')
        else:
            logging.warning("Some app_id values in combined_data are missing from genres_vectorized. Skipping genres.")
    
        if all(app_id in aligned_categories.index for app_id in combined_data.index):
            combined_data = combined_data.join(aligned_categories, how='inner')
        else:
            logging.warning("Some app_id values in combined_data are missing from categories_vectorized. Skipping categories.")
    
        # Reset index to preserve original structure
        combined_data.reset_index(inplace=True)
    
    logging.info(f"Final combined dataset shape: {combined_data.shape}")

    # Save combined data
    combined_data_path = config.COMBINED_CLUSTERING_STEAM_DATASET
    combined_data.to_csv(combined_data_path, index=False)
    logging.info(f"Combined dataset saved to: {combined_data_path}")

    return combined_data


def prepare_clustering_dataset():
    """
    Combines BERT embeddings with store data for clustering.
    """
    config.log_section("PREPARING CLUSTERING DATASET")

    # Combine BERT embeddings with store data
    # combined_data_mean = combine_bert_with_steam_data(aggregation_method='mean') 
    # logging.info(f"Prepared clustering dataset with shape: {combined_data_mean.shape}") 
    combined_data_vectorized_mean = combine_bert_with_steam_data(include_vectorized_features=True, aggregation_method='mean') 
    logging.info(f"Prepared clustering dataset with shape: {combined_data_vectorized_mean.shape}")

    # print(f"combined_data stats\n: {combined_data_mean.drop(columns=['app_id']).describe().round(2)}")
    # print(combined_data_mean.info())


def encode_overall_review(input_data):
    """
    Encode the 'overall_review' column into numeric values using LabelEncoder.

    Args:
        data (pd.DataFrame): The dataset containing the 'overall_review' column.

    Returns:
        pd.DataFrame: The dataset with the encoded 'overall_review' column.
    """
    if 'overall_review' in input_data.columns:
        encoder = LabelEncoder()
        input_data['overall_review_encoded'] = encoder.fit_transform(input_data['overall_review'])
        print(f"\nEncoded 'overall_review' into numeric values: {list(encoder.classes_)}\n")
    else:
        print("\n'overall_review' column not found in the dataset.\n")
    return input_data


def main():
    # Extract BERT Embeddings
    extract_bert_embeddings()

    # Prepare Clustering Dataset
    prepare_clustering_dataset()

    logging.info("BERT extraction and clustering dataset preparation complete.")

if __name__ == "__main__":
    main()