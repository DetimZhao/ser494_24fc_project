# Builds machine learning models (e.g., clustering or classification models).
# Saves models to the models/ folder.

import os
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import wf_config as config  


def create_training_testing_datasets():
    """
    Create training and testing datasets for clustering.

    This function reads the combined preprocessed dataset, dynamically aggregates BERT embeddings at the `app_id` level,
    splits the data into training and testing datasets (80-20 split), and standardizes the numeric features.
    """
    config.log_section("Creating Training and Testing Datasets")
    logging.info("Loading the combined dataset for clustering preparation...")
    
    # Load combined dataset (without embeddings)
    combined_data_path = config.COMBINED_CLUSTERING_STEAM_DATASET
    combined_data = pd.read_csv(combined_data_path)
    logging.info(f"Combined dataset loaded with shape: {combined_data.shape}")

    # Load reviews data (to aggregate BERT embeddings)
    reviews_data_path = config.STEAM_REVIEWS_DATA_CLEANED
    reviews_data = pd.read_csv(reviews_data_path)
    logging.info(f"Reviews dataset loaded with shape: {reviews_data.shape}")

    # Load BERT embeddings
    bert_embeddings_path = config.STEAM_REVIEWS_DATA_BERT_EMBEDDINGS_NPY
    bert_embeddings = np.load(bert_embeddings_path)
    logging.info(f"BERT embeddings loaded with shape: {bert_embeddings.shape}")

    # Ensure embeddings align with reviews
    if len(bert_embeddings) != len(reviews_data):
        raise ValueError(
            f"Mismatch between reviews rows ({len(reviews_data)}) and BERT embeddings ({len(bert_embeddings)})."
        )

    # Add BERT embeddings as a column to reviews_data
    reviews_data['bert_embeddings'] = list(bert_embeddings)

    # Aggregate BERT embeddings at the `app_id` level
    aggregated_embeddings = (
        reviews_data.groupby('app_id')['bert_embeddings']
        .apply(lambda x: np.mean(np.stack(x), axis=0))  # Compute the mean embedding for each `app_id`
        .reset_index()
        .rename(columns={'bert_embeddings': 'avg_bert_embeddings'})
    )
    logging.info(f"Aggregated BERT embeddings shape: {aggregated_embeddings.shape}")

    # Merge aggregated embeddings with the combined dataset
    combined_data = pd.merge(combined_data, aggregated_embeddings, on='app_id', how='inner')
    logging.info(f"Final combined dataset shape after merging BERT embeddings: {combined_data.shape}")

    # Expand `avg_bert_embeddings` into individual feature columns
    bert_features = np.stack(combined_data['avg_bert_embeddings'].values)
    # Create a DataFrame for embedding columns
    embedding_columns = pd.DataFrame(
        bert_features,
        columns=[f'bert_embedding_{i}' for i in range(bert_features.shape[1])],
        index=combined_data.index  # Ensure alignment with the combined_data index
    )
    
    # Concatenate the embedding columns with the combined DataFrame
    combined_data = pd.concat([combined_data, embedding_columns], axis=1)
    
    # Optional: Drop the original avg_bert_embeddings column if it’s no longer needed
    if 'avg_bert_embeddings' in combined_data.columns:
        combined_data.drop(columns=['avg_bert_embeddings'], inplace=True)

    # Separate features and IDs
    app_ids = combined_data['app_id']
    features = combined_data.drop(columns=['app_id'], errors='ignore')  # Drop non-feature columns

    # Split into training and testing datasets (80-20 split)
    train_features, test_features, train_ids, test_ids = train_test_split(
        features, app_ids, test_size=0.2, random_state=47, shuffle=True
    )
    logging.info(f"Training set size: {train_features.shape}, Testing set size: {test_features.shape}")
    # 24 (-1 app_id, -1 BERT embeddings) features + 768 BERT embeddings = 790 total features

    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # Save the scaler for reuse
    scaler_path = config.FEATURES_SCALER_PICKLE
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")

    # Save the training and testing datasets
    train_output_path = config.TRAIN_FEATURES_NPY
    test_output_path = config.TEST_FEATURES_NPY

    np.save(train_output_path, train_scaled)
    np.save(test_output_path, test_scaled)
    logging.info(f"Training features saved to {train_output_path}")
    logging.info(f"Testing features saved to {test_output_path}")

    # Save the corresponding IDs for reference
    train_ids_path = config.TRAIN_IDS_CSV
    test_ids_path = config.TEST_IDS_CSV

    train_ids.to_csv(train_ids_path, index=False)
    test_ids.to_csv(test_ids_path, index=False)
    logging.info(f"Training IDs saved to {train_ids_path}")
    logging.info(f"Testing IDs saved to {test_ids_path}")
    
    print("\nTraining and testing datasets created successfully.")


def train_kmeans_model(k, train_features):
    """
    Train a KMeans model with a specified number of clusters (k).

    Args:
        k (int): Number of clusters.
        train_features (np.ndarray): Training dataset features.

    Returns:
        KMeans: Trained KMeans model.
    """
    logging.info(f"Training KMeans model with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=47)
    kmeans.fit(train_features)
    
    # Save the trained model
    model_path = os.path.join(config.MODELS_FOLDER, f"kmeans_model_k{k}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    logging.info(f"KMeans model with k={k} saved to {model_path}")
    
    return kmeans



def plot_elbow(features, max_clusters=10):
    """
    Plot the elbow method for determining the optimal number of clusters.

    Args:
        features (np.ndarray): Training feature array.
        max_clusters (int): Maximum number of clusters to evaluate.
    """
    logging.info(f"Generating Elbow plot for KMeans with up to {max_clusters} clusters...")
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=47, n_init=10, max_iter=300)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # Plot the inertia values
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid()
    plt.savefig(os.path.join(config.EVALUATION_FOLDER, "KMeans Elblow plot.png"))
    # plt.show()


def main():
    create_training_testing_datasets() # Create training and testing datasets
    train_kmeans_model()

if __name__ == "__main__":
    main()
