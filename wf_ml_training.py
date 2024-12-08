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
# Kmmeans clustering
from sklearn.cluster import KMeans
# Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

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
    
    # Optional: Drop the original avg_bert_embeddings column if it’s no longer needed (we will in this case)
    if 'avg_bert_embeddings' in combined_data.columns:
        combined_data.drop(columns=['avg_bert_embeddings'], inplace=True)

    # Verify that TF-IDF features exist
    tfidf_columns = [col for col in combined_data.columns if col.startswith(('genres_tfidf_', 'categories_tfidf_'))]
    logging.info(f"TF-IDF features found: {len(tfidf_columns)}")

    # Separate features and IDs
    app_ids = combined_data['app_id']
    features = combined_data.drop(columns=['app_id'], errors='ignore')  # Drop non-feature columns

    # Ensure TF-IDF columns are included in the features
    features = features[tfidf_columns + [col for col in features.columns if col.startswith('bert_embedding_')]]

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


def plot_elbow(features, max_clusters=15, title_prefix=None):
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
    plt.title(f"{title_prefix}: Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid()
    plt.savefig(os.path.join(config.EVALUATION_FOLDER, f"{title_prefix}_Elblow_plot.png"))
    # plt.show()

##############################################################################################################
# KMeans Clustering
##############################################################################################################

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

##############################################################################################################
# Agglomerative Clustering
##############################################################################################################


def train_agglomerative_clustering(train_features, n_clusters_list, column_names=None, feature_weights=None, precomputed_linkage_matrix=None):
    """
    Train Agglomerative Clustering models for multiple cluster sizes and save the results.

    Args:
        train_features (np.ndarray): The input feature matrix.
        n_clusters_list (list): A list of cluster sizes to train models for.
        column_names (list): List of column names for the features (optional).
        feature_weights (dict): A dictionary mapping feature names to weights (optional).
        precomputed_linkage_matrix (np.ndarray): Optional precomputed linkage matrix to reuse.

    Returns:
        dict: A dictionary with models and cluster labels for each `n_clusters`.
    """
    results = {}

    # Check if distance matrix is already computed
    distance_matrix_path = os.path.join(config.DATA_PROCESSED_FOLDER, "distance_matrix.pkl")
    if os.path.exists(distance_matrix_path):
        logging.info(f"Loading precomputed distance matrix from {distance_matrix_path}...")
        with open(distance_matrix_path, 'rb') as f:
            distance_matrix = pickle.load(f)
    else:
        # Compute custom distance matrix
        logging.info("Computing custom distance matrix...")
        distance_matrix = compute_distance_matrix(features=train_features, column_names=column_names, feature_weights=feature_weights)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Symmetrize the matrix
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is zero

        # Save the computed distance matrix
        with open(distance_matrix_path, 'wb') as f:
            pickle.dump(distance_matrix, f)
        logging.info(f"Distance matrix saved to: {distance_matrix_path}")

    # Verify distance matrix validity
    assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix is not symmetric!"
    assert np.all(np.diagonal(distance_matrix) == 0), "Diagonal values are not zero!"

    # Compute or reuse linkage matrix
    if precomputed_linkage_matrix is None:
        linkage_matrix = linkage(squareform(distance_matrix), method="ward")

        # Save the linkage matrix for future use
        linkage_matrix_path = os.path.join(config.DATA_PROCESSED_FOLDER, "linkage_matrix.pkl")
        with open(linkage_matrix_path, 'wb') as f:
            pickle.dump(linkage_matrix, f)
        logging.info(f"Linkage matrix saved to: {linkage_matrix_path}")
    else:
        linkage_matrix = precomputed_linkage_matrix

    for n_clusters in n_clusters_list:
        model_path = os.path.join(config.MODELS_FOLDER, f"agglomerative_n{n_clusters}.pkl")
        train_labels_path = os.path.join(config.DATA_PROCESSED_FOLDER, f"train_clusters_{n_clusters}_agglomerative.pkl")

        # Check if the model and cluster assignments already exist
        if os.path.exists(model_path) and os.path.exists(train_labels_path):
            logging.info(f"Loading precomputed model and labels for {n_clusters} clusters...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(train_labels_path, 'rb') as f:
                cluster_labels = pickle.load(f)
        else:
            logging.info(f"Training Agglomerative Clustering model with {n_clusters} clusters...")

            # Train the model
            model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
            cluster_labels = model.fit_predict(distance_matrix)

            # Save cluster assignments
            with open(train_labels_path, 'wb') as f:
                pickle.dump(cluster_labels, f)
            logging.info(f"Cluster assignments for {n_clusters} clusters saved to: {train_labels_path}")

            # Save the trained model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Agglomerative model with {n_clusters} clusters saved to: {model_path}")

        # Store results
        results[n_clusters] = {"model": model, "labels": cluster_labels}

    return results, distance_matrix, linkage_matrix


def custom_distance(record1, record2, column_names=None, feature_weights=None, start_euclidean_index=124):
    """
    Computes a custom distance metric between two records.

    Args:
        record1 (np.ndarray): The first record (row from the dataset).
        record2 (np.ndarray): The second record (row from the dataset).
        column_names (list): List of column names for the features (optional).
        feature_weights (dict): A dictionary mapping feature names to weights for custom distances (optional).
        start_euclidean_index (int): The index at which Euclidean-only features (e.g., embeddings) start.

    Notes:
        - 124 is the default index for embeddings since that is based on the shape of the data.  

    Returns:
        float: The calculated distance between record1 and record2.
    """
    # Initialize distance
    distance = 0.0

    # Custom handling for named features (if provided)
    if column_names:
        numeric_features = [
            'original_price_INR', 'percentage_of_original_price', 'discounted_price_INR',
            'dlc_available', 'awards_count', 'overall_positive_review_percentage',
            'overall_review_count', 'mean_compound', 'mean_positive', 'mean_negative',
            'mean_neutral', 'mean_engagement_ratio', 'mean_playtime_percentile_review',
            'mean_playtime_percentile_total', 'mean_votes_up', 'mean_votes_funny',
            'mean_weighted_vote_score', 'median_playtime_at_review', 'mean_review_length',
            'mean_word_count'
        ]
        boolean_features = ['age_restricted']

        # Process numeric and boolean features
        for feature in numeric_features:
            index = column_names.index(feature)
            weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
            distance += weight * (record1[index] - record2[index]) ** 2

        for feature in boolean_features:
            index = column_names.index(feature)
            weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
            distance += weight * (record1[index] != record2[index])  # Binary distance

    # Compute Euclidean distance for embeddings (from start_euclidean_index onward)
    distance += np.sum((record1[start_euclidean_index:] - record2[start_euclidean_index:]) ** 2)

    return np.sqrt(distance)


def compute_distance_matrix(features, feature_weights=None, column_names=None):
    """
    Compute a pairwise distance matrix using a custom distance function.

    Args:
        features (np.ndarray): The input feature matrix.
        feature_weights (dict): A dictionary mapping feature names to weights for custom distances (optional).
        column_names (list): List of column names for the features (optional). If None, assume embeddings start dynamically.

    Returns:
        np.ndarray: A 2D distance matrix.
    """
    config.log_section("Computing Pairwise Distance Matrix")
    n_samples, n_features = features.shape

    # Dynamically set start index for embeddings if column names are missing
    if column_names:
        start_euclidean_index = len(column_names)  # Embeddings start where named features end
    else:
        start_euclidean_index = 124  # Default if no column names are provided

    logging.info(f"Start index for Euclidean distance features: {start_euclidean_index}")

    # Initialize distance matrix
    distance_matrix = np.empty((n_samples, n_samples))

    # Compute pairwise distances
    logging.info("Computing custom distances...")
    for i in range(n_samples):
        for j in range(i + 1, n_samples):  # Upper triangle computation
            distance_matrix[i, j] = custom_distance(
                record1=features[i],
                record2=features[j],
                column_names=column_names,
                feature_weights=feature_weights,
                start_euclidean_index=start_euclidean_index,
            )
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric

    logging.info(f"Distance matrix completed. Shape: {distance_matrix.shape}")
    return distance_matrix


def main():
    create_training_testing_datasets() # Create training and testing datasets

if __name__ == "__main__":
    main()
