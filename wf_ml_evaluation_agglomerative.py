# Evaluates Agglomerative Clustering by computing Silhouette Score and Calinski-Harabasz Index.
# Saves evaluation results to the evaluation/ folder.

import os
import logging
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA   
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean, squareform, cdist


import wf_config as config
from wf_ml_training import plot_elbow

# TODO Move the training portion to wf_ml_training.py
def evaluate_agglomerative_clustering(n_clusters=[3,4,5], distance_matrix=None):
    """
    Evaluate the Agglomerative Clustering model and generate metrics for multiple cluster sizes.

    Args:
        n_clusters_list (list): A list of cluster sizes to evaluate.
        distance_matrix (np.ndarray): Optional precomputed distance matrix for clustering.
    """
    results = []

    # Compute linkage matrix using the distance matrix
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    for n_clusters_value in n_clusters:
        config.log_section(f"Evaluating Agglomerative Clustering with {n_clusters_value} clusters")
        logging.info("Loading data for clustering...")

        if distance_matrix is None:
            # Load training features if no distance matrix is provided
            train_features_path = config.TRAIN_FEATURES_NPY
            train_features = np.load(train_features_path)
        else:
            train_features = distance_matrix

        # Perform Agglomerative Clustering
        logging.info(f"Performing Agglomerative Clustering with n_clusters={n_clusters_value}...")
        if distance_matrix is not None:
            model = AgglomerativeClustering(n_clusters=n_clusters_value, metric="precomputed", linkage="average")
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters_value, metric="euclidean", linkage="ward")
        train_labels = model.fit_predict(train_features)

        # Evaluate clustering
        silhouette_train = silhouette_score(train_features, train_labels, metric="precomputed" if distance_matrix is not None else "euclidean")
        ch_train = calinski_harabasz_score(train_features, train_labels)

        logging.info(f"Silhouette Score on Training Set: {silhouette_train}")
        logging.info(f"Calinski-Harabasz Index on Training Set: {ch_train}")

        # Save cluster assignments
        train_labels_path = os.path.join(config.DATA_PROCESSED_FOLDER, f"train_clusters_{n_clusters_value}_agglomerative.pkl")
        with open(train_labels_path, 'wb') as f:
            pickle.dump(train_labels, f)
        logging.info(f"Cluster assignments for training set saved to: {train_labels_path}")

        # Save the trained model
        model_path = os.path.join(config.MODELS_FOLDER, f"agglomerative_n{n_clusters_value}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Agglomerative model with n_clusters={n_clusters_value} saved to {model_path}")

        # Save evaluation results
        evaluation_results = {
            "n_clusters": n_clusters_value,
            "silhouette_score": silhouette_train,
            "calinski_harabasz_score": ch_train,
        }
        results.append(evaluation_results)
        print(f"\nEvaluation Results for {n_clusters_value} clusters: {evaluation_results}\n")

    # Save results to summary file
    summary_file_path = os.path.join(config.EVALUATION_FOLDER, f"agglomerative_{config.EVALUATION_SUMMARY_TEXT_FILE}")
    with open(summary_file_path, "w") as f:
        for result in results:
            f.write(f"n_clusters={result['n_clusters']}, Silhouette Score={result['silhouette_score']}, "
                    f"CH Index={result['calinski_harabasz_score']}\n")
    logging.info(f"Evaluation summary saved to: {summary_file_path}")

    return results, linkage_matrix, train_labels, model


def plot_dendrogram(linkage_matrix, title="Dendrogram for Hierarchical Clustering", label_interval=10, p=10, cluster_thresholds=None):
    """
    Plot a dendrogram using a precomputed linkage matrix.

    Args:
        linkage_matrix (array): The linkage matrix computed during evaluation.
        title (str): Title for the dendrogram plot.
        label_interval (int): Interval for x-axis labels to reduce clutter.
        p (int): Number of clusters (lastp) or levels (level) to display.
        cluster_thresholds (list): List of thresholds to draw vertical dashed lines for clusters.
    """

    # Generate sparse labels for clarity
    labels_list = [f"Sample {i}" if i % label_interval == 0 else "" for i in range(len(linkage_matrix) + 1)]

    # Plot dendrogram   
    plt.figure(figsize=(14, 10))
    dendrogram(
        linkage_matrix,               # Precomputed linkage matrix
        truncate_mode='lastp',        # Truncate modes: 'lastp', 'level'
        p=p,                          # Number of clusters/levels to display
        orientation='right',          # Horizontal layout for readability
        distance_sort="descending",
        labels=labels_list,           # Sparse labels
        show_leaf_counts=True         # Show counts in leaf nodes
    )

    # Add titles and labels
    plt.title(f"{title} (Truncated Clusters p={p})")
    plt.xlabel("Distance")
    plt.ylabel("Sample Index (truncated)")
    plt.yticks(fontsize=7)

    # Draw vertical lines at the distances corresponding to the specified number of clusters
    if cluster_thresholds is not None:
        # palette = sns.color_palette("tab20")
        # colors = palette[6:] # Skip first 3 colors, which are blue, orange, and green
        colors = sns.hls_palette(len(cluster_thresholds), h=.5, s=.9)
        for num_clusters, color in zip(cluster_thresholds, colors):
            # Starting from the last cluster, find the distance that would result in the desired number of clusters
            distance = linkage_matrix[-num_clusters, 2]
            plt.axvline(x=distance, linestyle=':', color=color, label=f'Clusters: {num_clusters}')
        plt.legend(loc='lower right', ncol=2, title='Cluster Thresholds/Number of Merged Clusters')

    # Save and display dendrogram
    save_path = os.path.join(config.EVALUATION_FOLDER, "Hierarchical_Clustering_Dendrogram.png")
    plt.savefig(save_path)
    print(f"Dendrogram saved to: {save_path}")
    plt.show()
    plt.close()


# TODO: This might be broken, check it
def visualize_clusters(features, labels, title="Hierarchical Cluster Visualization"):
    """
    Visualize the clusters in 2D using PCA.
    
    Args:
        features (np.ndarray): The high-dimensional feature array.
        labels (np.ndarray): The cluster labels.
        title (str): The title of the plot.
    """

    # Reduce dimensions using PCA
    if features is None or len(features) == 0:
        logging.error("Features array is empty or None. Cannot perform PCA.")
        return

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Create a DataFrame for seaborn
    df = pd.DataFrame(reduced_features, columns=["PCA1", "PCA2"])
    df["Cluster"] = labels

    # Scatter plot using seaborn
    plt.figure(figsize=(10, 7))
    df["Cluster"] = df["Cluster"].astype('category')
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", style="Cluster", palette=sns.color_palette('viridis', df["Cluster"].nunique()), s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    save_path = os.path.join(config.EVALUATION_FOLDER, "Hierarchical_Cluster_Visualization.png")
    plt.savefig(save_path)
    print(f"Cluster visualization plot saved to: {save_path}")
    plt.close()


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
    # Load the training dataset
    train_features_path = config.TRAIN_FEATURES_NPY
    train_features = np.load(train_features_path)  # Use standardized training features

    # Column names of combined dataset features for custom distance computation
    column_names = [
        'original_price_INR',
        'percentage_of_original_price',
        'discounted_price_INR',
        'dlc_available',
        'age_restricted',
        'awards_count',
        'overall_positive_review_percentage',
        'overall_review_count',
        'overall_review_encoded',
        'mean_compound',
        'mean_positive',
        'mean_negative',
        'mean_neutral',
        'mean_engagement_ratio',
        'mean_playtime_percentile_review',
        'mean_playtime_percentile_total',
        'mean_votes_up',
        'mean_votes_funny',
        'mean_weighted_vote_score',
        'median_playtime_at_review',
        'mean_review_length',
        'mean_word_count',
        'genres_tfidf_access',
        'genres_tfidf_action',
        'genres_tfidf_adventure',
        'genres_tfidf_animation',
        'genres_tfidf_audio',
        'genres_tfidf_casual',
        'genres_tfidf_design',
        'genres_tfidf_development',
        'genres_tfidf_early',
        'genres_tfidf_education',
        'genres_tfidf_free',
        'genres_tfidf_game',
        'genres_tfidf_illustration',
        'genres_tfidf_indie',
        'genres_tfidf_massively',
        'genres_tfidf_modeling',
        'genres_tfidf_movie',
        'genres_tfidf_multiplayer',
        'genres_tfidf_play',
        'genres_tfidf_production',
        'genres_tfidf_publishing',
        'genres_tfidf_racing',
        'genres_tfidf_rpg',
        'genres_tfidf_simulation',
        'genres_tfidf_software',
        'genres_tfidf_sports',
        'genres_tfidf_strategy',
        'genres_tfidf_to',
        'genres_tfidf_training',
        'genres_tfidf_utilities',
        'genres_tfidf_video',
        'genres_tfidf_web',
        'categories_tfidf_about',
        'categories_tfidf_achievements',
        'categories_tfidf_anti',
        'categories_tfidf_app',
        'categories_tfidf_available',
        'categories_tfidf_captions',
        'categories_tfidf_cards',
        'categories_tfidf_cheat',
        'categories_tfidf_cloud',
        'categories_tfidf_co',
        'categories_tfidf_collectibles',
        'categories_tfidf_commentary',
        'categories_tfidf_controller',
        'categories_tfidf_cross',
        'categories_tfidf_editor',
        'categories_tfidf_enabled',
        'categories_tfidf_family',
        'categories_tfidf_features',
        'categories_tfidf_game',
        'categories_tfidf_hdr',
        'categories_tfidf_hl2',
        'categories_tfidf_in',
        'categories_tfidf_includes',
        'categories_tfidf_is',
        'categories_tfidf_lan',
        'categories_tfidf_leaderboards',
        'categories_tfidf_learning',
        'categories_tfidf_level',
        'categories_tfidf_limited',
        'categories_tfidf_mmo',
        'categories_tfidf_mods',
        'categories_tfidf_multiplayer',
        'categories_tfidf_notifications',
        'categories_tfidf_on',
        'categories_tfidf_online',
        'categories_tfidf_only',
        'categories_tfidf_op',
        'categories_tfidf_phone',
        'categories_tfidf_platform',
        'categories_tfidf_play',
        'categories_tfidf_player',
        'categories_tfidf_profile',
        'categories_tfidf_purchases',
        'categories_tfidf_pvp',
        'categories_tfidf_remote',
        'categories_tfidf_require',
        'categories_tfidf_screen',
        'categories_tfidf_sdk',
        'categories_tfidf_shared',
        'categories_tfidf_sharing',
        'categories_tfidf_single',
        'categories_tfidf_source',
        'categories_tfidf_split',
        'categories_tfidf_stats',
        'categories_tfidf_steam',
        'categories_tfidf_steamvr',
        'categories_tfidf_support',
        'categories_tfidf_supported',
        'categories_tfidf_tablet',
        'categories_tfidf_this',
        'categories_tfidf_together',
        'categories_tfidf_tracked',
        'categories_tfidf_trading',
        'categories_tfidf_turn',
        'categories_tfidf_tv',
        'categories_tfidf_valve',
        'categories_tfidf_vr',
        'categories_tfidf_workshop'
    ] 

    # Compute custom distance matrix
    logging.info("Computing custom distance matrix...")
    distance_matrix = compute_distance_matrix(features=train_features, column_names=column_names)

    # Evaluate Agglomerative Clustering
    evaluation_results, linkage_matrix, train_labels, model = evaluate_agglomerative_clustering(
        distance_matrix=distance_matrix
    )
    
    # Plot elbow 
    plot_elbow(features=distance_matrix, title_prefix='Agglomerative')

    # Plot Dendrogram
    plot_dendrogram(linkage_matrix=linkage_matrix, 
                    label_interval=10, 
                    p=75, 
                    cluster_thresholds = [3, 4, 5, 6, 7, 8, 9, 10]
                    )

    # Visualize clusters in 2D
    visualize_clusters(
        features=train_features,  # Use original feature space for PCA visualization
        labels=train_labels, 
    )


if __name__ == "__main__":
    main()
