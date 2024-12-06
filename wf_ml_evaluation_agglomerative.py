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
    logging.info(f"Cluster visualization plot saved to: {save_path}")
    plt.close()


def custom_distance(record1, record2, feature_weights=None):
    """
    Computes a custom distance metric between two records.
    
    Args:
        record1 (pd.Series): The first record (row from a DataFrame).
        record2 (pd.Series): The second record (row from a DataFrame).
        feature_weights (dict): A dictionary mapping feature names to weights (optional).
        
    Returns:
        float: The calculated distance between record1 and record2.
    """
    # Initialize distance
    distance = 0.0

    # Define features for custom handling
    numeric_features = [
        'original_price_INR', 
        'percentage_of_original_price', 
        'discounted_price_INR',
        'dlc_available', 
        'awards_count', 
        'overall_positive_review_percentage', 
        'overall_review_count',
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
        'mean_word_count'
    ]
    # Define the boolean features
    boolean_features = ['age_restricted']
    
    tfidf_features = [col for col in record1.index if col.startswith('genres_tfidf_') or col.startswith('categories_tfidf_')]

    # Compute distances
    for feature in numeric_features:
        weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
        distance += weight * (record1[feature] - record2[feature]) ** 2 # 

    for feature in boolean_features:
        weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
        distance += weight * (record1[feature] != record2[feature])  # Binary distance

    for feature in tfidf_features:
        weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
        distance += weight * (record1[feature] - record2[feature]) ** 2 # 

    # Return the square root to get the Euclidean distance
    return np.sqrt(distance)


def compute_distance_matrix(features, metric='euclidean'):
    """
    Compute a pairwise distance matrix using an optimized approach.

    Args:
        features (np.ndarray): The input feature matrix.
        metric (str): The distance metric to use (default is 'euclidean').

    Returns:
        np.ndarray: A 2D distance matrix.
    """
    config.log_section("Computing Pairwise Distance Matrix")
    if not isinstance(features, np.ndarray):
        features = np.array(features)

    logging.info(f"Computing pairwise distance matrix using {metric} metric...")
    # Use scipy's cdist for efficient pairwise distance computation
    distance_matrix = cdist(features, features, metric=metric)
    logging.info(f"Distance matrix completed. Final shape: {distance_matrix.shape}")
    return distance_matrix


def main():
    # Load the training dataset
    train_features_path = config.TRAIN_FEATURES_NPY
    train_features = np.load(train_features_path)  # Use standardized training features

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(train_features)

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
