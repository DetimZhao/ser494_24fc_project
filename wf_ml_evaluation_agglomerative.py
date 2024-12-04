# Evaluates Agglomerative Clustering by computing Silhouette Score and Calinski-Harabasz Index.
# Saves evaluation results to the evaluation/ folder.

import os
import logging

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


def evaluate_agglomerative_clustering(n_clusters=[3,4,5], distance_matrix=None):
    """
    Evaluate the Agglomerative Clustering model and generate metrics for multiple cluster sizes.

    Args:
        n_clusters_list (list): A list of cluster sizes to evaluate.
        distance_matrix (np.ndarray): Optional precomputed distance matrix for clustering.
    """
    results = []
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
        train_labels_path = os.path.join(config.TRAIN_CLUSTERS_CSV.replace(".csv", f"_{n_clusters_value}_clusters.csv"))
        pd.DataFrame(train_labels, columns=["Cluster"]).to_csv(train_labels_path, index=False)
        logging.info(f"Cluster assignments for training set saved to: {train_labels_path}")

        # Save evaluation results
        evaluation_results = {
            "n_clusters": n_clusters_value,
            "silhouette_score": silhouette_train,
            "calinski_harabasz_score": ch_train,
        }
        results.append(evaluation_results)
        config.log_section(f"{n_clusters_value} clusters: Agglomerative Clustering Evaluation")
        print(f"\nEvaluation Results for {n_clusters_value} clusters: {evaluation_results}\n")

    # Save results to summary file
    summary_file_path = os.path.join(config.EVALUATION_FOLDER, "agglomerative_summary.txt")
    with open(summary_file_path, "w") as f:
        for result in results:
            f.write(f"n_clusters={result['n_clusters']}, Silhouette Score={result['silhouette_score']}, "
                    f"CH Index={result['calinski_harabasz_score']}\n")
    logging.info(f"Evaluation summary saved to: {summary_file_path}")

    return results


def plot_dendrogram(train_features, method="ward", title="Dendrogram for Hierarchical Clustering"):
    """
    Plot a dendrogram for the hierarchical clustering.
    """
    config.log_section("Generating dendrogram...")
    logging.info(f"Plotting dendrogram using {method} linkage...")
    
    # Convert to condensed format
    condensed_distance_matrix = squareform(train_features)
    linkage_matrix = linkage(condensed_distance_matrix, method=method)

    plt.figure(figsize=(16, 12))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    save_path = os.path.join(config.EVALUATION_FOLDER, "Hierarchical_Clustering_Dendrogram.png")
    plt.savefig(save_path)
    print(f"Dendrogram saved to: {save_path}")
    plt.close()


def visualize_clusters(features, labels, title="Hierarchical Cluster Visualization"):
    """
    Visualize the clusters in 2D using PCA.
    
    Args:
        features (np.ndarray): The high-dimensional feature array.
        labels (np.ndarray): The cluster labels.
        title (str): The title of the plot.
    """

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Create a DataFrame for seaborn
    df = pd.DataFrame(reduced_features, columns=["PCA1", "PCA2"])
    df["Cluster"] = labels

    # Scatter plot using seaborn
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette='viridis', s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    save_path = os.path.join(config.EVALUATION_FOLDER, "Hierarchical_Cluster_Visualization.png")
    # plt.savefig(save_path)
    # logging.info(f"Cluster visualization plot saved to: {save_path}")
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
        distance += weight * (record1[feature] - record2[feature]) ** 2

    for feature in boolean_features:
        weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
        distance += weight * (record1[feature] != record2[feature])  # Binary distance

    for feature in tfidf_features:
        weight = feature_weights.get(feature, 1.0) if feature_weights else 1.0
        distance += weight * (record1[feature] - record2[feature]) ** 2

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

    # Plot Dendrogram
    plot_dendrogram(distance_matrix, method="ward")

    # Plot elbow 
    plot_elbow(features=distance_matrix, max_clusters=20, save_as='agglomerative')

    # Evaluate Agglomerative Clustering
    model, train_labels, evaluation_results = evaluate_agglomerative_clustering(
        distance_matrix=distance_matrix
    )

    # Visualize clusters in 2D
    visualize_clusters(
        features=train_features,  # Use original feature space for PCA visualization
        labels=train_labels, 
    )





if __name__ == "__main__":
    main()
