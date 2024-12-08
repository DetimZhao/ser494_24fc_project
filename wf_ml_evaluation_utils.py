# Contains evaluation functions for clustering models.

import os
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

import wf_config as config

def visualize_clusters(features, labels, title="Cluster Visualization"):
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
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    save_path = os.path.join(config.EVALUATION_FOLDER, "Cluster_Visualization.png")
    plt.savefig(save_path)
    print(f"Saved cluster visualization plot to: {save_path}")
    plt.close()


def visualize_cluster_sizes(labels, title="Cluster Sizes"):
    """
    Visualize the sizes of clusters.
    
    Args:
        labels (np.ndarray): The cluster labels.
        title (str): The title of the plot.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"Cluster": unique_labels, "Count": counts})

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Cluster", y="Count", hue="Cluster", palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Points")
    save_path = os.path.join(config.EVALUATION_FOLDER, "Cluster_Sizes.png")
    plt.savefig(save_path)
    print(f"Saved cluster sizes plot to: {save_path}")
    plt.close()


def visualize_clusters_and_sizes():
    """
    Visualize the clusters using scatter plots and bar plots for sizes.
    """
    config.log_section("Visualizing Clusters")
    logging.info("Loading data for cluster visualization...")

    # Load the features and labels
    train_features = np.load(config.TRAIN_FEATURES_NPY)  
    train_labels = pd.read_csv(config.TRAIN_CLUSTERS_CSV)['Cluster']

    # Validate alignment
    logging.info(f"Train features shape: {train_features.shape}")
    logging.info(f"Train labels shape: {len(train_labels)}")
    if len(train_features) != len(train_labels):
        raise ValueError(f"Mismatch: train_features ({len(train_features)}) and train_labels ({len(train_labels)}) are not aligned.")

    # Ensure features are numeric
    train_features = train_features.astype(float)

    # Visualize clusters (2D scatter plot)
    visualize_clusters(train_features, train_labels.values, title="Cluster Visualization on Training Data")

    # Visualize cluster sizes
    visualize_cluster_sizes(train_labels.values, title="Cluster Sizes on Training Data")


def silhouette_plot(features, labels, title="Silhouette Plot", save_path=None):
    """
    Generate a silhouette plot for clustering results.

    Args:
        features (np.ndarray): The dataset used for clustering.
        labels (np.ndarray): Cluster labels for each sample.
        title (str): Title for the silhouette plot.
        save_path (str): Path to save the plot (optional).

    Returns:
        float: Silhouette Score for the clustering.
    """
    # Compute average silhouette score
    silhouette_avg = silhouette_score(features, labels)
    print(f"Average Silhouette Score: {silhouette_avg:.4f}")

    # Compute silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(features, labels)

    # Number of clusters
    n_clusters = len(np.unique(labels))
    y_lower = 10

    plt.figure(figsize=(10, 7))
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster

        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, alpha=0.7, label=f"Cluster {i}")
        plt.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette Score")
    plt.title(title)
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Silhouette plot saved to: {save_path}")
    
    # plt.show()
    plt.close()

    return silhouette_avg

