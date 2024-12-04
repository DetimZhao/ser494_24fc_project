# Evaluates trained models (e.g., metrics, performance).
# Saves evaluation results to the evaluation/ folder.

import os
import pickle
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import wf_ml_training as training
import wf_ml_evaluation_kmeans as kmeans_eval
import wf_ml_evaluation_agglomerative as agglomerative_eval
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



def main():
    # Create training and testing datasets
    training.main()

    # Run evaluation of the KMeans models
    kmeans_eval.main()
    
    # Visualize clusters and cluster sizes
    visualize_clusters_and_sizes()  

    # Run evaluation of the Agglomerative models
    agglomerative_eval.main()


if __name__ == "__main__":
    main()
