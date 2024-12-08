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
from sklearn.preprocessing import PowerTransformer, StandardScaler
from umap import UMAP
# for umap.plot to work: pip install datashader bokeh holoviews scikit-image colorcet
# not necessary for the code to run, so not included in requirements.txt, but can be added if curious
from umap.plot import points 
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram

import wf_config as config
from wf_ml_training import train_agglomerative_clustering
from wf_ml_evaluation_utils import silhouette_plot


def evaluate_agglomerative_clustering():
    """
    Evaluate Agglomerative Clustering models by computing metrics and generating visualizations.
    """
    # Load training data
    train_features = np.load(config.TRAIN_FEATURES_NPY)

    # Check for precomputed linkage matrix
    linkage_matrix_path = os.path.join(config.DATA_PROCESSED_FOLDER, "linkage_matrix.pkl")
    if os.path.exists(linkage_matrix_path):
        logging.info(f"Loading precomputed linkage matrix from {linkage_matrix_path}...")
        with open(linkage_matrix_path, 'rb') as f:
            linkage_matrix = pickle.load(f)
    else:
        linkage_matrix = None

    # Train models and get results, reusing linkage matrix if available
    n_clusters_list = [3, 4, 5]
    training_results, distance_matrix, linkage_matrix = train_agglomerative_clustering(
        train_features=train_features,
        n_clusters_list=n_clusters_list,
        column_names=config.COL_NAMES_DIST_FUNC_INPUT,  # Pass the column names if defined in config
        feature_weights=None,
        precomputed_linkage_matrix=linkage_matrix  # Pass precomputed linkage matrix
    )

    results = []

    # Evaluate each trained model
    for n_clusters_value, result in training_results.items():
        config.log_section(f"Evaluating Agglomerative Clustering with {n_clusters_value} clusters") 
        model = result["model"]
        train_labels = result["labels"]

        # Compute evaluation metrics
        silhouette_train = silhouette_score(train_features, train_labels)
        ch_train = calinski_harabasz_score(train_features, train_labels)

        # Save evaluation results
        evaluation_results = {
            "n_clusters": n_clusters_value,
            "silhouette_score": silhouette_train,
            "calinski_harabasz_score": ch_train,
        }
        results.append(evaluation_results)
        print(f"\nEvaluation Results for {n_clusters_value} clusters: {evaluation_results}\n")

        # Generate silhouette plot
        silhouette_agglomerative(
            train_features, 
            labels=train_labels, 
            save_path=os.path.join(config.EVALUATION_FOLDER, f'silhouette_plot_agglomerative_{n_clusters_value}.png'),
            n_clusters=n_clusters_value
        )

    # Save results to summary file
    summary_file_path = os.path.join(config.EVALUATION_FOLDER, f"agglomerative_{config.EVALUATION_SUMMARY_TEXT_FILE}")
    with open(summary_file_path, "w") as f:
        for result in results:
            f.write(f"n_clusters={result['n_clusters']}, Silhouette Score={result['silhouette_score']}, "
                    f"CH Index={result['calinski_harabasz_score']}\n")
    logging.info(f"Evaluation summary saved to: {summary_file_path}")

    # Plot dendrogram
    plot_dendrogram(
        linkage_matrix=linkage_matrix,
        title="Dendrogram for Agglomerative Clustering",
        label_interval=10,
        p=10,
        cluster_thresholds=n_clusters_list
    )

    # Visualize clusters in 2D
    visualize_clusters(
        features=train_features,
        labels=training_results[n_clusters_list[0]]["labels"],  # Example visualization for the first cluster size
        title="Hierarchical Cluster Visualization"
    )


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
    # plt.show()
    plt.close()


def visualize_clusters(features, labels, title="Hierarchical Cluster Visualization"):
    """
    Visualize the clusters in 2D using PCA.
    
    Args:
        features (np.ndarray): The high-dimensional feature array.
        labels (np.ndarray): The cluster labels.
        title (str): The title of the plot.
    """
    if features is None or len(features) == 0:
        logging.error("Features array is empty or None. Cannot perform PCA.")
        return

    # Standardize features before PCA
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(standardized_features)

    # Create a DataFrame for seaborn
    df = pd.DataFrame(reduced_features, columns=["PCA1", "PCA2"])
    df["Cluster"] = labels

    # Scatter plot using seaborn
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette=sns.color_palette('viridis', df["Cluster"].nunique()), s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    save_path = os.path.join(config.EVALUATION_FOLDER, "Hierarchical_Cluster_Visualization.png")
    plt.savefig(save_path)
    print(f"Cluster visualization plot saved to: {save_path}")
    plt.close()


def silhouette_agglomerative(features, labels, save_path=None, n_clusters=None):
    """
    Generate silhouette plot for Agglomerative Clustering.

    Args:
        features (np.ndarray): The dataset used for clustering.
        labels (np.ndarray): The cluster labels.
        save_path (str): Path to save the plot (optional).
        n_clusters (int): Number of clusters (optional).
    """
    labels = labels.astype(int)
    title = f"Silhouette Plot for Agglomerative Clustering (n_clusters={n_clusters})"
    silhouette_avg = silhouette_plot(features, labels, title=title, save_path=save_path)
    return silhouette_avg


def visualize_umap_point_cloud(features, labels, custom_distance_matrix=None, output_folder="./umap_plots",
                           n_neighbors_list=[5, 10, 30], min_dist_list=[0.05, 0.1, 0.2], spread_list=[1.0, 1.5],
                           n_components=2):
    """
    Perform UMAP with preprocessing, dimensionality reduction, and parameter optimization.

    Args:
        features (np.ndarray): High-dimensional feature array.
        labels (np.ndarray): Cluster labels.
        custom_distance_matrix (np.ndarray): Precomputed distance matrix (optional).
        output_folder (str): Directory to save plots.
        n_neighbors_list (list): List of n_neighbors values to iterate over.
        min_dist_list (list): List of min_dist values to iterate over.
        spread_list (list): List of spread values to iterate over.
        n_components (int): Number of UMAP components (3 would be for 3D but it would need to be plotted with matplotlib or plotpy).

    """
    # Preprocessing (Optional Scaling)
    transformer = PowerTransformer()  # Use PowerTransformer for skewed data
    scaled_features = transformer.fit_transform(features)
    print("Features scaled using PowerTransformer.")

    # Dimensionality Reduction
    if scaled_features.shape[1] > 100:  # Reduce dimensions only if features are very high-dimensional
        svd = TruncatedSVD(n_components=100, random_state=47)
        reduced_features = svd.fit_transform(scaled_features)
        print(f"Reduced features shape: {reduced_features.shape}")
    else:
        reduced_features = scaled_features

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through UMAP parameters
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            for spread in spread_list:
                print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}, n_components={n_components}")

                # UMAP Fitting
                if custom_distance_matrix is not None:
                    umap = UMAP(metric="precomputed", n_components=n_components, random_state=47)
                    umap.fit(custom_distance_matrix)
                else:
                    umap = UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        spread=spread,
                        n_components=n_components,
                        metric="euclidean",
                        random_state=47,
                        low_memory=True
                    )
                    umap.fit(reduced_features)

                # Plot UMAP Points
                try:
                    points(umap, labels=labels, theme="viridis")
                    plt.title(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}")
                    # Save the plot
                    save_path = os.path.join(
                        output_folder,
                        f"UMAP_nn{n_neighbors}_md{min_dist}_sp{spread}_comp{n_components}.png"
                    )
                    plt.savefig(save_path)
                    print(f"Saved UMAP point cloud plot to: {save_path}")
                    plt.close()
                except Exception as e:
                    print(f"Failed to plot UMAP points for n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}: {e}")

    print("UMAP visualizations completed.")


def main():
    # Evaluate Agglomerative Clustering
    evaluate_agglomerative_clustering()

    # Visualize UMAP point cloud, commented out since it's not that meaningful
    # output_folder = f"{config.EVALUATION_FOLDER}umap_plots/"
    # visualize_umap_point_cloud(
    #     features=train_features,
    #     labels=train_labels,
    #     custom_distance_matrix=distance_matrix,  # Or none too
    #     output_folder=output_folder,
    #     n_neighbors_list=[5, 10, 30],
    #     min_dist_list=[0.05, 0.1, 0.2],
    #     spread_list=[1.0, 1.5],
    #     n_components=2
    # )

if __name__ == "__main__":
    main()
