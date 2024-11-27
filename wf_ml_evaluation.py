# Evaluates trained models (e.g., metrics, performance).
# Saves evaluation results to the evaluation/ folder.

import os
import pickle
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA

from wf_ml_training import train_kmeans_model  # Import the training function
from wf_ml_prediction import predict_input # Import the prediction function
import wf_config as config


def evaluate_kmeans_model():
    """
    Evaluate the trained KMeans model and generate metrics.
    """
    config.log_section("Evaluating KMeans Model")
    logging.info("Loading trained KMeans model...")

    # Load the trained KMeans model
    model_path = os.path.join(config.MODELS_FOLDER, "kmeans_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained KMeans model not found at {model_path}. Train the model first.")

    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
    logging.info(f"KMeans model loaded from {model_path}")

    # Load the training and testing datasets
    train_features_path = config.TRAIN_FEATURES_NPY
    test_features_path = config.TEST_FEATURES_NPY

    train_features = np.load(train_features_path)
    test_features = np.load(test_features_path)
    logging.info(f"Training features shape: {train_features.shape}, Testing features shape: {test_features.shape}")

    # Evaluate on the training dataset
    train_labels = kmeans.predict(train_features)
    silhouette_train = silhouette_score(train_features, train_labels)
    ch_train = calinski_harabasz_score(train_features, train_labels)
    logging.info(f"Silhouette Score on Training Set: {silhouette_train}")
    logging.info(f"Calinski-Harabasz Index on Training Set: {ch_train}")

    # Evaluate on the testing dataset
    test_labels = kmeans.predict(test_features)
    silhouette_test = silhouette_score(test_features, test_labels)
    ch_test = calinski_harabasz_score(test_features, test_labels)
    logging.info(f"Silhouette Score on Testing Set: {silhouette_test}")
    logging.info(f"Calinski-Harabasz Index on Testing Set: {ch_test}")

    # Save cluster assignments
    train_labels_path = os.path.join(config.TRAIN_CLUSTERS_CSV)
    test_labels_path = os.path.join(config.TEST_CLUSTERS_CSV)

    pd.DataFrame(train_labels, columns=["Cluster"]).to_csv(train_labels_path, index=False)
    pd.DataFrame(test_labels, columns=["Cluster"]).to_csv(test_labels_path, index=False)
    logging.info(f"Cluster assignments for training set saved to: {train_labels_path}")
    logging.info(f"Cluster assignments for testing set saved to: {test_labels_path}")

    # Save the results to a summary file
    evaluation_results = {
        "train_silhouette_score": silhouette_train,
        "test_silhouette_score": silhouette_test,
        "train_calinski_harabasz_score": ch_train,
        "test_calinski_harabasz_score": ch_test,
    }
    config.log_section("Evaluation Results and Saving...")
    print(f"\nEvaluation Results: {evaluation_results}\n")

    summary_file_path = os.path.join(config.EVALUATION_SUMMARY_TEXT_FILE)
    with open(summary_file_path, "w") as f:
        f.write(str(evaluation_results))
    logging.info(f"Evaluation summary saved to: {summary_file_path}")

    return evaluation_results


def evaluate_predictions():
    """
    Evaluate the predictions made by the trained model on a test dataset or specific inputs.
    """
    config.log_section("Evaluating Predictions")
    logging.info("Starting evaluation of predictions...")

    # File paths
    model_path = os.path.join(config.MODELS_FOLDER, "kmeans_model.pkl")
    scaler_path = os.path.join(config.FEATURES_SCALER_PICKLE)

    # Batch input from test set
    test_features_path = config.TEST_FEATURES_NPY
    test_features = np.load(test_features_path)  # Load testing features
    batch_input = test_features[:10, :]  # Take the first 10 samples

    # Predict clusters for the batch input
    batch_clusters = predict_input(batch_input, model_path, scaler_path)
    logging.info(f"Cluster assignments for batch input: {batch_clusters}")
    print(f"Cluster assignments for batch input: {batch_clusters}")

    # Example: Predict the cluster for a specific input (optional)
    single_input = test_features[0:1, :]  # Take the first sample as a single input
    single_cluster = predict_input(single_input, model_path, scaler_path)
    logging.info(f"Cluster assignment for the single test input: {single_cluster[0]}")
    print(f"Cluster assignment for the single test input: {single_cluster[0]}")


def train_and_evaluate_alternative_models():
    """
    Train and evaluate KMeans models with different values of k and save results.
    """
    config.log_section("Training and Evaluating Alternative Models")
    
    # Load training dataset
    train_features_path = config.TRAIN_FEATURES_NPY
    train_features = np.load(train_features_path)

    # Values of k to test
    k_values = [3, 4, 5]  # Updated based on elbow method plot 
    results = []

    for k in k_values:
        # Train KMeans model
        kmeans = train_kmeans_model(k, train_features)

        # Evaluate the model
        train_labels = kmeans.predict(train_features)
        silhouette_train = silhouette_score(train_features, train_labels)
        ch_train = calinski_harabasz_score(train_features, train_labels)
        
        print(f"\nModel with k={k}: Silhouette Score={silhouette_train}, CH Index={ch_train}\n")
        
        # Append results
        results.append({
            "k": k,
            "silhouette_train": silhouette_train,
            "ch_train": ch_train
        })

    # Save results to summary file
    summary_file_path = os.path.join(config.EVALUATION_SUMMARY_TEXT_FILE)
    with open(summary_file_path, "w") as f:
        for result in results:
            f.write(f"k={result['k']}, Silhouette Score={result['silhouette_train']}, "
                    f"CH Index={result['ch_train']}\n")
    logging.info(f"Evaluation summary saved to: {summary_file_path}")


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
    train_labels = pd.read_csv(config.TRAIN_CLUSTERS_CSV)['Cluster'].values  

    # Ensure that train_features and train_labels are numeric
    train_features = train_features.astype(float)
    train_labels = train_labels.astype(int)

    # Visualize clusters (2D scatter plot)
    visualize_clusters(train_features, train_labels, title="Cluster Visualization on Training Data")

    # Visualize cluster sizes
    visualize_cluster_sizes(train_labels, title="Cluster Sizes on Training Data")


def experiment_with_features():
    """
    Experiment with variations of specific features to observe their impact on clustering.
    """
    config.log_section("Experimenting with Feature Impact")
    logging.info("Starting feature impact analysis...")

    # File paths
    model_path = os.path.join(config.MODELS_FOLDER, "kmeans_model.pkl")

    # Load the standardized `.npy` file
    train_features = np.load(config.TRAIN_FEATURES_NPY)

    # Load the `.csv` file to map feature names
    combined_csv_path = config.COMBINED_CLUSTERING_STEAM_DATASET
    combined_data = pd.read_csv(combined_csv_path)

    # Drop columns not used in the model (like app_id) to match the `train_features`
    combined_data = combined_data.drop(columns=['app_id'], errors='ignore')

    # Features to experiment with

    # feature_names = ['overall_positive_review_percentage', 'mean_compound', 'median_playtime_at_review', 'awards_count']
    # indices = [combined_data.columns.tolist().index(feature) for feature in feature_names]
    # print(f"Feature indices: {indices}")

    # Indices of `overall_positive_review_percentage` and `mean_compound` in the original dataset
    selected_features = [6, 10]  # indices of the features to experiment with
    feature_names = ['overall_positive_review_percentage', 'mean_compound']
    # feature_names = ['median_playtime_at_review', 'awards_count']
    feature_values = [
        [20, 50, 80, 100],  # Values to test for `overall_positive_review_percentage`
        [-0.5, 0.0, 0.5, 1.0],  # Values to test for `mean_compound`
    ]
    # feature_values = [
    #     [10, 50, 100, 200],  # Values to test for `median_playtime_at_review`
    #     [0, 1, 5, 10],  # Values to test for `awards_count`
    # ]
    

    # Run the experiment
    results = experiment_with_selected_features(
        model_path=model_path,
        selected_features=selected_features,
        feature_names=feature_names,
        feature_values=feature_values
    )

    # Print the results
    logging.info("Experimentation results:")
    for feature, clusters in results.items():
        logging.info(f"Feature: {feature}, Clusters: {clusters}")
        print(f"Feature: {feature}, Clusters: {clusters}")


def experiment_with_selected_features(model_path, selected_features, feature_names, feature_values):
    """
    Experiment with specific features' variations and their impact on clustering.

    Args:
        model_path (str): Path to the trained KMeans model file.
        selected_features (list): Indices of the features to experiment with (e.g., [10, 15]).
        feature_names (list): Names of the features being varied (for logging purposes).
        feature_values (list of lists): Each sublist contains the values for one feature to vary.

    Returns:
        dict: A dictionary mapping feature variation to cluster assignments.
    """
    # Load the trained KMeans model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained KMeans model not found at {model_path}. Ensure the model has been trained.")
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)

    # Prepare a template for the full feature vector (zeros for simplicity)
    full_feature_vector = np.zeros(kmeans.cluster_centers_.shape[1])

    # Track results
    results = {}

    # Iterate through combinations of selected feature values
    for i, feature in enumerate(selected_features):
        logging.info(f"Experimenting with feature: {feature_names[i]}")
        clusters = []
        for value in feature_values[i]:
            # Update the feature vector for the current feature
            modified_vector = full_feature_vector.copy()
            modified_vector[feature] = value
            cluster = kmeans.predict(modified_vector.reshape(1, -1))
            clusters.append(cluster[0])
        results[feature_names[i]] = clusters

        logging.info(f"Feature '{feature_names[i]}' values: {feature_values[i]}")
        logging.info(f"Predicted clusters: {clusters}")
        print(f"Feature '{feature_names[i]}' values: {feature_values[i]}")
        print(f"Predicted clusters: {clusters}")

    return results


def main():
    # Train the KMeans model (if not already trained)
    # kmeans_model = train_kmeans_model()
    
    # Evaluate the model
    # evaluation_results = evaluate_kmeans_model()
    
    # Evaluate specific predictions
    evaluate_predictions()

    # Train and evaluate alternative models
    train_and_evaluate_alternative_models() # default k values to test are [3, 4, 5]

    # Visualize clusters and cluster sizes
    visualize_clusters_and_sizes()  

    # Experiment with feature variations
    experiment_with_features()

if __name__ == "__main__":
    main()
