# Entry point to evaluate trained models (e.g., metrics, performance).
# Saves evaluation results to the evaluation/ folder.

import wf_ml_training as training
import wf_ml_evaluation_kmeans as kmeans_eval
import wf_ml_evaluation_agglomerative as agglomerative_eval
from wf_ml_evaluation_utils import visualize_clusters_and_sizes


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
