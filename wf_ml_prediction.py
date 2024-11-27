# Loads trained models.
# Performs predictions on new or test data.

import os
import pickle
import logging
import numpy as np

import wf_config as config

def load_model(model_path):
    """
    Load the KMeans model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from {model_path}")
    return model

def load_scaler(scaler_path):
    """
    Load the StandardScaler from the specified path.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Train the model first.")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logging.info(f"Scaler loaded from {scaler_path}")
    return scaler

def predict_input(input_data, model_path, scaler_path):
    """
    Predict the cluster for the given input data using a trained KMeans model.

    Args:
        input_data (np.ndarray): The input feature array (single sample or batch).
        model_path (str): Path to the trained KMeans model file.
        scaler_path (str): Path to the StandardScaler pickle file.

    Returns:
        np.ndarray: Predicted cluster(s) for the input data.
    """
    # Load the scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Ensure the model has been trained.")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load the KMeans model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained KMeans model not found at {model_path}. Ensure the model has been trained.")
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)

    # Standardize the input data
    if isinstance(input_data, np.ndarray):
        input_data = scaler.transform(input_data)  # Directly standardize numpy array
    else:
        raise ValueError("Input data must be a NumPy array.")
    # logging.info(f"Input data standardized: {input_data}\n")

    # Predict cluster(s)
    clusters = kmeans.predict(input_data)
    logging.info(f"Predicted cluster(s): {clusters}")
    return clusters

def main():
    pass

if __name__ == "__main__":
    main()
