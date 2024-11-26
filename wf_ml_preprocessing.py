# Prepares data specifically for ML tasks.
# Generates BERT embeddings.
# My personal device is an m1 mac, so I will be utilizing tensorflow-metal: https://developer.apple.com/metal/tensorflow-plugin/

# Import libraries
import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

import wf_config as config 

save_checkpoint_path = f"{config.DATA_PROCESSED_FOLDER}bert_embeddings_checkpoint.npy"  # Temporary file for checkpointing
print(save_checkpoint_path)

def compute_bert_embeddings(texts, tokenizer, model, max_length=512, batch_size=32, save_path=None):
    """
    Compute BERT embeddings for a list of text reviews.

    Args:
        texts (list[str]): List of text reviews to embed.
        tokenizer (BertTokenizer): Pre-trained BERT tokenizer.
        model (TFBertModel): Pre-trained BERT model.
        max_length (int): Maximum sequence length for BERT inputs. Default is 512.
        batch_size (int): Number of texts processed in each batch. Default is 32.

    Returns:
        np.ndarray: Array of BERT embeddings (shape: [num_texts, embedding_dim]).
    """
    embeddings = []

    # Check if checkpoint exists
    if save_path and os.path.exists(save_path):
        logging.info(f"Resuming from existing embeddings checkpoint: {save_path}")
        embeddings = list(np.load(save_path, allow_pickle=True))
        start_batch = len(embeddings) // batch_size
    else:
        start_batch = 0

    # Calculate total batches
    num_batches = len(texts) // batch_size + int(len(texts) % batch_size != 0)

    for i in range(start_batch, num_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors="tf")
        outputs = model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
        batch_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
        embeddings.append(batch_embeddings)
        logging.info(f"Processed batch {i + 1}/{num_batches}")

        # Save progress after each batch
        if save_path:
            np.save(save_path, np.vstack(embeddings))

    return np.vstack(embeddings)


def main():
    config.log_section("BERT EMBEDDING EXTRACTION")

    # Save embeddings as a .npy file
    embeddings_path = config.STEAM_REVIEWS_DATA_BERT_EMBEDDINGS_NPY

    # Check if the embeddings file already exists
    if os.path.exists(embeddings_path):
        print(f"BERT embeddings already exist at {embeddings_path}. Skipping computation.")
        return  # Exit early

    # Load cleaned reviews dataset
    data_path = config.STEAM_REVIEWS_DATA_CLEANED
    data = pd.read_csv(data_path)
    logging.info(f"Loaded dataset from {data_path} with shape: {data.shape}")

    # Load pre-trained BERT model and tokenizer
    logging.info("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    # Compute BERT embeddings
    logging.info("Computing BERT embeddings...")
    texts = data['review'].tolist()
    save_checkpoint_path = f"{config.DATA_PROCESSED_FOLDER}bert_embeddings_checkpoint.npy"  # Temporary file for checkpointing
    embeddings = compute_bert_embeddings(texts, tokenizer, model, save_path=save_checkpoint_path)

    np.save(embeddings_path, embeddings) # Save embeddings to file
    logging.info(f"BERT embeddings saved to: {embeddings_path}")

    # Clean up checkpoint file after successful run
    if os.path.exists(save_checkpoint_path):
        os.remove(save_checkpoint_path)
        logging.info(f"Removed temporary checkpoint file: {save_checkpoint_path}")

    logging.info("BERT embedding extraction complete.")

if __name__ == "__main__":
    main()