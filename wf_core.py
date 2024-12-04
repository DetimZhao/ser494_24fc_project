# Call all necessary functions to carry out workflow.

import wf_config as config
from data_gen import wf_datagen as dg
import wf_dataprocessing as dp
import wf_visualization as vis
import wf_ml_preprocessing as ml_prep
import wf_ml_evaluation as ml_eval

def main():
    
    # Clean inital Steam store data
    dp.clean_steam_store_data()
    
    # Generate Steam reviews data and anything necessary for it
    # dg.main() # Calls a series of other scripts to generate the reviews data
    
    # Clean Steam review data
    # TAKES LIKE 30 SECONDS, comment out if you don't want to run it
    # dp.clean_steam_reviews_data()

    # Generate Summary Stats 
    # dp.generate_all_summary_stats()

    # Do visualizations
    # vis.main() # NOTE: some things need to be updated in the visualization script

    # Compute BERT embeddings and combine datasets 
    ml_prep.main() # Assume that we have the data already (Takes a long time to run otherwise)

    # Call ml evaluation to handle training and testing and model creation
    ml_eval.main() # Assume that we have the data already (Takes a long time to run otherwise)


if __name__ == '__main__':
    main()