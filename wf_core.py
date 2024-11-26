# Call all necessary functions to carry out workflow.

import wf_config as config
import wf_dataprocessing as dp
from data_gen import wf_datagen as dg
import wf_visualization as vis
import wf_ml_preprocessing as ml_prep

def main():
    
    # Clean inital Steam store data
    dp.clean_steam_store_data()
    
    # Generate Steam reviews data and anything necessary for it
    dg.main() # Calls a series of other scripts to generate the reviews data
    
    # Clean Steam review data
    dp.clean_steam_reviews_data()

    # Generate Summary Stats 
    dp.generate_all_summary_stats() 

    # Do visualizations
    vis.main()

    # Compute BERT embeddings
    ml_prep.main() # Assume that we have the data already (Takes a long time to run otherwise)

if __name__ == '__main__':
    main()