# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script evaluates multiple pre-trained models (LightGBM, RandomForest, ExtraTrees, XGBoost) 
# using a weighted combination approach. The script identifies the best weight combinations 
# that maximize the Probability of Detection (POD) while minimizing the False Alarm Ratio (FAR).
#
# Outputs:
# - A log file ('weight_search_new.log') containing the evaluation metrics for each model and 
#   combination of weights.
#
# For a detailed description of the methodologies and further insights, please refer to:
# Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Foltz, G., and Sen Gupta, A. (2024). 
# Global predictability of marine heatwave induced rapid intensification of tropical cyclones. Earth’s Future.
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developers assume no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix
from itertools import product
import logging

# Logging configuration
logging.basicConfig(filename='weight_search_new.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define custom scorer
def custom_scorer(y_true, y_pred):
    """Calculates POD and FAR based on true and predicted values."""
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    POD = TP / (TP + FN) if (TP + FN) != 0 else 0
    FAR = FP / (FP + TP) if (FP + TP) != 0 else 0
    return POD, FAR

def main():
    logging.info("Starting the weight search")

    # Load models
    model_names = ['lgbm_gbdt', 'rf_base', 'et_base', 'xgb']
    models = {name: load(f'../{name}_pipeline.joblib') for name in model_names}
    thresholds = {'lgbm_gbdt': opt1, 'rf_base': opt2, 'et_base': opt3, 'xgb': opt4}

    # Load and preprocess data
    data = pd.read_csv('../new_ML_input.csv')
    drop_columns = ['ISO_TIME', 'SEASON', 'duration', 'intensity_max',
                    'intensity_var', 'intensity_cumulative', 'intensity_mean_relThresh',
                    'intensity_max_relThresh', 'intensity_var_relThresh',
                    'intensity_cumulative_relThresh', 'intensity_var_abs',
                    'intensity_cumulative_abs', 'rate_onset', 'rate_decline',
                    'distance_in_km', 'window_start', 'window_end', 'window_peak', 'srt_MM', 'end_MM', 'peak_MM']
    data = data.drop(columns=drop_columns)

    X_test = data.iloc[3626901:].drop(columns=['RI'])
    y_test = data.iloc[3626901:]['RI']
    
    # Evaluate each model separately with its unique threshold
    for model_name, model in models.items():
        threshold = thresholds[model_name]
        y_pred_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_probs >= threshold).astype(int)
        POD, FAR = custom_scorer(y_test, y_pred)
        logging.info(f"Model: {model_name}, Threshold: {threshold}, POD: {POD}, FAR: {FAR}")

    # Define weight combinations
    weight_options = np.linspace(0, 1, 11)
    weight_combinations = [comb for comb in product(weight_options, repeat=4) if sum(comb) == 1]

    best_score = float('-inf')
    best_weights = None

    for weights in weight_combinations:
        combined_predictions = np.zeros(len(X_test))
        
        # Generate binary predictions using unique thresholds and combine them
        for weight, model_name in zip(weights, model_names):
            model = models[model_name]
            threshold = thresholds[model_name]
            predictions = model.predict_proba(X_test)[:, 1]
            predictions_binary = (predictions >= threshold).astype(int) * weight
            combined_predictions += predictions_binary
            
        # Normalize combined predictions to get the final binary prediction
        combined_predictions_binary = (combined_predictions > max(weights) / 2).astype(int)
        
        # Evaluate performance
        POD, FAR = custom_scorer(y_test, combined_predictions_binary)
        score = POD - FAR
        
        # Log and track best weights
        if score > best_score:
            best_score = score
            best_weights = weights
        logging.info(f"Weight: {weights}, POD: {POD}, FAR: {FAR}, POD - FAR: {score}")

    logging.info(f"Best weights: {best_weights} with POD - FAR: {best_score}")

if __name__ == "__main__":
    main()