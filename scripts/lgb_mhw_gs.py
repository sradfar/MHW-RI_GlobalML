# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script uses the LightGBM algorithm for predicting rapid intensification 
# (RI) of tropical cyclones based on marine heatwave (MHW) and storm characteristics. 
# It sets up a pipeline for the classifier, defines a parameter grid for hyperparameter 
# tuning using grid search, and logs the model performance metrics, specifically 
# Probability of Detection (POD) and False Alarm Ratio (FAR).
#
# Outputs:
# - The script logs various parameter sets tested along with corresponding POD 
#   and FAR values to assess the performance of each configuration.
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
import itertools
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import logging

# Logging configuration
logging.basicConfig(filename='lgb_T26PodFar1.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def custom_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    POD = TP / (TP + FN) if (TP + FN) != 0 else 0
    FAR = FP / (FP + TP) if (FP + TP) != 0 else 0
    return POD, FAR

def main():
    logging.info("Starting the main function")

    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 12))
    logging.info(f"Number of CPUs allocated: {n_cpus}")

    # Load and preprocess data
    data = pd.read_csv('../new_ML_input.csv')
    data = data.drop(columns=['ISO_TIME', 'SEASON', 'event_no', 'index_start', 'index_peak', 'index_end',
                              'NAME', 'date_start', 'date_peak', 'date_end', 'LANDFALL',
                              'intensity_cumulative_abs', 'intensity_var_abs', 'gap_end', 'gap_peak',
                              'intensity_var_relThresh', 'intensity_cumulative_relThresh',
                              'intensity_var', 'intensity_cumulative', 'distance_in_km',
                              'intensity_max_relThresh', 'intensity_max', 'duration',
                              'intensity_mean_relThresh', 'gap_start', 'rate_decline'])
    
    X_train = data.iloc[:3626900].drop(columns=['RI'])
    y_train = data.iloc[:3626900]['RI']
    X_test = data.iloc[3626901:].drop(columns=['RI'])
    y_test = data.iloc[3626901:]['RI']
    
    # Pipeline setup
    pipeline = Pipeline([
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])

    # Parameter grid based on supplementary material
    param_grid = {
        'classifier__n_estimators': np.arange(100, 301, 50).tolist(),
        'classifier__colsample_bytree': np.linspace(0.0, 1.0, 3).tolist(),
        'classifier__learning_rate': np.linspace(0.0, 0.3, 4).tolist(),
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__min_child_weight': np.linspace(0.5, 5.0, 3).tolist(),
        'classifier__reg_alpha': np.linspace(0.5, 20.0, 3).tolist(),
        'classifier__reg_lambda': np.linspace(0.5, 20.0, 3).tolist(),
        'classifier__num_leaves': np.arange(3, 9, 1).tolist(),
        'classifier__subsample': np.linspace(0.0, 1.0, 3).tolist(),
        'classifier__scale_pos_weight': np.arange(1.0, 16.1, 0.1).tolist()
    }

    # Iterate through parameter combinations
    keys, values = zip(*param_grid.items())
    for param_set in [dict(zip(keys, v)) for v in itertools.product(*values)]:
        # Set parameters
        best_clf = pipeline.set_params(**param_set)
        best_clf.fit(X_train, y_train)

        # Make predictions
        y_pred = best_clf.predict(X_test)

        # Calculate POD and FAR
        POD, FAR = custom_scorer(y_test, y_pred)

        # Log results
        logging.info(f"Parameter Set: {param_set}")
        logging.info(f"POD: {POD}, FAR: {FAR}")

    logging.info("Completed model training and evaluation")

if __name__ == "__main__":
    main()
