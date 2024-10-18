# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script trains an XGBoost model for predicting rapid intensification 
# (RI) of tropical cyclones based on marine heatwave (MHW) and storm characteristics. It 
# performs grid search to identify the best hyperparameters and logs model 
# performance metrics such as the Probability of Detection (POD) and False Alarm Ratio (FAR).
#
# Outputs:
# - A log file ('xgb_trainingPodFar.log') containing the best parameters found, 
#   evaluation metrics, confusion matrix, and classification report.
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
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import logging

# Logging configuration
logging.basicConfig(filename='xgb_trainingPodFar.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def custom_scorer(y_true, y_pred):
    """Custom scorer to calculate POD and FAR."""
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    POD = TP / (TP + FN) if (TP + FN) != 0 else 0
    FAR = FP / (FP + TP) if (FP + TP) != 0 else 0
    return POD, FAR

def main():
    logging.info("Starting the main function")

    # Get the number of CPUs allocated for the task
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 12))
    logging.info(f"Number of CPUs allocated: {n_cpus}")

    # Load and preprocess the dataset
    data = pd.read_csv('../new_ML_input.csv')
    data = data.drop(columns=['ISO_TIME', 'SEASON', 'duration', 'intensity_max',
                              'intensity_var', 'intensity_cumulative', 'intensity_mean_relThresh',
                              'intensity_max_relThresh', 'intensity_var_relThresh',
                              'intensity_cumulative_relThresh', 'intensity_var_abs',
                              'intensity_cumulative_abs', 'rate_onset', 'rate_decline', 'srt_MM',
                              'distance_in_km', 'window_start', 'window_end', 'window_peak', 'end_MM', 'peak_MM'])

    # Splitting the data into training and testing sets
    X_train = data.iloc[:3626900].drop(columns=['RI'])
    y_train = data.iloc[:3626900]['RI']
    X_test = data.iloc[3626901:].drop(columns=['RI'])
    y_test = data.iloc[3626901:]['RI']
    
    # Apply SMOTE to oversample the minority class
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    pipeline = Pipeline([
        ('classifier', XGBClassifier(random_state=42, n_jobs=n_cpus))
    ])

    # Parameter grid based on the table, ensuring each range includes the final value
    param_grid = {
        'classifier__n_estimators': np.arange(100, 301, 50).tolist(),        
        'classifier__colsample_bytree': np.linspace(0.0, 1.0, 4).tolist(),   
        'classifier__gamma': [0.0, 1.0],                                  
        'classifier__learning_rate': np.linspace(0.0, 0.3, 4).tolist(),      
        'classifier__max_depth': np.arange(3, 13, 3).tolist(),              
        'classifier__min_child_weight': [0.5, 2.5, 5],                     
        'classifier__reg_alpha': np.linspace(0.5, 20.0, 3).tolist(),      
        'classifier__reg_lambda': [0.5, 10.25, 20.0],                       
        'classifier__subsample': np.linspace(0.0, 1.0, 3).tolist(),         
        'classifier__scale_pos_weight': np.linspace(1, 16, 4).tolist()      
    }

    # Create a product of all parameter combinations
    keys, values = zip(*param_grid.items())
    for param_set in [dict(zip(keys, v)) for v in itertools.product(*values)]:
        # Set parameters
        best_clf = pipeline.set_params(**param_set)
        best_clf.fit(X_train_resampled, y_train_resampled)

        # Predict the class labels directly without using any threshold
        y_pred = best_clf.predict(X_test)

        # Calculate POD and FAR
        POD, FAR = custom_scorer(y_test, y_pred)

        # Log results
        logging.info(f"Parameter Set: {param_set}")
        logging.info(f"POD: {POD}, FAR: {FAR}")

    logging.info("Completed model training and evaluation")

if __name__ == "__main__":
    main()