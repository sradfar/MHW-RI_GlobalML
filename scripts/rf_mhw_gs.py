# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script trains a RandomForestClassifier model for predicting rapid intensification 
# (RI) of tropical cyclones based on marine heatwave (MHW) and storm characteristics. It 
# performs grid search to identify the best hyperparameters and logs model performance metrics 
# such as the Probability of Detection (POD) and False Alarm Ratio (FAR).
#
# Outputs:
# - A log file ('model_RF_gs.log') containing the best parameters found, evaluation metrics, 
#   confusion matrix, and classification report.
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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
import logging

# Logging configuration
logging.basicConfig(filename='model_RF_gs.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def f1_score_class_1(y_true, y_pred):
    """Custom F1 score function for the class labeled as 1."""
    return f1_score(y_true, y_pred, pos_label=1)

def perform_grid_search(X_train, y_train, n_jobs):
    """Performs grid search to find the best parameters for RandomForestClassifier."""
    # Parameter grid based on the table with ranges ensuring the final value is included
    param_grid = {
        'n_estimators': np.arange(100, 301, 100).tolist(), 
        'min_samples_split': np.arange(2, 11, 1).tolist(),  
        'max_depth': np.arange(10, 21, 5).tolist(),        
        'min_samples_leaf': np.arange(1, 6, 1).tolist(),    
        'min_impurity_decrease': np.arange(0.0, 0.3, 0.1).tolist(),  
        'max_features': np.arange(3, 13, 1).tolist()        
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs, 
        scoring=make_scorer(f1_score_class_1)
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def main():
    """Main function to run the RandomForest model."""
    logging.info("Starting the main function")

    # Get the number of CPUs allocated for the task
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 12))
    logging.info(f"Number of CPUs allocated: {n_cpus}")

    # Load the dataset and drop unnecessary columns based on the supplementary material
    data = pd.read_csv('../new_ML_input.csv')
    data = data.drop(columns=['ISO_TIME', 'SEASON', 'duration', 'intensity_max',
                              'intensity_var', 'intensity_cumulative', 'intensity_mean_relThresh',
                              'intensity_max_relThresh', 'intensity_var_relThresh',
                              'intensity_cumulative_relThresh', 'intensity_var_abs',
                              'intensity_cumulative_abs', 'rate_onset', 'rate_decline', 'srt_MM',
                              'distance_in_km', 'window_start', 'window_end', 'window_peak', 'end_MM', 'peak_MM'])

    # Splitting data into training and testing sets
    X_train = data.iloc[:3626900].drop(columns=['RI'])
    y_train = data.iloc[:3626900]['RI']
    X_test = data.iloc[3626901:].drop(columns=['RI'])
    y_test = data.iloc[3626901:]['RI']

    # Apply SMOTE to balance the dataset
    sampler = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

    # Perform grid search to find the best model parameters
    grid_search = perform_grid_search(X_train_resampled, y_train_resampled, n_cpus)
    best_params = grid_search.best_params_
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best F1 score: {grid_search.best_score_}")

    # Log parameter details for all configurations evaluated
    for i in range(len(grid_search.cv_results_['params'])):
        logging.info(f"Params: {grid_search.cv_results_['params'][i]} - Mean Test Score: {grid_search.cv_results_['mean_test_score'][i]}")

    # Train the best model with the best parameters
    best_clf = RandomForestClassifier(**best_params, random_state=42)
    best_clf.fit(X_train_resampled, y_train_resampled)

    # Predict class labels directly without using any threshold
    y_pred = best_clf.predict(X_test)

    # Calculate and log confusion matrix and metrics
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")

    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    POD = TP / (TP + FN) if (TP + FN) != 0 else 0
    FAR = FP / (TP + FP) if (TP + FP) != 0 else 0

    logging.info(f"Probability of Detection (POD): {POD}")
    logging.info(f"False Alarm Ratio (FAR): {FAR}")
    
    # Generate and log the classification report
    report = classification_report(y_test, y_pred)
    logging.info(f"Classification Report:\n{report}")

    logging.info("Completed model training and evaluation")

if __name__ == "__main__":
    main()