# Global Predictability of Marine Heatwave-Induced Rapid Intensification of Tropical Cyclones

This repository contains Python scripts developed for the analysis and prediction of marine heatwave-induced rapid intensification (RI) of tropical cyclones (TCs) using a global machine learning (ML) approach. The study aims to improve RI forecasts by integrating marine heatwave (MHW) characteristics into predictive models. The results and methods are detailed in the manuscript: *Global Predictability of Marine Heatwave-Induced Rapid Intensification of Tropical Cyclones*.

## Cite

If you use the codes, data, ideas, or results from this project, please cite the following paper:

**Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Foltz, G., and Sen Gupta, A. (2024). Global predictability of marine heatwave-induced rapid intensification of tropical cyclones. Earth’s Future.**

- **Link to the Published Paper:** [Earth’s Future Journal](https://doi.org/10.1029/xxxxxxx)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

To run the code in this repository, you'll need to have the following dependencies installed:

### Python Dependencies
- Python 3.7 or higher
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Joblib
- SHAP
- Basemap

You can install the required Python packages using pip:
```bash
pip install numpy pandas matplotlib xarray scipy seaborn tqdm netCDF4 basemap
```

## Usage

Each script file has a description on top that clearly describes the objectives of that code and expected outputs. Brief explanations of the scripts are as follows:

- **`IAN_mhw_ri (Fig 2).py`**
  - **Description**: Examines the spatiotemporal relationship between MHW events and the RI phase of Hurricane Ian (2022).
  - **Outputs**: A map depicting the track and MHW events associated with Hurricane Ian.

- **`global_RI_map (Fig 4).py`**
  - **Description**: Generates a global map showing the distribution of RI starting points across different basins from 1981 to 2023.
  - **Outputs**: A map highlighting initiation zones of RI events with color-coded wind speed intensity.

- **`decadal_mhw_ri (Fig 5).py`**
  - **Description**: Generates decadal analysis plots for MHW-induced RI events across different basins.
  - **Outputs**: Plots illustrating changes in RI frequency over decades.

- **`xgb_SHAP.py`**
  - **Description**: Applies SHAP (Shapley Additive Explanations) analysis to interpret the XGBoost model results.
  - **Outputs**: Feature importance rankings and visual explanations of model predictions.
  
- **`xgb_mhw_gs.py`**
  - **Description**: Grid search implementation for optimizing XGBoost model parameters specific to MHW data.
  - **Outputs**: Evaluation metrics and visualizations of model performance.

- **`lgb_mhw_gs.py`**
  - **Description**: Optimizes the LightGBM model for MHW-induced RI prediction using grid search.
  - **Outputs**: Best parameters and model performance metrics.

- **`rf_mhw_gs.py`**
  - **Description**: Grid search implementation for optimizing Random Forest model parameters specific to MHW data.
  - **Outputs**: Evaluation metrics and visualizations of model performance.

- **`et_mhw_gs.py`**
  - **Description**: Grid search implementation for optimizing Extra Trees (ET) model parameters specific to MHW data.
  - **Outputs**: Evaluation metrics and visualizations of model performance.
 
- **`en_ml_mhw.py`**
  - **Description**: Focuses on ensemble ML modeling specifically for MHW-induced RI events.
  - **Outputs**: Model evaluation metrics highlighting the impact of MHW characteristics.
  
- **`tc_xgb_gs.py`**
  - **Description**: Runs hyperparameter tuning for the SST-based XGBoost algorithm using grid search to optimize the RI prediction model.
  - **Outputs**: The best parameter set and corresponding performance metrics.

- **`tc_lgb_gs.py`**
  - **Description**: Performs grid search for the SST-based LightGBM algorithm hyperparameters, focusing on optimizing RI predictions with MHW data.
  - **Outputs**: Optimized hyperparameters for LightGBM and evaluation metrics.

- **`tc_rf_gs.py`**
  - **Description**: Runs the SST-based Random Forest algorithm with grid search for hyperparameter optimization.
  - **Outputs**: Performance evaluation and best hyperparameter settings.

- **`tc_et_gs.py`**
  - **Description**: Applies the SST-based Extra Trees (ET) algorithm and tunes its parameters using grid search.
  - **Outputs**: Model performance metrics and optimized settings for ET.

- **`tc_en_ml.py`**
  - **Description**: Constructs and evaluates the SST-based ensemble ML model by combining the outputs of XGBoost, LightGBM, Random Forest, and Extra Trees algorithms.
  - **Outputs**: The ensemble model's prediction accuracy and comparison metrics (POD and FAR).

- **`basin_performance (Fig 7).py`**
  - **Description**: Evaluates the model's performance in predicting RI events across different tropical cyclone basins.
  - **Outputs**: Performance metrics such as Probability of Detection (POD), False Alarm Ratio (FAR), and visualizations of model performance per basin.

- **`RI_lead_time (Sup Fig 1).py`**
  - **Description**: Analyzes the lead time required for TCs to undergo RI under MHW conditions.
  - **Outputs**: Visualization showing lead time distribution for RI events.


## File Structure
```bash
├── scripts/
│   ├── basin_performance (Fig 7).py
│   ├── global_RI_map (Fig 4).py
│   ├── tc_xgb_gs.py
│   ├── tc_lgb_gs.py
│   ├── tc_rf_gs.py
│   ├── tc_et_gs.py
│   ├── tc_en_ml.py
│   ├── xgb_SHAP.py
│   ├── tc_en_mhw.py
│   ├── en_ml_mhw.py
│   ├── rf_mhw_gs.py
│   ├── xgb_mhw_gs.py
│   ├── lgb_mhw_gs.py
│   ├── RI_lead_time (Sup Fig 1).py
│   ├── decadal_mhw_ri (Fig 5).py
│   ├── IAN_mhw_ri (Fig 2).py
├── LICENSE
└── README.md
```

## Data

All data supporting the findings of this study are publicly accessible and available for download. The analysis covers the time period from September 1, 1981, to October 19, 2023. The datasets used include:

1. **Tropical Cyclone Best Track Data**:
   - TC best track data were obtained from the **International Best Track Archive for Climate Stewardship (IBTrACS)** dataset (Knapp et al., 2018). This dataset is freely available in CSV format through the National Centers for Environmental Information (NCEI) website. Access the data here: [IBTrACS CSV Format](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/).

2. **Sea Surface Temperature (SST) Data**:
   - SST data were sourced from the **NOAA Optimum Interpolation Sea Surface Temperature (OISST) version 2.1** dataset (Huang et al., 2021). The dataset is provided in NetCDF format and can be accessed through the ERDDAP data server. Access the data here: [NOAA OISST v2.1](https://www.ncei.noaa.gov/erddap/info/index.html?page=1&itemsPerPage=1000).

These datasets are publicly available, ensuring the transparency and reproducibility of the results presented in this study.

## Results
The main output of this analysis is a set of visualizations and machine learning models examining the impact of marine heatwaves on the rapid intensification of tropical cyclones globally. The results are provided in the cited manuscript.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the Apache License.

## Acknowledgments
This research is supported by the Coastal Hydrology Lab and the Center for Complex Hydrosystems Research at the University of Alabama. Funding was awarded to Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement with The University of Alabama (NA22NWS4320003). Partial support was also provided by NSF award # 2223893.

## Contact
For any questions or inquiries, please contact the project maintainer at [sradfar@ua.edu].
