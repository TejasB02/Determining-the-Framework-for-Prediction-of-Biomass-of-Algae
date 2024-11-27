# Determining the Framework for Prediction of Biomass of Algae

This project aims to predict the biomass of algae using various machine learning frameworks. It includes multiple datasets in `.xlsx` and `.csv` formats and pre-trained models in `.pkl` files. The project provides a comprehensive workflow for data preprocessing, model evaluation, and prediction.

## Features

- **Pre-Trained Models**: Includes several `.pkl` files for different algorithms such as CatBoost, Gradient Boosting, SVR, and XGBoost.
- **Dataset Variety**: Contains multiple datasets in `.xlsx` and `.csv` formats, including resampled and cleaned data.
- **End-to-End Workflow**: Covers data loading, preprocessing, model inference, and result visualization.
- **Performance Evaluation**: Compares the accuracy and metrics of different models for predicting algae biomass.

## Project Structure

. ├── algae_final.ipynb # Main Jupyter Notebook for execution ├── pkl files/ # Directory containing pre-trained models │ ├── best_model_CatBoost.pkl │ ├── catboost_model.pkl │ ├── final_model.keras │ ├── gbm_model.pkl │ ├── model_stacking.pkl │ ├── svr_model.pkl │ └── xgboost_model.pkl ├── dataset/ # Directory containing datasets │ ├── new_data.csv │ ├── raceway1.xlsx │ ├── raceway2.xlsx │ ├── raceway3.xlsx │ └── resampled_data.csv ├── README.md # Project documentation └── requirements.txt # Python dependencies


## Datasets

The `dataset/` directory includes:
- **`new_data.csv`**: A cleaned CSV dataset for prediction.
- **`raceway1.xlsx`, `raceway2.xlsx`, `raceway3.xlsx`**: Raw data collected from different raceways for Spirulina cultivation.
- **`resampled_data.csv`**: Preprocessed and resampled data for model training.

### Dataset Usage
- The `.xlsx` files contain raw data from various sources.
- The `.csv` files (`new_data.csv` and `resampled_data.csv`) are cleaned and resampled versions of the datasets, ready for training and testing.

## Pre-Trained Models

The `pkl files/` directory contains:
- **`best_model_CatBoost.pkl`**: Optimized CatBoost model.
- **`catboost_model.pkl`**: Baseline CatBoost model.
- **`gbm_model.pkl`**: Gradient Boosting model.
- **`xgboost_model.pkl`**: XGBoost model.
- **`svr_model.pkl`**: Support Vector Regression model.
- **`model_stacking.pkl`**: Stacked ensemble of multiple models.
- **`final_model.keras`**: Deep learning model saved in Keras format.

These models can be used directly for predictions or fine-tuned further for specific tasks.

## Prerequisites

- **Python 3.x**
- Required Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - catboost
  - xgboost
  - keras
  - joblib (for loading `.pkl` files)
  - openpyxl (for handling Excel files)

Install the required libraries using:
```bash
pip install -r requirements.txt

How to Run

    Clone the Repository:

git clone https://github.com/yourusername/biomass-prediction-algae.git
cd biomass-prediction-algae

Prepare the Dataset:

    Ensure your datasets (.xlsx and .csv files) are in the dataset/ directory.

Run the Jupyter Notebook:

    jupyter notebook algae_final.ipynb

        Follow the steps in the notebook to preprocess data, load models, and make predictions.

    Model Predictions:
        The notebook provides functionality to load and evaluate pre-trained models from the pkl files/ directory.

Results

    Performance Metrics: The models are evaluated using metrics like RMSE, MAE, and R².
    Comparison: best_model_CatBoost.pkl is identified as the top-performing model based on evaluation.
