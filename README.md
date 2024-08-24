# Obesity Classification with Machine Learning Pipeline

This Jupyter notebook demonstrates a comprehensive approach to analyzing and classifying obesity levels using machine learning models. The workflow involves data loading, cleaning, exploratory data analysis (EDA), data preprocessing, and model evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries](#libraries)
3. [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Data Preprocessing and Pipeline Creation](#data-preprocessing-and-pipeline-creation)
7. [Model Selection and Evaluation](#model-selection-and-evaluation)
8. [Results](#results)

## Introduction

This notebook aims to analyze a dataset related to obesity and build classification models to predict obesity levels based on various features. It covers the following key steps:

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Data preprocessing for machine learning
5. Model training and evaluation

## Libraries

The following Python libraries are used in this notebook:

* `pandas` and `numpy` for data manipulation
* `seaborn` and `matplotlib` for data visualization
* `scikit-learn` for machine learning and preprocessing

## Data Loading and Initial Exploration

1. **Load the Dataset**: The dataset is loaded from a CSV file named `ObesityDataSet_raw_and_data_sinthetic.csv`.
    
2. **Initial Exploration**:
    * Display the first few rows of the dataframe
    * Check the shape, information, and statistical summary of the dataset

## Data Cleaning and Preprocessing

1. **Rename Columns**: Improve column names for better readability.
    
2. **Data Cleaning**:
    * Replace underscores in text data
    * Convert height from meters to centimeters and round the values
    * Round weight and age values

3. **Replace Numeric Values with Descriptive Categories**:
    * Apply mappings to ordinal features for better interpretability

## Exploratory Data Analysis (EDA)

1. **Visualizations**:
    * Box plots and scatter plots for height and weight by gender
    * Pie charts for obesity distribution by gender
    * Bar plots for eating and exercise habits

## Data Preprocessing and Pipeline Creation

1. **Feature Transformers**:
    * Scale features (Age, Height, Weight) using `StandardScaler`
    * Encode ordinal features using `OrdinalEncoder`
    * Encode non-ordinal features using `OneHotEncoder`
    * Handle missing values with `SimpleImputer`

2. **Pipeline Creation**:
    * Combine preprocessing steps into a `Pipeline` for ease of use

3. **Data Splitting**:
    * Split the data into training and testing sets
    * Encode target variable `Obesity` with `LabelEncoder`

## Model Selection and Evaluation

1. **Model Training**:
    * Train various classifiers, including K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, and Stochastic Gradient Descent

2. **Model Evaluation**:
    * Evaluate models based on accuracy
    * Generate classification reports for performance analysis

## Results

The notebook concludes by displaying the top-performing classifiers based on their accuracy scores. The performance metrics include precision, recall, F1-score, and support for each class in the classification report.

## Usage

1. **Install Dependencies**: Make sure you have the required libraries installed. You can install them using pip:

    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

2. **Run the Notebook**: Open the notebook in Jupyter and execute the cells sequentially to follow the analysis and model evaluation process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
