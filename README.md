# Car Price Predictor

A Deep Learning regression model built with PyTorch and Scikit-Learn to predict the price of used vehicles based on their attributes.

## Project Overview
This project takes raw vehicle listing data and trains a Multi-Layer Perceptron (MLP) to estimate car prices. It features a robust Scikit-Learn data preprocessing pipeline to handle missing values, scale numeric features, and encode categorical variables of varying cardinalities before feeding them into a PyTorch neural network.

## Dataset
The model is trained on the **[Craigslist Cars and Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)** from Kaggle. 

**Data Preprocessing Highlights:**
* **Outlier Removal:** Filters vehicle prices to only include realistic listings between $2,000 and $65,000.
* **Missing Data:** Drops columns with over 50% missing data and removes rows missing critical identifiers (`manufacturer` and `model`).
* **Feature Pruning:** Strips out non-predictive or redundant columns like URLs, VINs, images, and descriptions.

## Model Architecture & Pipeline

### 1. Data Transformation Pipeline (`Scikit-Learn`)
* **Numeric Features:** Median Imputation $\rightarrow$ Standard Scaling.
* **Low-Cardinality Categorical:** Most Frequent Imputation $\rightarrow$ One-Hot Encoding.
* **High-Cardinality Categorical (`model`):** Most Frequent Imputation $\rightarrow$ Target Encoding $\rightarrow$ Standard Scaling.

### 2. Neural Network (`PyTorch`)
The core model is an MLP designed for regression tasks:
* **Hidden Layers:** 4 layers `[128, 64, 32, 16]`
* **Regularization:** Batch Normalization (`BatchNorm1d`) and 20% Dropout to prevent overfitting.
* **Activation:** ReLU.
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** Adam (Learning Rate: 0.001).

## How to create the model yourself

### Prerequisites
Ensure you have Python installed along with the following libraries:
* `torch`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`

### Installation & Setup
1. Clone this repository.
2. Download the `vehicles.csv` dataset from the [Kaggle link](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).
3. Create a `data` folder one level above your script directory and place the CSV file inside it so the relative path matches: `../data/vehicles.csv`.
4. Run the script:
   ```bash
   python CarPredictor.py
