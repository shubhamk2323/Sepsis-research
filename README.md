# Sepsis: Early Sepsis Prediction Pipeline

This repository contains **Sepsis**, an advanced End-to-End Machine Learning pipeline. The pipeline trains an XGBoost classifier on massive, real-world ICU patient records to predict life-threatening Sepsis onset hours before it occurs.

## 🚀 Project Overview
Sepsis is a severe clinical condition with high mortality rates if not detected early. This project focuses on analyzing messy, real-world intensive care unit (ICU) data (vital signs, laboratory test results) from over 40,000 patients. 

The original dataset consists of over 40k individual `.psv` files which were merged into a single, massive 1.7+ million row dataset (`combined_sepsis_data.csv`). Due to file size limits, this dataset is not included directly in the repository. We engineered a highly optimized feature extraction and modeling architecture to train our final model on this data.

## 🧠 Key Features & Technical Highlights

* **Z-Score Normalization:** Normalized 40 distinct clinical features using carefully computed population baseline means and standard deviations.
* **Log Scaling:** Applied logarithmic scaling to highly skewed laboratory results (like Bilirubin and Creatinine) to stabilize variance.
* **Delta Calculations:** Calculated the exact hour-to-hour velocity of every vital sign and lab value (`Current Hour - Previous Hour`).
* **Temporal Array Flattening:** Implemented a 6-hour sliding window architecture. Instead of just taking the mean, we shifted the entire clinical array up to `T-5`, flattening 6 hours of patient history into a single 720-dimensional feature vector per hour.
* **XGBoost Classifier:** Trained an advanced XGBoost model tuned with specific hyperparameters to combat severe class imbalance (e.g., `scale_pos_weight: 40`, `max_depth: 4`, `n_estimators: 30`).

## 📊 Evaluation & Metrics
The pipeline includes an Ultra Comprehensive Sepsis Evaluation Report to evaluate performance under extreme class imbalance. Typical realistic results on this dataset include:

- **AUROC (ROC curve area):** ~0.71 (Strong overall discrimination)
- **AUPRC (PR curve area):** ~0.05 (A classic indicator of extreme 2% minority class prevalence)
- **Recall (Sensitivity):** ~0.52 (Successfully catching over 52% of actual sepsis events among severe noise)
- **F2-Score:** ~0.17 (Prioritizing recall over precision)

## 📁 Repository Structure

* `Sepsis_Model_Pipeline.ipynb`: The main End-to-End training pipeline. It handles loading the massive dataset, applying the complex temporal feature engineering, training the XGBoost model, and outputting the final metrics.
* **Note on Data:** The `combined_sepsis_data.csv` is required to run the notebook but is not uploaded due to GitHub size limits. Please follow the instructions below to download and prepare the data.

## ⚙️ How to Run

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy xgboost scikit-learn
   ```
3. **Download the Data:** Download the original dataset from the [PhysioNet 2019 Challenge website](https://physionet.org/content/challenge-2019/1.0.0/).
4. Extract the `.psv` files, combine them into a single file named `combined_sepsis_data.csv`, and place it in the same directory as the notebook.
5. Open the Jupyter Notebook:
   - `Sepsis_Model_Pipeline.ipynb`
6. Click **"Restart & Run All"** to execute the pipeline. The notebook will automatically chunk the data, apply the 720-feature extraction, and evaluate the final XGBoost model.
