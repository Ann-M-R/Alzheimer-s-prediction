# Alzheimer-s-prediction
This project predicts the likelihood of Alzheimer's disease based on patient data using machine learning. Various models were trained, including Logistic Regression, Decision Tree, Random Forest, and XGBoost. XGBoost was selected for deployment because it achieved the best `roc_auc_score` during evaluation. The final model is served through a Flask API and is containerized using Docker for easy deployment.

---

## Problem Statement

Alzheimer's is a progressive neurological disease, and early diagnosis can significantly improve treatment outcomes. The aim of this project is to classify the likelihood of Alzheimer's disease based on features such as age, BMI, medical history, and lifestyle factors.

---

## Solution

1. **Model Training**:
   - Multiple models were trained and evaluated for performance.
   - XGBoost was selected as the best-performing model.
   - The trained XGBoost model was saved as `model.bin`.

2. **Deployment**:
   - A Flask API serves the XGBoost model, allowing real-time predictions.
   - Docker is used to containerize the application for consistent deployment across environments.

---

## Dataset

The dataset used for training the models is available [here](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data).  

---

## How to Set Up and Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

## 2. Install Dependencies

Install Python dependencies using Pipenv:

```bash
pip install pipenv
pipenv install
pipenv shell


### 1. Run the Flask Application Locally

To test without Docker:
```bash
python predict.py

The API will run at http://localhost:9696/predict.





