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
git clone https://github.com/Ann-M-R/Alzheimer-s-prediction.git
cd Alzheimer-s-prediction
```

### 2. Install Dependencies

Install Python dependencies using Pipenv:
```bash
pip install pipenv
pipenv install
pipenv shell
```

### 3. Run the Flask Application Locally

To run without Docker:
```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

The API will run at http://localhost:9696/predict.

To test the app:
```bash
python predict-test.py
```

### Using Docker

### 1.Build the Docker Image
```bash
docker build -t alzheimers-diagnosis .
```

### 2. Run the Docker Container
```bash
docker run -p 9696:9696 alzheimers-diagnosis
```
---

## Files in the Repository

- **notebook.ipynb**: Contains data exploration, feature engineering, and training of multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
- **train.py**: Script to train the best model and save it as `model.bin`.
- **predict.py**: Flask application for serving predictions.
- **predict-test.py**: Script to test the Flask API.
- **model.bin**: Saved XGBoost model.
- **alzheimers_disease_data.csv**: Dataset used for training.
- **dockerfile**: Docker configuration file.








