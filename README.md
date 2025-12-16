# 🧠 Stroke Risk Prediction Interface  
An interactive stroke‑risk prediction system built using **Machine Learning (Logistic Regression)**, **FastAPI backend**, and a **Next.js + Tailwind modern UI**.  
This project predicts the probability of a person getting a stroke based on medical and lifestyle factors using a trained ML model.

---

## 📌 Features

### ✔ Machine Learning  
- Logistic Regression model  
- Data preprocessing with imputation, scaling, one‑hot encoding  
- Model evaluation (Accuracy, ROC‑AUC)  
- Balanced dataset handling  
- Saved trained model (`stroke_pipeline.joblib`)

### ✔ Backend (FastAPI)
- `/predict` API endpoint  
- Input validation  
- JSON prediction response  
- Fast & lightweight API server  
- CORS enabled for frontend access

### ✔ Frontend (Next.js + Tailwind CSS)
- Apple‑style clean modern UI  
- Step‑by‑step interactive questionnaire  
- Animated progress bar  
- Stunning result dashboard  
- Data‑driven insights  
- Risk classification with circular animation  
- Recommended health actions

---

## 📂 Project Structure

stroke-risk-prediction-interface/
│── backend/
│ ├── main.py
│ ├── train_model.py
│ ├── models/
│ │ └── stroke_pipeline.joblib
│ └── data/
│ └── healthcare-dataset-stroke-data.csv
│
│── frontend/
│ ├── app/
│ │ └── page.tsx
│ ├── public/
│ └── ...
│
│── README.md
│── .gitignore



---

## Installation & Setup

### 1.Clone the repository  
```bash
git clone https://github.com/vaibhavpanchal9/stroke-risk-prediction-interface.git
cd stroke-risk-prediction-interface
```

### 2.Backend Setup (FastAPI + ML Model)
Create virtual environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
Train the Machine Learning model
```bash
python train_model.py
```
This loads the dataset → preprocesses → trains → saves model to models/stroke_pipeline.joblib.
Run the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8005
FastAPI docs available at:
👉 http://localhost:8005/docs

### 3.Frontend Setup (Next.js)
  Go to frontend folder
  ```bash
  cd ../frontend
  ```
  Install dependencies
  ```
  npm install
  ```
  Create .env.local file
  ```
  NEXT_PUBLIC_API_URL=http://localhost:8005
  ```
  Start frontend
  ```
  npm run dev
  ```
  The app opens at:
  👉 http://localhost:3000

## Machine Learning Details
Dataset
Public stroke dataset
Includes medical features: age, BMI, average glucose, smoking status, etc.
Model Pipeline
SimpleImputer (median & most_frequent)
StandardScaler
OneHotEncoder
Logistic Regression
Evaluation
Classification report
ROC‑AUC Score
Train/test split (25%)

## Future Enhancements
SHAP Explainability charts
Feature importance plots
Model comparison notebook
Calibration curve
Exportable PDF report
Multi‑model ensemble

## Author
Vaibhav Panchal & Ananya Nikam
