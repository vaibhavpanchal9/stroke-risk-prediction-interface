# Stroke Risk Prediction Web App (FastAPI + Next.js + Scikit-learn)

## 1. Overview

This project is an end-to-end **stroke risk prediction system** built on:

- **Python 3.13**
- **Scikit-learn** for model training
- **FastAPI** for the backend API
- **Next.js (React)** for the frontend

The app takes basic patient health information as input (age, BMI, hypertension, etc.) and returns:

- A binary prediction: **high risk / low risk**
- A stroke probability score
- A qualitative risk category (low / moderate / very high)

> ⚠️ **Disclaimer**: This is a demo / academic project, **not** a medical device. It must not be used for real clinical decisions.

---

## 2. Project Structure

```text
stroke_app/
├── backend/
│   └── main.py                 # FastAPI app with /predict endpoint
├── data/
│   └── Training_data/
│       └── healthcare-dataset-stroke-data.csv
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx            # React UI (stroke form)
│   ├── package.json
│   └── ...
├── models/
│   └── stroke_pipeline.joblib  # Trained sklearn Pipeline
├── train_model.py              # Training script
└── README.md
