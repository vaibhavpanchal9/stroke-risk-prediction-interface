import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


DATA_PATH = "data/Training_data/healthcare-dataset-stroke-data.csv"
MODEL_PATH = "models/stroke_pipeline.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["stroke"], inplace=True)
    return df


def build_preprocessor():
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    categorical_features = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print(f"=== {type(model).__name__} ===")
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    return y_proba


def plot_curves(y_test, y_proba):
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.savefig("models/roc_curve.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.savefig("models/pr_curve.png")
    plt.close()


def shap_explain(model, X_train, preprocessor):
    X_encoded = preprocessor.fit_transform(X_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_encoded)
    shap.summary_plot(shap_values, X_encoded, show=False)
    plt.savefig("models/shap_summary.png")
    plt.close()


def main():
    df = load_data()
    X = df.drop("stroke", axis=1)
    y = df.stroke

    preprocessor = build_preprocessor()

    X_prep = preprocessor.fit_transform(X)

    # SMOTE
    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_resample(X_prep, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42
    )

    models = build_models()

    best_auc = -1
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}")
        y_proba = evaluate(model, X_train, X_test, y_train, y_test)
        if roc_auc_score(y_test, y_proba) > best_auc:
            best_auc = roc_auc_score(y_test, y_proba)
            best_model = model

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model saved at {MODEL_PATH}")

    # visual explainability
    plot_curves(y_test, y_proba)
    shap_explain(best_model, X_train, preprocessor)


if __name__ == "__main__":
    main()
