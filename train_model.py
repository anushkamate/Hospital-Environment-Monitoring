"""
Train a demonstration hospital environment classifier from the provided CSV.

Usage:
    python train_model.py "Indoor Air Pollution Data(1).csv"

Labels are generated from configurable rules and are not verified hospital safety labels.
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["NH3", "NO2", "CO", "PM2.5", "Temp", "Pressure", "Humidity", "O3"]
POLLUTANTS = ["NH3", "NO2", "CO", "PM2.5", "O3"]
CLASSES = ["Normal", "Warning", "Critical"]

def normalize_columns(frame):
    df = frame.copy()
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    aliases = {
        "NH3": ["nh3", "ammonia"],
        "NO2": ["no2", "nitrogendioxide"],
        "CO": ["co", "carbonmonoxide"],
        "PM2.5": ["pm2.5", "pm25", "pm2_5", "pm2,5"],
        "Temp": ["temp", "temperature", "temperaturec"],
        "Pressure": ["pressure", "atmosphericpressure"],
        "Humidity": ["humidity", "relativehumidity"],
        "O3": ["o3", "ozone"],
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in normalized:
                rename[normalized[key]] = target
                break
    return df.rename(columns=rename)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Indoor Air Pollution Data(1).csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path, low_memory=False)
    df = normalize_columns(raw)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {FEATURES}")

    df = df[FEATURES].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("Need at least 30 valid rows.")

    warning_limits = df[POLLUTANTS].quantile(0.75)
    critical_limits = df[POLLUTANTS].quantile(0.90)

    def classify(row):
        warn = sum(row[c] > warning_limits[c] for c in POLLUTANTS)
        crit = sum(row[c] > critical_limits[c] for c in POLLUTANTS)
        temp, humidity = row["Temp"], row["Humidity"]
        if crit >= 2 or temp < 18 or temp > 30 or humidity < 30 or humidity > 70:
            return "Critical"
        if warn >= 2 or temp < 20 or temp > 27 or humidity < 35 or humidity > 65:
            return "Warning"
        return "Normal"

    df["label"] = df.apply(classify, axis=1)
    print("Label counts:\n", df["label"].value_counts())
    if df["label"].value_counts().min() < 2:
        raise ValueError("A class has fewer than 2 samples; revise demonstration thresholds.")

    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=120, max_depth=16, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\nHeld-out test accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
    print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred, labels=CLASSES))

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "class_names": CLASSES,
        "warning_limits": warning_limits.to_dict(),
        "critical_limits": critical_limits.to_dict(),
        "label_note": "Rule-generated demonstration labels, not validated hospital safety labels."
    }, "hospital_environment_model.joblib")
    df.to_csv("hospital_environment_labelled_demo.csv", index=False)
    print("\nSaved hospital_environment_model.joblib")
    print("Saved hospital_environment_labelled_demo.csv")

if __name__ == "__main__":
    main()
