import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "data" / "spotify.csv").dropna()

# Same mood labels as train_mood.py
e_med = df["Energy"].median()
v_med = df["Valence"].median()
df["mood"] = np.select(
    [
        (df["Energy"] >= e_med) & (df["Valence"] >= v_med),
        (df["Energy"] >= e_med) & (df["Valence"] < v_med),
        (df["Energy"] < e_med) & (df["Valence"] >= v_med),
    ],
    ["energetic", "intense", "calm"],
    default="sad",
)

# Is Duration stored in milliseconds or seconds?
duration_in_ms = bool(df["Duration"].median() > 1000)
print("Duration looks like:", "milliseconds" if duration_in_ms else "seconds")

features = ["Loudness", "Tempo", "Duration"]
X = df[features]
y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Lite model accuracy:", round(accuracy_score(y_test, pred), 3), "\n")
print(classification_report(y_test, pred))

joblib.dump(
    {"model": model, "features": features, "duration_in_ms": duration_in_ms},
    BASE / "models" / "mood_lite.joblib",
)
print("Saved models/mood_lite.joblib")