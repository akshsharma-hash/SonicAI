import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "data" / "spotify.csv")
df = df.dropna()

# 1) Make mood labels from Energy + Valence (split at the median)
e_med = df["Energy"].median()
v_med = df["Valence"].median()

def make_mood(row):
    if row["Energy"] >= e_med and row["Valence"] >= v_med:
        return "energetic"
    if row["Energy"] >= e_med and row["Valence"] < v_med:
        return "intense"
    if row["Energy"] < e_med and row["Valence"] >= v_med:
        return "calm"
    return "sad"

df["mood"] = df.apply(make_mood, axis=1)
print("Mood counts:")
print(df["mood"].value_counts(), "\n")

# 2) Features (Energy and Valence are left out on purpose)
features = [
    "Danceability", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Tempo", "Key", "Mode", "Duration",
]
X = df[features]
y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Train
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4) Evaluate
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3), "\n")
print(classification_report(y_test, pred))

ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=45)
plt.tight_layout()
(BASE / "models").mkdir(exist_ok=True)
plt.savefig(BASE / "models" / "confusion_matrix.png")

# 5) Feature importance
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature importance:")
print(imp.round(3))

# 6) Save model
joblib.dump({"model": model, "features": features}, BASE / "models" / "mood_model.joblib")
print("\nSaved models/mood_model.joblib and models/confusion_matrix.png")