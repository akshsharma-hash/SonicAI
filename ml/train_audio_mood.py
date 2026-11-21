import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

BASE = Path(__file__).parent
feat = pd.read_csv(BASE / "features.csv")
lab = pd.read_csv(BASE / "labels.csv")
df = feat.merge(lab, on="song")
df["mood"] = df["mood"].astype(str).str.strip().str.lower()
df = df[df["mood"].isin(["energetic", "intense", "calm", "sad"])]

print("Labeled songs per mood:")
print(df["mood"].value_counts(), "\n")

X = df.drop(columns=["song", "mood"])
y = df["mood"]

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))

k = min(5, int(y.value_counts().min()))
if k >= 2:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Cross-validated accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
else:
    print("Not enough labeled songs per mood for cross-validation.")

model.fit(X, y)
joblib.dump({"model": model, "features": list(X.columns)}, BASE / "models" / "mood_audio.joblib")
print("Saved models/mood_audio.joblib")