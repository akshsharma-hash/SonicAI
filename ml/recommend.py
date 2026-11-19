import sys
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "features.csv")

feature_cols = [c for c in df.columns if c != "song"]
X = StandardScaler().fit_transform(df[feature_cols])
sim = cosine_similarity(X)

if len(sys.argv) > 1:
    query = sys.argv[1]
    idx = df.index[df["song"].str.contains(query, case=False, regex=False)]
    if len(idx) == 0:
        print("Song not found. Available songs:")
        print(df["song"].to_string())
        sys.exit()
    i = idx[0]
else:
    i = 0

print("Songs similar to:", df.loc[i, "song"])
scores = sorted(enumerate(sim[i]), key=lambda x: x[1], reverse=True)[1:4]
for j, s in scores:
    print(f"  {df.loc[j, 'song']}  (similarity: {s:.2f})")