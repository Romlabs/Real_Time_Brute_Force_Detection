# -*- coding: utf-8 -*-
"""
One-time training script.
Run this in Colab / locally to produce model.pkl, scaler.pkl,
feature_names.json, and encoders.json.
Then commit those 4 artifacts to the repo.
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
import kagglehub, os
path = kagglehub.dataset_download(
    "dnkumars/cybersecurity-intrusion-detection-dataset"
)
csv_file = next(f for f in os.listdir(path) if f.endswith(".csv"))
df = pd.read_csv(os.path.join(path, csv_file))

# ---------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------
df = df.drop(columns=["session_id"])
df["encryption_used"] = df["encryption_used"].fillna("None")

# Save label encoders so we can reproduce them at inference time
encoders = {}
for col in ["protocol_type", "browser_type", "encryption_used"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}

df = df.drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------
X = df.drop("attack_detected", axis=1)
y = df["attack_detected"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 4. Scale
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 5. Train (class_weight='balanced')
# ---------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
model.fit(X_train_s, y_train)

print(classification_report(y_test, model.predict(X_test_s), digits=4))

# ---------------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------------
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)
with open("encoders.json", "w") as f:
    json.dump(encoders, f, indent=2)

print("\n✅ Saved: model.pkl, scaler.pkl, "
      "feature_names.json, encoders.json")