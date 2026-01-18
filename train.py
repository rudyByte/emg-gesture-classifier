import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = "data/Synapse_Dataset"
TRAIN_SESSIONS = ["Session1", "Session2"]
VAL_SESSIONS = ["Session3"]

def bandpass(signal, low=20, high=450, fs=1000):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal):
    feats = []
    for ch in signal.T:
        rms = np.sqrt(np.mean(ch**2))
        mav = np.mean(np.abs(ch))
        wl = np.sum(np.abs(np.diff(ch)))
        feats.extend([rms, mav, wl])
    return feats

def load_session(session_name):
    X, y = [], []
    session_path = os.path.join(BASE_DIR, session_name)
    for subject in os.listdir(session_path):
        subject_path = os.path.join(session_path, subject)
        for file in os.listdir(subject_path):
            if not file.endswith(".csv"):
                continue
            gesture = int(file.split("gesture")[1].split("_")[0])
            df = pd.read_csv(os.path.join(subject_path, file))
            signal = df.values[:, :8]
            signal = np.apply_along_axis(bandpass, 0, signal)
            feats = extract_features(signal)
            X.append(feats)
            y.append(gesture)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = [], []
for s in TRAIN_SESSIONS:
    Xs, ys = load_session(s)
    X_train.append(Xs)
    y_train.append(ys)

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

print("Loading validation data...")
X_val, y_val = load_session(VAL_SESSIONS[0])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds, average="macro")

print("Validation Accuracy:", acc)
print("Validation F1 Score:", f1)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
