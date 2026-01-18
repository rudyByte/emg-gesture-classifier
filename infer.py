import sys
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

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
    return np.array(feats).reshape(1, -1)

if len(sys.argv) < 2:
    print("Usage: python infer.py <path_to_csv>")
    sys.exit(1)

csv_path = sys.argv[1]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(csv_path)
signal = df.values[:, :8]
signal = np.apply_along_axis(bandpass, 0, signal)

features = extract_features(signal)
features = scaler.transform(features)

prediction = model.predict(features)[0]

print("Predicted Gesture Class:", prediction)
