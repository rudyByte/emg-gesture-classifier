# Synapse: sEMG Gesture Classification  
**The NeuroTech Challenge – IIT Dharwad**

## Overview
This repository contains a complete end-to-end machine learning pipeline for decoding hand gestures from multichannel surface electromyography (sEMG) signals. The project was developed as part of **Synapse: The NeuroTech Challenge** and focuses on robustness, generalization, and physiological interpretability rather than inflated accuracy.

The system predicts one of five discrete hand gestures from raw 8-channel sEMG recordings and is designed to generalize across subjects and recording sessions.

---

## Problem Statement
Given multichannel sEMG signals recorded from forearm muscles, the goal is to classify the intended hand gesture among five predefined gesture classes. The main challenges include biological signal noise, subject variability, and session-to-session variation.

---

## Dataset Description
- Multichannel sEMG dataset provided by the organizers
- 3 recording sessions (Session1, Session2, Session3)
- Multiple subjects per session
- Each subject contains multiple gesture trials
- Each CSV file contains raw 8-channel time-series sEMG data
- Gesture labels are encoded in filenames (`gesture00` to `gesture04`)

⚠️ **Note:** The dataset is not included in this repository and must be obtained from the official competition source.

---

## Approach Summary

### Signal Processing
- Band-pass filtering (20–450 Hz) to remove noise and motion artifacts
- Channel-wise processing to preserve muscle activation patterns

### Feature Engineering
For each of the 8 channels:
- Root Mean Square (RMS)
- Mean Absolute Value (MAV)
- Waveform Length (WL)

This results in a 24-dimensional feature vector per gesture trial.

### Model
- Random Forest Classifier
- Lightweight, interpretable, and robust to noisy biosignals
- Chosen to balance performance and generalization

### Training Strategy
- Training: Session1 + Session2
- Validation: Session3 (unseen)
- Session-aware split to avoid data leakage

---

## Repository Structure

synapse-emg-gesture-classification/
│
├── train.py # Model training script
├── infer.py # Inference script
├── model.pkl # Trained model
├── scaler.pkl # Feature scaler
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── report.tex # LaTeX technical report
└── HOW_TO_RUN.md # Step-by-step execution guide


---

## Evaluation
The model is evaluated using:
- Accuracy
- Macro F1-score

Session-wise validation is used to reflect real-world prosthetic deployment scenarios.

---

## Key Highlights
- No data leakage
- Session-aware evaluation
- Physiologically meaningful features
- Clean and reproducible pipeline
- Deployment-ready inference script

---

## Disclaimer
This project is intended for academic and research purposes as part of a hackathon challenge. The dataset belongs to the competition organizers and is not redistributed here.
