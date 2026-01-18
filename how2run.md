# How to Run and Test the Project

This document explains how to train the model and run inference step by step.

---

## 1. Environment Setup

### Prerequisites
- Python 3.9 or higher
- pip installed

### Install Dependencies
Run the following command in the project root:

```bash
pip install -r requirements.txt


2. Dataset Setup

Download the dataset from the official competition source and place it in the following structure:

data/
└── Synapse_Dataset/
    ├── Session1/
    ├── Session2/
    └── Session3/


⚠️ Do not rename files or folders.

3. Training the Model

Run the training script:

python train.py


This will:

Load Session1 and Session2 for training

Validate on Session3

Print validation accuracy and F1-score

Save model.pkl and scaler.pkl


4. Running Inference

To predict the gesture for a single CSV file:

python infer.py <path_to_csv_file>


Example:

python infer.py data/Synapse_Dataset/Session3/session3_subject_1/gesture02_trial03.csv


Output:

Predicted Gesture Class: 2


Gesture classes range from 0 to 4.


5. Notes

The inference script applies the same preprocessing and feature extraction used during training

Session-aware validation is intentionally used to avoid data leakage

Results prioritize generalization over artificially inflated accuracy
