import pandas as pd
import numpy as np

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')

# Merge
train = pd.merge(train_features, train_labels, on='id')

# Check boolean columns
print("Columns with boolean type:")
bool_cols = train.select_dtypes(include=['bool']).columns
print(bool_cols.tolist())

print("\nUnique values in boolean columns:")
for col in bool_cols:
    print(f"{col}: {train[col].unique()}")

print("\nColumns with object type that might be boolean:")
obj_cols = train.select_dtypes(include=['object']).columns
for col in obj_cols:
    unique_vals = train[col].unique()
    if len(unique_vals) <= 3:
        print(f"{col}: {unique_vals[:10]}")