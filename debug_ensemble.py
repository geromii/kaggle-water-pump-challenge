#!/usr/bin/env python3
"""
Debug Ensemble Issues
====================
- Check label encoding and data flow
- Identify where the problem occurs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def debug_data_flow():
    """Debug each step of the data processing"""
    print("🔍 DEBUGGING DATA FLOW")
    print("=" * 50)
    
    # Load data
    print("1. Loading data...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    print(f"   ✅ Loaded {len(train):,} rows")
    print(f"   📊 Target classes: {train['status_group'].value_counts().to_dict()}")
    
    # Take small sample for debugging
    train_sample = train.head(1000).copy()
    print(f"   🎯 Using {len(train_sample)} rows for debugging")
    
    # 2. Check target variable
    print("\n2. Checking target variable...")
    print(f"   Status groups: {train_sample['status_group'].unique()}")
    print(f"   Value counts: {train_sample['status_group'].value_counts().to_dict()}")
    print(f"   Any nulls: {train_sample['status_group'].isnull().sum()}")
    
    # 3. Basic feature engineering
    print("\n3. Basic feature engineering...")
    X = train_sample.copy()
    
    # Simple numeric features
    numeric_features = []
    if 'longitude' in X.columns:
        X['longitude'] = X['longitude'].replace(0, np.nan)
        X['longitude_clean'] = X['longitude'].fillna(X['longitude'].median())
        numeric_features.append('longitude_clean')
    
    if 'latitude' in X.columns:
        X['latitude'] = X['latitude'].replace(0, np.nan)
        X['latitude_clean'] = X['latitude'].fillna(X['latitude'].median())
        numeric_features.append('latitude_clean')
    
    if 'amount_tsh' in X.columns:
        X['amount_tsh_clean'] = X['amount_tsh'].fillna(0)
        numeric_features.append('amount_tsh_clean')
    
    if 'population' in X.columns:
        X['population'] = X['population'].replace(0, np.nan)
        X['population_clean'] = X['population'].fillna(X['population'].median())
        numeric_features.append('population_clean')
    
    # Key categorical feature
    if 'quantity' in X.columns:
        le_quantity = LabelEncoder()
        X['quantity_encoded'] = le_quantity.fit_transform(X['quantity'].astype(str))
        numeric_features.append('quantity_encoded')
        print(f"   Quantity classes: {le_quantity.classes_}")
    
    print(f"   ✅ Created {len(numeric_features)} features: {numeric_features}")
    
    # 4. Prepare for modeling
    print("\n4. Preparing for modeling...")
    X_model = X[numeric_features]
    y_model = X['status_group']
    
    print(f"   Feature matrix shape: {X_model.shape}")
    print(f"   Target shape: {y_model.shape}")
    print(f"   Feature dtypes: {X_model.dtypes.to_dict()}")
    print(f"   Any NaN in features: {X_model.isnull().sum().sum()}")
    print(f"   Any NaN in target: {y_model.isnull().sum()}")
    
    # Check feature ranges
    print(f"\n   Feature summary:")
    for col in numeric_features:
        print(f"     {col}: min={X_model[col].min():.2f}, max={X_model[col].max():.2f}, mean={X_model[col].mean():.2f}")
    
    # 5. Train-test split
    print("\n5. Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.3, random_state=42, stratify=y_model
    )
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"   Test target distribution: {y_test.value_counts().to_dict()}")
    
    # 6. Simple model test
    print("\n6. Testing simple Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Model trained successfully")
    print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Check predictions
    print(f"   Predicted classes: {np.unique(y_pred)}")
    print(f"   True classes: {np.unique(y_test)}")
    print(f"   Prediction distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    
    # Feature importance
    print(f"\n   Feature importances:")
    for i, feature in enumerate(numeric_features):
        print(f"     {feature}: {rf.feature_importances_[i]:.4f}")
    
    # Classification report
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if accuracy < 0.1:
        print("\n❌ PROBLEM DETECTED: Very low accuracy!")
        print("Possible issues:")
        print("- Label encoding problem")
        print("- Feature scaling issue")
        print("- Data leakage or contamination")
        print("- Wrong train/test split")
    elif accuracy > 0.5:
        print(f"\n✅ SUCCESS: Model working correctly!")
        print(f"Accuracy of {accuracy:.2%} is reasonable for a simple model")
    else:
        print(f"\n⚠️  MODERATE ISSUE: Accuracy of {accuracy:.2%} is low but not zero")
        print("Consider feature engineering improvements")
    
    return accuracy > 0.1

def test_with_original_ensemble():
    """Test using a simplified version of the original ensemble approach"""
    print(f"\n🔬 TESTING SIMPLIFIED ENSEMBLE")
    print("=" * 50)
    
    # Load data
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    # Use small sample
    train_sample = train.head(2000).copy()
    print(f"Using {len(train_sample)} rows")
    
    # Simple feature engineering (mimicking our successful previous approach)
    X = train_sample.copy()
    
    # Handle coordinates
    coord_cols = ['longitude', 'latitude', 'gps_height']
    for col in coord_cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
            X[col+'_MISSING'] = X[col].isnull().astype(int)
    
    # Basic categorical encoding
    key_cats = ['quantity', 'waterpoint_type', 'payment']
    for col in key_cats:
        if col in X.columns:
            le = LabelEncoder()
            X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
    
    # Select features for modeling
    feature_cols = []
    
    # Numeric
    numeric_cols = ['amount_tsh', 'longitude', 'latitude', 'gps_height', 'population']
    for col in numeric_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Missing indicators
    missing_cols = [col+'_MISSING' for col in coord_cols]
    for col in missing_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Encoded categoricals
    encoded_cols = [col+'_encoded' for col in key_cats]
    for col in encoded_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    print(f"Selected features: {feature_cols}")
    
    # Prepare data
    X_features = X[feature_cols].fillna(-1)
    y = X['status_group']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train multiple models
    models = {}
    
    # Random Forest
    models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    models['rf'].fit(X_train, y_train)
    
    # Make predictions
    predictions = {}
    accuracies = {}
    
    for name, model in models.items():
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        predictions[name] = pred
        accuracies[name] = acc
        print(f"   {name}: {acc:.4f} ({acc*100:.2f}%)")
    
    return accuracies

if __name__ == "__main__":
    print("🔍 ENSEMBLE DEBUGGING SESSION")
    print("=" * 60)
    
    # Step 1: Debug basic data flow
    success = debug_data_flow()
    
    if success:
        print(f"\n✅ Basic model works! Proceeding to ensemble test...")
        # Step 2: Test simplified ensemble
        test_with_original_ensemble()
    else:
        print(f"\n❌ Basic model failed! Need to fix fundamental issues first.")
    
    print(f"\n🎯 DEBUGGING COMPLETE")
    print("=" * 60)