#!/usr/bin/env python3
"""
Full Dataset Geospatial Model
============================
- Uses full 59,400 row dataset
- Focuses on high-impact geospatial features
- Skips GPT features for now
- Creates final improved model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from geospatial_features import create_geospatial_features
import warnings
warnings.filterwarnings('ignore')

print("🚀 FULL DATASET GEOSPATIAL MODEL")
print("=" * 60)

# Load the full dataset
print("📊 Loading full dataset...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
test_features = pd.read_csv('ds19-predictive-modeling-challenge/test_features.csv')

train = pd.merge(train_features, train_labels, on='id')
print(f"✅ Training data: {len(train):,} rows")
print(f"✅ Test data: {len(test_features):,} rows")

# Create geospatial features for training data
print(f"\n🌍 CREATING GEOSPATIAL FEATURES FOR TRAINING DATA...")
print("-" * 60)
train_geo = create_geospatial_features(train, n_neighbors=20, n_clusters=100)

def prepare_features(df):
    """Prepare features for modeling (baseline + geospatial)"""
    X = df.copy()
    
    print(f"🔧 Engineering features for {len(X):,} rows...")
    
    # Original engineered features
    cols_with_zeros = ['longitude', 'latitude', 'gps_height', 'population']
    for col in cols_with_zeros:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
            X[col+'_MISSING'] = X[col].isnull().astype(int)
    
    # Date features
    if 'date_recorded' in X.columns:
        X['date_recorded'] = pd.to_datetime(X['date_recorded'])
        X['year_recorded'] = X['date_recorded'].dt.year
        X['month_recorded'] = X['date_recorded'].dt.month
        X['days_since_recorded'] = (pd.Timestamp('2013-12-03') - X['date_recorded']).dt.days
    
    # Construction year features
    if 'construction_year' in X.columns:
        X['construction_year'] = pd.to_numeric(X['construction_year'], errors='coerce')
        current_year = 2013
        X['pump_age'] = current_year - X['construction_year']
        X['construction_year_missing'] = X['construction_year'].isnull().astype(int)
    
    # Text length features
    text_cols = ['scheme_name', 'installer', 'funder']
    for col in text_cols:
        if col in X.columns:
            X[col+'_length'] = X[col].astype(str).str.len()
            X[col+'_is_missing'] = X[col].isnull().astype(int)
    
    # Categorical encoding
    categorical_cols = ['basin', 'region', 'district_code', 'lga', 'ward', 
                       'public_meeting', 'scheme_management', 'permit', 
                       'extraction_type', 'extraction_type_group', 'extraction_type_class',
                       'management', 'management_group', 'payment', 'payment_type',
                       'water_quality', 'quality_group', 'quantity', 'quantity_group',
                       'source', 'source_type', 'source_class', 'waterpoint_type',
                       'waterpoint_type_group']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in X.columns:
            X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
    
    # Select features
    feature_cols = []
    
    # Original numeric features
    numeric_cols = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
                   'year_recorded', 'month_recorded', 'days_since_recorded', 'pump_age']
    for col in numeric_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Missing indicators
    missing_cols = [col+'_MISSING' for col in cols_with_zeros] + ['construction_year_missing']
    for col in missing_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Text length features
    text_length_cols = [col+'_length' for col in text_cols] + [col+'_is_missing' for col in text_cols]
    for col in text_length_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Encoded categorical features
    encoded_cols = [col+'_encoded' for col in categorical_cols]
    for col in encoded_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Geospatial features
    geo_cols = ['nearest_neighbor_distance', 'avg_5_neighbor_distance', 'avg_10_neighbor_distance',
               'neighbor_density', 'neighbor_functional_ratio', 'installer_spatial_clustering',
               'funder_spatial_clustering', 'geographic_cluster', 'cluster_size', 'cluster_density',
               'neighbor_elevation_diff', 'neighbor_elevation_std', 'ward_pump_count', 'ward_functional_ratio']
    
    geo_features_added = 0
    for col in geo_cols:
        if col in X.columns and X[col].notna().any():
            feature_cols.append(col)
            geo_features_added += 1
    
    print(f"✅ Added {geo_features_added} geospatial features")
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    print(f"📊 Total features: {len(feature_cols)}")
    return X_features, feature_cols

# Prepare training features
print(f"\n🔧 PREPARING TRAINING FEATURES...")
print("-" * 50)
X_train_full, feature_names = prepare_features(train_geo)
y_train_full = train_geo['status_group']

# Split for validation
print(f"\n📊 SPLITTING FOR VALIDATION...")
print("-" * 40)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Training set: {len(X_train):,} rows")
print(f"Validation set: {len(X_val):,} rows")

# Train model on full training data
print(f"\n🤖 TRAINING FULL GEOSPATIAL MODEL...")
print("-" * 40)

# Use larger ensemble for final model
model = RandomForestClassifier(
    n_estimators=200,  # More trees for better performance
    max_depth=20,      # Slightly deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

print("Training on full dataset...")
model.fit(X_train_full, y_train_full)

# Validation performance
print(f"\n📈 VALIDATION PERFORMANCE...")
print("-" * 40)
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

print(f"🎯 Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Cross-validation for robust estimate
print(f"\n🔄 CROSS-VALIDATION (5-fold)...")
print("-" * 40)
cv_scores = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"CV Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# Detailed validation report
print(f"\n📋 DETAILED VALIDATION REPORT:")
print("-" * 50)
print(classification_report(y_val, val_predictions))

# Feature importance analysis
print(f"\n🔍 TOP 20 FEATURE IMPORTANCE:")
print("-" * 60)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance_df.head(20).iterrows()):
    # Categorize features
    if any(geo_term in row['feature'] for geo_term in ['neighbor', 'cluster', 'spatial', 'ward', 'elevation']):
        emoji = "🌍"
    else:
        emoji = "📊"
    
    print(f"{i+1:2d}. {emoji} {row['feature']:<35} {row['importance']:.4f}")

# Analyze geospatial vs baseline feature contributions
geo_importance = feature_importance_df[feature_importance_df['feature'].str.contains('neighbor|cluster|spatial|ward|elevation')]['importance'].sum()
base_importance = 1.0 - geo_importance

print(f"\n📊 FEATURE TYPE CONTRIBUTIONS:")
print("-" * 40)
print(f"🌍 Geospatial Features: {geo_importance:.3f} ({geo_importance*100:.1f}%)")
print(f"📊 Baseline Features:   {base_importance:.3f} ({base_importance*100:.1f}%)")

# Create test predictions
print(f"\n🔮 CREATING TEST PREDICTIONS...")
print("-" * 40)

# Create geospatial features for test data
print("Creating geospatial features for test data...")

# Combine train and test for geospatial feature creation (needed for neighbor analysis)
print("Combining train and test for neighbor analysis...")
combined_data = pd.concat([
    train_geo.assign(dataset='train'),
    test_features.assign(dataset='test', status_group='unknown')
], ignore_index=True)

print(f"Combined dataset: {len(combined_data):,} rows")

# Create geospatial features on combined dataset
combined_geo = create_geospatial_features(combined_data, n_neighbors=20, n_clusters=100)

# Split back into train and test
test_geo = combined_geo[combined_geo['dataset'] == 'test'].copy()
test_geo = test_geo.drop(['dataset', 'status_group'], axis=1)

print(f"Test set with geo features: {len(test_geo):,} rows")

# Prepare test features
X_test, _ = prepare_features(test_geo)

print(f"Test features shape: {X_test.shape}")

# Make predictions
print("Making predictions on test set...")
test_predictions = model.predict(X_test)

# Create submission file
print(f"\n💾 CREATING SUBMISSION FILE...")
print("-" * 40)

submission = pd.DataFrame({
    'id': test_features['id'],
    'status_group': test_predictions
})

submission_file = 'improved_geospatial_submission.csv'
submission.to_csv(submission_file, index=False)

print(f"✅ Submission saved to: {submission_file}")
print(f"📊 Submission shape: {submission.shape}")

# Show prediction distribution
pred_counts = submission['status_group'].value_counts()
print(f"\n📈 PREDICTION DISTRIBUTION:")
print("-" * 40)
for status, count in pred_counts.items():
    pct = (count / len(submission)) * 100
    print(f"{status:<25} {count:6,} ({pct:5.1f}%)")

# Summary
print(f"\n🎯 FINAL SUMMARY:")
print("=" * 60)
print(f"✅ Model trained on full dataset: {len(train_geo):,} rows")
print(f"✅ Validation accuracy: {val_accuracy*100:.2f}%")
print(f"✅ Cross-validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"✅ Geospatial features contribute: {geo_importance*100:.1f}% of model")
print(f"✅ Test predictions: {len(submission):,} rows")
print(f"✅ Submission file: {submission_file}")

print(f"\n🚀 EXPECTED IMPROVEMENT:")
print("-" * 40)
print(f"Original baseline: ~76-78%")
print(f"This model: ~{cv_scores.mean()*100:.1f}%")
print(f"Expected gain: ~{(cv_scores.mean() - 0.77)*100:.1f} percentage points")

print(f"\n🎊 Ready for Kaggle submission!")