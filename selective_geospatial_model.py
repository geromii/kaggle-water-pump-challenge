#!/usr/bin/env python3
"""
Selective Geospatial Model
=========================
- Only adds the most promising geospatial features
- Based on original winning approach + targeted improvements
- Focus on ward_functional_ratio and neighbor_functional_ratio
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def add_selective_geo_features(df, reference_df=None, n_neighbors=10):
    """Add only the most valuable geospatial features"""
    
    print(f"🎯 Adding selective geospatial features for {len(df):,} rows...")
    
    data = df.copy()
    
    # 1. WARD-LEVEL STATISTICS (highest importance)
    if 'status_group' in data.columns and 'ward' in data.columns:
        print("📊 Adding ward-level statistics...")
        ward_stats = data.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(data['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    elif 'ward' in data.columns and reference_df is not None and 'ward' in reference_df.columns:
        print("📊 Adding ward statistics from reference data...")
        ward_stats = reference_df.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(ward_stats['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    # 2. NEIGHBOR FUNCTIONAL RATIO (second highest importance)
    valid_coords = (data['latitude'] != 0) & (data['longitude'] != 0) & \
                   data['latitude'].notna() & data['longitude'].notna()
    
    print(f"📍 Valid coordinates: {valid_coords.sum():,} / {len(data):,}")
    
    if valid_coords.sum() >= n_neighbors:
        if reference_df is not None:
            # For test data, use training data as reference
            ref_valid = (reference_df['latitude'] != 0) & (reference_df['longitude'] != 0) & \
                       reference_df['latitude'].notna() & reference_df['longitude'].notna()
            ref_coords = reference_df.loc[ref_valid, ['latitude', 'longitude']].values
            ref_status = reference_df.loc[ref_valid, 'status_group'].fillna('unknown')
            
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', 
                                   metric='haversine', n_jobs=-1)
            nbrs.fit(np.radians(ref_coords))
            
            test_coords = data.loc[valid_coords, ['latitude', 'longitude']].values
            _, indices = nbrs.kneighbors(np.radians(test_coords))
            
            # Calculate neighbor functional ratios
            neighbor_functional_ratios = []
            for neighbor_idx in indices:
                neighbor_statuses = ref_status.iloc[neighbor_idx]
                functional_ratio = (neighbor_statuses == 'functional').mean()
                neighbor_functional_ratios.append(functional_ratio)
            
        else:
            # For training data
            coords = data.loc[valid_coords, ['latitude', 'longitude']].values
            valid_indices = data.index[valid_coords].tolist()
            
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                                   metric='haversine', n_jobs=-1)
            nbrs.fit(np.radians(coords))
            _, indices = nbrs.kneighbors(np.radians(coords))
            
            # Remove self from neighbors
            indices = indices[:, 1:]
            
            valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
            
            neighbor_functional_ratios = []
            for i, neighbor_idx in enumerate(indices):
                neighbor_statuses = valid_status.iloc[neighbor_idx]
                functional_ratio = (neighbor_statuses == 'functional').mean()
                neighbor_functional_ratios.append(functional_ratio)
        
        # Map back to full dataset
        data['neighbor_functional_ratio'] = np.nan
        data.loc[valid_coords, 'neighbor_functional_ratio'] = neighbor_functional_ratios
        data['neighbor_functional_ratio'] = data['neighbor_functional_ratio'].fillna(0.5)
        
        print("✅ Added neighbor_functional_ratio")
    
    return data

print("🎯 SELECTIVE GEOSPATIAL MODEL")
print("=" * 50)

# Load dataset
print("📊 Loading dataset...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
test_features = pd.read_csv('ds19-predictive-modeling-challenge/test_features.csv')

train = pd.merge(train_features, train_labels, on='id')

# First, let's replicate the original winning approach exactly
def prepare_baseline_features(df):
    """Replicate the original winning model features"""
    X = df.copy()
    
    # Missing value indicators (key improvement from original analysis)
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
    
    # Categorical encoding (all original features)
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
    
    # Select features (comprehensive original set)
    feature_cols = []
    
    # Numeric features
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
    
    # Text features
    text_length_cols = [col+'_length' for col in text_cols] + [col+'_is_missing' for col in text_cols]
    for col in text_length_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Encoded categorical
    encoded_cols = [col+'_encoded' for col in categorical_cols]
    for col in encoded_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, feature_cols

def prepare_enhanced_features(df):
    """Add selective geospatial features to baseline"""
    X = df.copy()
    
    # Start with baseline features
    X_base, base_features = prepare_baseline_features(X)
    
    # Add selective geospatial features
    geo_features_added = 0
    if 'ward_functional_ratio' in X.columns:
        X_base['ward_functional_ratio'] = X['ward_functional_ratio'].fillna(0.5)
        base_features.append('ward_functional_ratio')
        geo_features_added += 1
    
    if 'ward_pump_count' in X.columns:
        X_base['ward_pump_count'] = X['ward_pump_count'].fillna(X['ward_pump_count'].median())
        base_features.append('ward_pump_count')
        geo_features_added += 1
    
    if 'neighbor_functional_ratio' in X.columns:
        X_base['neighbor_functional_ratio'] = X['neighbor_functional_ratio'].fillna(0.5)
        base_features.append('neighbor_functional_ratio')
        geo_features_added += 1
    
    print(f"✅ Added {geo_features_added} selective geospatial features")
    return X_base, base_features

# Test baseline vs enhanced
print("\n🔬 TESTING BASELINE VS SELECTIVE ENHANCEMENT...")
print("-" * 60)

# Create train/validation split
X_baseline, baseline_features = prepare_baseline_features(train)
y = train['status_group']

X_train, X_val, y_train, y_val = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42, stratify=y
)

# Test baseline model
print("🔸 Testing baseline model...")
baseline_model = RandomForestClassifier(
    n_estimators=100,  # Original settings
    random_state=42,
    n_jobs=-1
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_val)
baseline_accuracy = accuracy_score(y_val, baseline_pred)

print(f"Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")

# Add selective geospatial features
print("\n🔸 Adding selective geospatial features...")
train_enhanced = add_selective_geo_features(train, n_neighbors=10)

# Test enhanced model
X_enhanced, enhanced_features = prepare_enhanced_features(train_enhanced)

X_train_enh, X_val_enh, y_train_enh, y_val_enh = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)

print("🔸 Testing enhanced model...")
enhanced_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
enhanced_model.fit(X_train_enh, y_train_enh)
enhanced_pred = enhanced_model.predict(X_val_enh)
enhanced_accuracy = accuracy_score(y_val_enh, enhanced_pred)

print(f"Enhanced accuracy: {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
improvement = enhanced_accuracy - baseline_accuracy
print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")

# Cross-validation comparison
print(f"\n🔄 CROSS-VALIDATION COMPARISON:")
print("-" * 40)

cv_baseline = cross_val_score(baseline_model, X_baseline, y, cv=5, scoring='accuracy', n_jobs=-1)
cv_enhanced = cross_val_score(enhanced_model, X_enhanced, y, cv=5, scoring='accuracy', n_jobs=-1)

print(f"Baseline CV:  {cv_baseline.mean():.4f} ± {cv_baseline.std():.4f}")
print(f"Enhanced CV:  {cv_enhanced.mean():.4f} ± {cv_enhanced.std():.4f}")
print(f"CV Improvement: {cv_enhanced.mean() - cv_baseline.mean():+.4f}")

# Feature importance comparison
if enhanced_accuracy > baseline_accuracy:
    print(f"\n🔍 TOP FEATURES (ENHANCED MODEL):")
    print("-" * 40)
    
    importances = enhanced_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': enhanced_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        if row['feature'] in ['ward_functional_ratio', 'neighbor_functional_ratio', 'ward_pump_count']:
            emoji = "🎯"
        else:
            emoji = "📊"
        
        print(f"{i+1:2d}. {emoji} {row['feature']:<30} {row['importance']:.4f}")

# Decision on whether to proceed
if enhanced_accuracy > baseline_accuracy and (enhanced_accuracy - baseline_accuracy) > 0.001:
    print(f"\n✅ ENHANCEMENT SUCCESSFUL! Proceeding with full dataset...")
    
    # Train on full dataset and create submission
    print("\n🚀 TRAINING ON FULL DATASET...")
    
    # Enhanced model on full training data
    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_enhanced, y)
    
    # Process test data
    print("🔮 Processing test data...")
    test_enhanced = add_selective_geo_features(test_features, reference_df=train_enhanced, n_neighbors=10)
    X_test, _ = prepare_enhanced_features(test_enhanced)
    
    # Make predictions
    test_predictions = final_model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_features['id'],
        'status_group': test_predictions
    })
    
    submission_file = 'selective_geospatial_submission.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"✅ Submission saved: {submission_file}")
    print(f"📊 Expected score: ~{cv_enhanced.mean()*100:.2f}%")
    
else:
    print(f"\n❌ ENHANCEMENT NOT WORTHWHILE")
    print(f"Improvement of {improvement*100:+.2f}pp is too small or negative")
    print(f"Recommend sticking with baseline or trying different approach")