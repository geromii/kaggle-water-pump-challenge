#!/usr/bin/env python3
"""
Test GPT Features with Limited Training Data
============================================
Train on only 5% of data to see if GPT features help when data is scarce
Hypothesis: GPT features provide useful priors when training data is limited
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import warnings
import sqlite3
warnings.filterwarnings('ignore')

print("🧪 TESTING GPT FEATURES WITH LIMITED TRAINING DATA")
print("=" * 60)

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Load GPT features
print("🤖 Loading GPT features...")
try:
    conn = sqlite3.connect('gpt_features_progress.db')
    gpt_features = pd.read_sql_query('''
        SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
               gpt_scheme_pattern, gpt_text_language, gpt_coordination 
        FROM gpt_features ORDER BY id
    ''', conn)
    conn.close()
    print(f"✅ GPT features loaded: {len(gpt_features):,} rows")
except:
    print("❌ No GPT features found")
    exit(1)

# Merge GPT features
train = train.merge(gpt_features, on='id', how='left')

def add_selective_geo_features(df, n_neighbors=10):
    """Add selective geospatial features"""
    data = df.copy()
    
    # Ward-level statistics
    if 'status_group' in data.columns and 'ward' in data.columns:
        ward_stats = data.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(data['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    # Neighbor functional ratio
    valid_coords = (data['latitude'] != 0) & (data['longitude'] != 0) & \
                   data['latitude'].notna() & data['longitude'].notna()
    
    if valid_coords.sum() >= n_neighbors:
        coords = data.loc[valid_coords, ['latitude', 'longitude']].values
        valid_indices = data.index[valid_coords].tolist()
        
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors+1, len(coords)), algorithm='ball_tree', 
                               metric='haversine', n_jobs=-1)
        nbrs.fit(np.radians(coords))
        _, indices = nbrs.kneighbors(np.radians(coords))
        indices = indices[:, 1:] if indices.shape[1] > 1 else indices  # Remove self if possible
        
        valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
        
        neighbor_functional_ratios = []
        for i, neighbor_idx in enumerate(indices):
            if len(neighbor_idx) > 0:
                neighbor_statuses = valid_status.iloc[neighbor_idx]
                functional_ratio = (neighbor_statuses == 'functional').mean()
            else:
                functional_ratio = 0.5  # Default if no neighbors
            neighbor_functional_ratios.append(functional_ratio)
        
        data['neighbor_functional_ratio'] = np.nan
        data.loc[valid_coords, 'neighbor_functional_ratio'] = neighbor_functional_ratios
        data['neighbor_functional_ratio'] = data['neighbor_functional_ratio'].fillna(0.5)
    
    return data

def prepare_features(df, include_gpt=False, le_dict=None):
    """Prepare features with optional GPT features"""
    X = df.copy()
    
    # Missing value indicators
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
    
    if le_dict is None:
        le_dict = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
    else:
        # Use existing encoders, handle unseen categories
        for col in categorical_cols:
            if col in X.columns and col in le_dict:
                le = le_dict[col]
                col_values = X[col].astype(str)
                encoded_values = []
                for val in col_values:
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        # Map to most common class
                        encoded_values.append(le.transform([le.classes_[0]])[0])
                X[col+'_encoded'] = encoded_values
    
    # Select features
    feature_cols = []
    
    # All numeric features
    numeric_cols = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
                   'year_recorded', 'month_recorded', 'days_since_recorded', 'pump_age']
    feature_cols.extend([col for col in numeric_cols if col in X.columns])
    
    # Missing indicators
    missing_cols = [col+'_MISSING' for col in cols_with_zeros] + ['construction_year_missing']
    feature_cols.extend([col for col in missing_cols if col in X.columns])
    
    # Text features
    text_feature_cols = [col+'_length' for col in text_cols] + [col+'_is_missing' for col in text_cols]
    feature_cols.extend([col for col in text_feature_cols if col in X.columns])
    
    # Encoded categorical
    encoded_cols = [col+'_encoded' for col in categorical_cols]
    feature_cols.extend([col for col in encoded_cols if col in X.columns])
    
    # Geo features
    geo_cols = ['ward_functional_ratio', 'neighbor_functional_ratio']
    feature_cols.extend([col for col in geo_cols if col in X.columns])
    
    # GPT features if requested
    if include_gpt:
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        gpt_added = 0
        for col in gpt_cols:
            if col in X.columns and X[col].notna().any():
                X[col] = X[col].fillna(3.0)
                feature_cols.append(col)
                gpt_added += 1
        print(f"      Added {gpt_added} GPT features")
    
    X_features = X[feature_cols].fillna(-1)
    
    # Add interaction features
    if 'quantity_encoded' in X_features.columns and 'water_quality_encoded' in X_features.columns:
        X_features['quantity_quality_interact'] = X_features['quantity_encoded'] * X_features['water_quality_encoded']
    
    if 'ward_functional_ratio' in X_features.columns and 'pump_age' in X_features.columns:
        X_features['ward_age_interact'] = X_features['ward_functional_ratio'] * X_features['pump_age'].fillna(X_features['pump_age'].median())
    
    if 'neighbor_functional_ratio' in X_features.columns and 'days_since_recorded' in X_features.columns:
        X_features['neighbor_time_interact'] = X_features['neighbor_functional_ratio'] * X_features['days_since_recorded'].fillna(X_features['days_since_recorded'].median())
    
    return X_features, le_dict

# Prepare full dataset with geospatial features
print("\n🌍 Adding geospatial features to full dataset...")
train_geo = add_selective_geo_features(train, n_neighbors=10)
y = train_geo['status_group']

# Test different training set sizes
training_sizes = [0.01, 0.02, 0.05, 0.10, 0.20]  # 1%, 2%, 5%, 10%, 20%

print(f"\n📊 TESTING DIFFERENT TRAINING SET SIZES")
print("=" * 60)

all_results = []

for train_size in training_sizes:
    print(f"\n🎯 Training with {train_size*100:.0f}% of data ({int(len(train_geo)*train_size):,} samples)")
    print("-" * 50)
    
    # Create stratified split to get small training set
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    train_idx, remaining_idx = next(splitter.split(train_geo, y))
    
    # Use remaining data for validation (or a subset if it's too large)
    if len(remaining_idx) > 5000:
        val_idx = np.random.choice(remaining_idx, 5000, replace=False)
    else:
        val_idx = remaining_idx
    
    train_subset = train_geo.iloc[train_idx]
    val_subset = train_geo.iloc[val_idx]
    y_train_subset = y.iloc[train_idx]
    y_val_subset = y.iloc[val_idx]
    
    print(f"   Train: {len(train_subset):,} | Val: {len(val_subset):,}")
    
    # Add geospatial features to subsets
    train_subset_geo = add_selective_geo_features(train_subset, n_neighbors=min(5, len(train_subset)//10))
    
    # Test baseline (no GPT)
    print(f"   🔸 Baseline (no GPT)...")
    X_train_base, le_dict = prepare_features(train_subset_geo, include_gpt=False)
    X_val_base, _ = prepare_features(val_subset, include_gpt=False, le_dict=le_dict)
    
    model_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_base.fit(X_train_base, y_train_subset)
    pred_base = model_base.predict(X_val_base)
    score_base = accuracy_score(y_val_subset, pred_base)
    
    print(f"      Features: {X_train_base.shape[1]}")
    print(f"      Score: {score_base:.4f}")
    
    # Test with GPT features
    print(f"   🔸 With GPT features...")
    X_train_gpt, le_dict_gpt = prepare_features(train_subset_geo, include_gpt=True)
    X_val_gpt, _ = prepare_features(val_subset, include_gpt=True, le_dict=le_dict_gpt)
    
    model_gpt = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_gpt.fit(X_train_gpt, y_train_subset)
    pred_gpt = model_gpt.predict(X_val_gpt)
    score_gpt = accuracy_score(y_val_subset, pred_gpt)
    
    improvement = score_gpt - score_base
    print(f"      Features: {X_train_gpt.shape[1]}")
    print(f"      Score: {score_gpt:.4f} ({improvement:+.4f})")
    
    # Analyze GPT feature importance in the small data regime
    gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
               'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
    
    gpt_importances = []
    feature_names = X_train_gpt.columns.tolist()
    for col in gpt_cols:
        if col in feature_names:
            importance = model_gpt.feature_importances_[feature_names.index(col)]
            gpt_importances.append((col, importance))
    
    # Sort by importance
    gpt_importances.sort(key=lambda x: x[1], reverse=True)
    
    if gpt_importances:
        print(f"      Top GPT features:")
        for col, imp in gpt_importances[:3]:
            print(f"        {col}: {imp:.4f}")
    
    all_results.append({
        'train_size': train_size,
        'train_samples': len(train_subset),
        'baseline_score': score_base,
        'gpt_score': score_gpt,
        'improvement': improvement,
        'baseline_features': X_train_base.shape[1],
        'gpt_features': X_train_gpt.shape[1]
    })

# Summary analysis
print(f"\n📈 SUMMARY ANALYSIS")
print("=" * 60)
print(f"{'Size':<6} {'Samples':<8} {'Baseline':<10} {'With GPT':<10} {'Improvement':<12} {'Better?'}")
print("-" * 60)

for result in all_results:
    size_str = f"{result['train_size']*100:.0f}%"
    samples_str = f"{result['train_samples']:,}"
    baseline_str = f"{result['baseline_score']:.4f}"
    gpt_str = f"{result['gpt_score']:.4f}"
    improvement_str = f"{result['improvement']:+.4f}"
    
    if result['improvement'] > 0.002:
        better = "✅ Yes"
    elif result['improvement'] > 0:
        better = "⚠️ Slight"
    else:
        better = "❌ No"
    
    print(f"{size_str:<6} {samples_str:<8} {baseline_str:<10} {gpt_str:<10} {improvement_str:<12} {better}")

# Find the crossover point
positive_improvements = [r for r in all_results if r['improvement'] > 0.001]
if positive_improvements:
    best_small_data = min(positive_improvements, key=lambda x: x['train_size'])
    print(f"\n🎯 KEY FINDINGS:")
    print("-" * 40)
    print(f"✅ GPT features help with ≤{best_small_data['train_size']*100:.0f}% training data")
    print(f"   Best improvement: {best_small_data['improvement']*100:+.2f}pp at {best_small_data['train_size']*100:.0f}%")
    print(f"   This suggests GPT features provide useful priors when data is scarce")
else:
    print(f"\n🎯 KEY FINDINGS:")
    print("-" * 40)
    print(f"❌ GPT features don't help even with limited training data")
    print(f"   The baseline categorical features are consistently better")

# Data efficiency analysis
print(f"\n📊 DATA EFFICIENCY ANALYSIS:")
print("-" * 40)
best_baseline = max(r['baseline_score'] for r in all_results)
best_gpt = max(r['gpt_score'] for r in all_results)

print(f"Best baseline score: {best_baseline:.4f}")
print(f"Best GPT score: {best_gpt:.4f}")

# Find how much data baseline needs to beat best GPT
baseline_beats_gpt = [r for r in all_results if r['baseline_score'] > best_gpt]
if baseline_beats_gpt:
    min_data_needed = min(r['train_size'] for r in baseline_beats_gpt)
    print(f"Baseline needs ≥{min_data_needed*100:.0f}% data to beat best GPT score")
else:
    print(f"Baseline never beats best GPT score in this test")

print("\n✅ Limited data testing complete!")