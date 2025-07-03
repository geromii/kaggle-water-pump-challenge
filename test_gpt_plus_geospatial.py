#!/usr/bin/env python3
"""
Test GPT + Selective Geospatial Model
====================================
- Use the 67% of GPT data we have (39,802 rows)
- Combine with selective geospatial features
- Compare against selective geospatial baseline
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
    
    # 1. WARD-LEVEL STATISTICS
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
    
    # 2. NEIGHBOR FUNCTIONAL RATIO
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
            
            indices = indices[:, 1:]  # Remove self
            
            valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
            
            neighbor_functional_ratios = []
            for i, neighbor_idx in enumerate(indices):
                neighbor_statuses = valid_status.iloc[neighbor_idx]
                functional_ratio = (neighbor_statuses == 'functional').mean()
                neighbor_functional_ratios.append(functional_ratio)
        
        data['neighbor_functional_ratio'] = np.nan
        data.loc[valid_coords, 'neighbor_functional_ratio'] = neighbor_functional_ratios
        data['neighbor_functional_ratio'] = data['neighbor_functional_ratio'].fillna(0.5)
        
        print("✅ Added neighbor_functional_ratio")
    
    return data

def prepare_baseline_features(df):
    """Baseline features (our improved version)"""
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
    
    # Key categorical features
    key_categorical_cols = ['basin', 'region', 'ward', 'public_meeting', 'permit',
                           'extraction_type', 'management', 'payment', 'water_quality', 
                           'quantity', 'source', 'waterpoint_type']
    
    le = LabelEncoder()
    for col in key_categorical_cols:
        if col in X.columns:
            X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
    
    # Select features
    feature_cols = []
    
    # Core numeric features
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
    
    # Key encoded categorical
    encoded_cols = [col+'_encoded' for col in key_categorical_cols]
    for col in encoded_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, feature_cols

def prepare_enhanced_features(df, include_gpt=True, include_geo=True):
    """Add GPT and/or geospatial features to baseline"""
    X_base, base_features = prepare_baseline_features(df)
    
    feature_cols = base_features.copy()
    
    # Add geospatial features
    if include_geo:
        geo_features_added = 0
        if 'ward_functional_ratio' in df.columns:
            X_base['ward_functional_ratio'] = df['ward_functional_ratio'].fillna(0.5)
            feature_cols.append('ward_functional_ratio')
            geo_features_added += 1
        
        if 'neighbor_functional_ratio' in df.columns:
            X_base['neighbor_functional_ratio'] = df['neighbor_functional_ratio'].fillna(0.5)
            feature_cols.append('neighbor_functional_ratio')
            geo_features_added += 1
        
        print(f"✅ Added {geo_features_added} geospatial features")
    
    # Add GPT features
    if include_gpt:
        gpt_features_added = 0
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        
        for col in gpt_cols:
            if col in df.columns and df[col].notna().any():
                X_base[col] = df[col].fillna(3.0)  # Default middle value
                feature_cols.append(col)
                gpt_features_added += 1
        
        print(f"✅ Added {gpt_features_added} GPT features")
    
    return X_base, feature_cols

print("🤖 TESTING GPT + SELECTIVE GEOSPATIAL MODEL")
print("=" * 60)

# Load datasets
print("📊 Loading datasets...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Load GPT features (67% of data)
print("🤖 Loading existing GPT features...")
gpt_features = pd.read_csv('gpt_features_progress.csv')
print(f"✅ GPT features: {len(gpt_features):,} rows ({len(gpt_features)/len(train)*100:.1f}% of dataset)")

# Merge GPT features
train_with_gpt = train.merge(gpt_features, on='id', how='left')
has_gpt = train_with_gpt['gpt_funder_org_type'].notna()
train_subset = train_with_gpt[has_gpt].copy()

print(f"🎯 Working with subset: {len(train_subset):,} rows with GPT features")

# Add geospatial features
print(f"\n🌍 Adding selective geospatial features...")
train_geo = add_selective_geo_features(train_subset, n_neighbors=10)

# Test different feature combinations
print(f"\n🔬 TESTING FEATURE COMBINATIONS...")
print("-" * 50)

y = train_geo['status_group']

# 1. Baseline only
X_baseline, _ = prepare_enhanced_features(train_geo, include_gpt=False, include_geo=False)
print(f"Baseline features: {X_baseline.shape[1]}")

# 2. Baseline + Geospatial
X_geo, _ = prepare_enhanced_features(train_geo, include_gpt=False, include_geo=True)
print(f"Baseline + Geo features: {X_geo.shape[1]}")

# 3. Baseline + GPT
X_gpt, _ = prepare_enhanced_features(train_geo, include_gpt=True, include_geo=False)
print(f"Baseline + GPT features: {X_gpt.shape[1]}")

# 4. All combined
X_combined, combined_features = prepare_enhanced_features(train_geo, include_gpt=True, include_geo=True)
print(f"All combined features: {X_combined.shape[1]}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
print(f"\n🤖 TRAINING MODELS...")
print("-" * 40)

models = {}
accuracies = {}

# Test each combination
feature_sets = [
    ('baseline', X_baseline),
    ('baseline_geo', X_geo), 
    ('baseline_gpt', X_gpt),
    ('all_combined', X_combined)
]

for name, X_features in feature_sets:
    print(f"Training {name} model...")
    
    X_train_cur, X_val_cur, _, _ = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_cur, y_train)
    
    val_pred = model.predict(X_val_cur)
    accuracy = accuracy_score(y_val, val_pred)
    
    models[name] = model
    accuracies[name] = accuracy

# Results
print(f"\n🎯 RESULTS COMPARISON:")
print("=" * 50)

sorted_results = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

for i, (model_name, accuracy) in enumerate(sorted_results):
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
    print(f"{emoji} {model_name:15} {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculate improvements over baseline
baseline_acc = accuracies['baseline']
print(f"\n📈 IMPROVEMENTS OVER BASELINE:")
print("-" * 40)
for model_name, accuracy in sorted_results:
    if model_name != 'baseline':
        improvement = accuracy - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        print(f"{model_name:15} +{improvement:.4f} ({improvement_pct:+.2f}%)")

# Feature importance for best model
best_model_name = sorted_results[0][0]
if best_model_name == 'all_combined':
    best_model = models[best_model_name]
    
    print(f"\n🔍 TOP 15 FEATURES ({best_model_name.upper()}):")
    print("-" * 50)
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': combined_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
        if row['feature'].startswith('gpt_'):
            emoji = "🤖"
        elif row['feature'] in ['ward_functional_ratio', 'neighbor_functional_ratio']:
            emoji = "🌍"
        else:
            emoji = "📊"
        
        print(f"{i+1:2d}. {emoji} {row['feature']:<30} {row['importance']:.4f}")

# Cross-validation of best model
print(f"\n🔄 CROSS-VALIDATION ({best_model_name}):")
print("-" * 40)

if best_model_name == 'all_combined':
    cv_scores = cross_val_score(models[best_model_name], X_combined, y, cv=5, scoring='accuracy', n_jobs=-1)
elif best_model_name == 'baseline_geo':
    cv_scores = cross_val_score(models[best_model_name], X_geo, y, cv=5, scoring='accuracy', n_jobs=-1)
elif best_model_name == 'baseline_gpt':
    cv_scores = cross_val_score(models[best_model_name], X_gpt, y, cv=5, scoring='accuracy', n_jobs=-1)
else:
    cv_scores = cross_val_score(models[best_model_name], X_baseline, y, cv=5, scoring='accuracy', n_jobs=-1)

print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\n🎯 CONCLUSION:")
print("-" * 40)
best_accuracy = sorted_results[0][1]
best_improvement = best_accuracy - baseline_acc

if best_improvement > 0.01:
    print(f"🎉 Excellent! {best_model_name} achieves {best_improvement*100:.2f}pp improvement")
    print(f"📊 Expected score on full dataset: ~{cv_scores.mean()*100:.2f}%")
elif best_improvement > 0.005:
    print(f"✅ Good! {best_model_name} achieves {best_improvement*100:.2f}pp improvement")
    print(f"📊 Modest but meaningful gain")
else:
    print(f"📊 Marginal improvement: {best_improvement*100:.2f}pp")
    print(f"💡 May not justify added complexity")

print(f"\n💾 Best approach: {best_model_name} with {best_accuracy*100:.2f}% accuracy")