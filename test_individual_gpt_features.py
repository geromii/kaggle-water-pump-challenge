#!/usr/bin/env python3
"""
Test Individual GPT Features
============================
Test each of the 6 GPT features one by one to see which helps most
Using the best baseline: feature interactions + geospatial (0.8288)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import warnings
import sqlite3
warnings.filterwarnings('ignore')

print("🧪 TESTING INDIVIDUAL GPT FEATURES")
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
gpt_coverage = train['gpt_funder_org_type'].notna().sum()
print(f"📊 GPT coverage: {gpt_coverage:,}/{len(train):,} ({gpt_coverage/len(train)*100:.1f}%)")

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
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                               metric='haversine', n_jobs=-1)
        nbrs.fit(np.radians(coords))
        _, indices = nbrs.kneighbors(np.radians(coords))
        indices = indices[:, 1:]
        
        valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
        
        neighbor_functional_ratios = []
        for i, neighbor_idx in enumerate(indices):
            neighbor_statuses = valid_status.iloc[neighbor_idx]
            functional_ratio = (neighbor_statuses == 'functional').mean()
            neighbor_functional_ratios.append(functional_ratio)
        
        data['neighbor_functional_ratio'] = np.nan
        data.loc[valid_coords, 'neighbor_functional_ratio'] = neighbor_functional_ratios
        data['neighbor_functional_ratio'] = data['neighbor_functional_ratio'].fillna(0.5)
    
    return data

def prepare_features_with_selective_gpt(df, gpt_feature=None):
    """Prepare features with optional single GPT feature"""
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
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in X.columns:
            X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
    
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
    
    # Single GPT feature if specified
    if gpt_feature and gpt_feature in X.columns and X[gpt_feature].notna().any():
        X[gpt_feature] = X[gpt_feature].fillna(3.0)  # Default middle value
        feature_cols.append(gpt_feature)
    
    X_features = X[feature_cols].fillna(-1)
    
    # ADD INTERACTION FEATURES (what made this approach best)
    if 'quantity_encoded' in X_features.columns and 'water_quality_encoded' in X_features.columns:
        X_features['quantity_quality_interact'] = X_features['quantity_encoded'] * X_features['water_quality_encoded']
    
    if 'ward_functional_ratio' in X_features.columns and 'pump_age' in X_features.columns:
        X_features['ward_age_interact'] = X_features['ward_functional_ratio'] * X_features['pump_age'].fillna(X_features['pump_age'].median())
    
    if 'neighbor_functional_ratio' in X_features.columns and 'days_since_recorded' in X_features.columns:
        X_features['neighbor_time_interact'] = X_features['neighbor_functional_ratio'] * X_features['days_since_recorded'].fillna(X_features['days_since_recorded'].median())
    
    # Add GPT interaction if we have a GPT feature
    if gpt_feature and gpt_feature in X_features.columns:
        if 'pump_age' in X_features.columns:
            X_features[f'{gpt_feature}_age_interact'] = X_features[gpt_feature] * X_features['pump_age'].fillna(X_features['pump_age'].median())
    
    return X_features

# Prepare data
print("\n🌍 Adding geospatial features...")
train_geo = add_selective_geo_features(train, n_neighbors=10)

# Prepare baseline (no GPT)
X_baseline = prepare_features_with_selective_gpt(train_geo, gpt_feature=None)
y = train['status_group']

# Split data
X_train_base, X_val_base, y_train, y_val = train_test_split(
    X_baseline, y, test_size=0.1, random_state=44, stratify=y
)

print(f"Baseline features: {X_baseline.shape[1]}")
print(f"Train: {X_train_base.shape[0]} | Val: {X_val_base.shape[0]}")

# Test baseline model first
print(f"\n📊 BASELINE MODEL (no GPT)")
print("-" * 40)

# Use best hyperparameters from previous test
baseline_model = RandomForestClassifier(n_estimators=300, random_state=456, n_jobs=-1)
baseline_model.fit(X_train_base, y_train)
baseline_pred = baseline_model.predict(X_val_base)
baseline_score = accuracy_score(y_val, baseline_pred)

print(f"Baseline score: {baseline_score:.4f}")

# Test each GPT feature individually
gpt_features = [
    'gpt_funder_org_type',
    'gpt_installer_quality', 
    'gpt_location_type',
    'gpt_scheme_pattern',
    'gpt_text_language',
    'gpt_coordination'
]

gpt_descriptions = {
    'gpt_funder_org_type': 'Organization type (gov/NGO/private/religious)',
    'gpt_installer_quality': 'Data quality/completeness of installer field',
    'gpt_location_type': 'Institution served (school/health/village/etc)',
    'gpt_scheme_pattern': 'Management scheme pattern analysis', 
    'gpt_text_language': 'Language/naming pattern (English/Swahili/mixed)',
    'gpt_coordination': 'Geographic clustering/coordination level'
}

print(f"\n🧪 TESTING INDIVIDUAL GPT FEATURES")
print("=" * 60)

results = []

for gpt_feature in gpt_features:
    print(f"\n🔬 Testing: {gpt_feature}")
    print(f"   {gpt_descriptions[gpt_feature]}")
    
    # Check data availability
    available_data = train_geo[gpt_feature].notna().sum()
    print(f"   Available data: {available_data:,}/{len(train_geo):,} ({available_data/len(train_geo)*100:.1f}%)")
    
    if available_data < 1000:
        print(f"   ⚠️  Too little data, skipping...")
        continue
    
    # Prepare features with this GPT feature
    X_with_gpt = prepare_features_with_selective_gpt(train_geo, gpt_feature=gpt_feature)
    
    # Split using same indices as baseline
    X_train_gpt = X_with_gpt.iloc[X_train_base.index]
    X_val_gpt = X_with_gpt.iloc[X_val_base.index]
    
    print(f"   Features: {X_with_gpt.shape[1]} (+{X_with_gpt.shape[1] - X_baseline.shape[1]} vs baseline)")
    
    # Train model
    model = RandomForestClassifier(n_estimators=300, random_state=456, n_jobs=-1)
    model.fit(X_train_gpt, y_train)
    
    # Evaluate
    pred = model.predict(X_val_gpt)
    score = accuracy_score(y_val, pred)
    improvement = score - baseline_score
    
    print(f"   Score: {score:.4f} ({improvement:+.4f})")
    
    # Check feature importance of the GPT feature
    feature_names = X_with_gpt.columns.tolist()
    if gpt_feature in feature_names:
        gpt_importance = model.feature_importances_[feature_names.index(gpt_feature)]
        print(f"   Feature importance: {gpt_importance:.4f}")
        
        # Check interaction feature importance if it exists
        interact_feature = f'{gpt_feature}_age_interact'
        if interact_feature in feature_names:
            interact_importance = model.feature_importances_[feature_names.index(interact_feature)]
            print(f"   Interaction importance: {interact_importance:.4f}")
    
    results.append({
        'feature': gpt_feature,
        'description': gpt_descriptions[gpt_feature],
        'score': score,
        'improvement': improvement,
        'available_data': available_data,
        'feature_importance': gpt_importance if gpt_feature in feature_names else 0
    })

# Sort results by improvement
results.sort(key=lambda x: x['improvement'], reverse=True)

print(f"\n🏆 RESULTS SUMMARY")
print("=" * 60)
print(f"Baseline score: {baseline_score:.4f}")
print(f"\nGPT Feature Rankings:")
print("-" * 60)

for i, result in enumerate(results):
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
    improvement_color = "✅" if result['improvement'] > 0.001 else "⚠️" if result['improvement'] > 0 else "❌"
    
    print(f"{emoji} {result['feature']:<25} {result['score']:.4f} ({result['improvement']:+.4f}) {improvement_color}")
    print(f"    {result['description']}")
    print(f"    Data: {result['available_data']:,} rows, Importance: {result['feature_importance']:.4f}")
    print()

# Find the best single GPT feature
if results and results[0]['improvement'] > 0.001:
    best_gpt = results[0]
    print(f"🎯 RECOMMENDATION:")
    print("-" * 40)
    print(f"✅ Use {best_gpt['feature']} as single GPT feature")
    print(f"   Improvement: {best_gpt['improvement']*100:+.2f} percentage points")
    print(f"   Final score: {best_gpt['score']:.4f}")
    
    # Save best single GPT feature
    with open('best_single_gpt_feature.txt', 'w') as f:
        f.write(f"Best single GPT feature: {best_gpt['feature']}\n")
        f.write(f"Improvement: {best_gpt['improvement']:+.4f}\n")
        f.write(f"Score: {best_gpt['score']:.4f}\n")
        f.write(f"Description: {best_gpt['description']}\n")
    
elif results and max(r['improvement'] for r in results) > 0:
    print(f"🤔 MARGINAL IMPROVEMENT:")
    print("-" * 40)
    print(f"Best GPT feature provides minimal improvement")
    print(f"Consider sticking with baseline model")
else:
    print(f"❌ NO IMPROVEMENT:")
    print("-" * 40)
    print(f"None of the GPT features improve performance")
    print(f"Recommend using baseline model without GPT features")

print("\n✅ Individual GPT feature testing complete!")