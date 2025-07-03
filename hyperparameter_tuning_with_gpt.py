#!/usr/bin/env python3
"""
Hyperparameter Tuning WITH GPT Features
=======================================
Testing n_estimators and random_state combinations
Using feature interactions + GPT features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import warnings
import time
import sqlite3
warnings.filterwarnings('ignore')

print("🎯 HYPERPARAMETER TUNING - FEATURE INTERACTIONS + GPT MODEL")
print("=" * 70)

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Load GPT features
print("🤖 Loading GPT features...")
try:
    gpt_features = pd.read_csv('gpt_features_complete.csv')
    print(f"✅ GPT features loaded: {len(gpt_features):,} rows")
except FileNotFoundError:
    try:
        conn = sqlite3.connect('gpt_features_progress.db')
        gpt_features = pd.read_sql_query('''
            SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
                   gpt_scheme_pattern, gpt_text_language, gpt_coordination 
            FROM gpt_features ORDER BY id
        ''', conn)
        conn.close()
        print(f"✅ GPT features from SQLite: {len(gpt_features):,} rows")
    except:
        print("❌ No GPT features found")
        gpt_features = pd.DataFrame()

# Merge GPT features
if len(gpt_features) > 0:
    train = train.merge(gpt_features, on='id', how='left')
    gpt_coverage = train['gpt_funder_org_type'].notna().sum()
    print(f"📊 GPT coverage: {gpt_coverage:,}/{len(train):,} ({gpt_coverage/len(train)*100:.1f}%)")
else:
    print("⚠️  No GPT features available - proceeding without them")

# Add geospatial features function
def add_selective_geo_features(df, reference_df=None, n_neighbors=10):
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
    
    return data

def prepare_features_with_interactions_and_gpt(df):
    """Prepare features including interactions and GPT features"""
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
    
    # Geo features if present
    geo_cols = ['ward_functional_ratio', 'neighbor_functional_ratio']
    feature_cols.extend([col for col in geo_cols if col in X.columns])
    
    # GPT features if present
    gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
               'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
    gpt_features_added = 0
    for col in gpt_cols:
        if col in X.columns and X[col].notna().any():
            X[col] = X[col].fillna(3.0)  # Default middle value
            feature_cols.append(col)
            gpt_features_added += 1
    
    print(f"✅ Added {gpt_features_added} GPT features")
    
    X_features = X[feature_cols].fillna(-1)
    
    # ADD INTERACTION FEATURES
    interaction_count = 0
    
    if 'quantity_encoded' in X_features.columns and 'water_quality_encoded' in X_features.columns:
        X_features['quantity_quality_interact'] = X_features['quantity_encoded'] * X_features['water_quality_encoded']
        interaction_count += 1
    
    if 'ward_functional_ratio' in X_features.columns and 'pump_age' in X_features.columns:
        X_features['ward_age_interact'] = X_features['ward_functional_ratio'] * X_features['pump_age'].fillna(X_features['pump_age'].median())
        interaction_count += 1
    
    if 'neighbor_functional_ratio' in X_features.columns and 'days_since_recorded' in X_features.columns:
        X_features['neighbor_time_interact'] = X_features['neighbor_functional_ratio'] * X_features['days_since_recorded'].fillna(X_features['days_since_recorded'].median())
        interaction_count += 1
    
    # NEW: GPT interactions if available
    if gpt_features_added > 0:
        if 'gpt_installer_quality' in X_features.columns and 'pump_age' in X_features.columns:
            X_features['gpt_installer_age_interact'] = X_features['gpt_installer_quality'] * X_features['pump_age'].fillna(X_features['pump_age'].median())
            interaction_count += 1
        
        if 'gpt_coordination' in X_features.columns and 'ward_functional_ratio' in X_features.columns:
            X_features['gpt_coord_ward_interact'] = X_features['gpt_coordination'] * X_features['ward_functional_ratio']
            interaction_count += 1
    
    print(f"✅ Added {interaction_count} interaction features")
    
    return X_features

# Prepare data
print("\n🌍 Preparing data with geospatial + GPT features...")
train_geo = add_selective_geo_features(train, n_neighbors=10)
X = prepare_features_with_interactions_and_gpt(train_geo)
y = train['status_group']

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=44, stratify=y
)

print(f"Total features: {X.shape[1]}")
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

# Define hyperparameter options - focus on the best ranges from previous test
n_estimators_options = [100, 150, 250, 300]
random_state_options = [42, 123, 789, 999]

print(f"\n🔬 Testing {len(n_estimators_options)}x{len(random_state_options)} = {len(n_estimators_options)*len(random_state_options)} combinations")
print("-" * 70)

# Store results
results = []
best_score = 0
best_params = {}

# Grid search
for n_est in n_estimators_options:
    print(f"\nn_estimators = {n_est}")
    print("  ", end="")
    
    for rs in random_state_options:
        start_time = time.time()
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_est,
            random_state=rs,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_score = accuracy_score(y_val, val_pred)
        
        # Store result
        results.append({
            'n_estimators': n_est,
            'random_state': rs,
            'val_score': val_score,
            'time': time.time() - start_time
        })
        
        # Update best
        if val_score > best_score:
            best_score = val_score
            best_params = {'n_estimators': n_est, 'random_state': rs}
        
        print(f"rs={rs}: {val_score:.4f} ", end="", flush=True)
    
    print()  # New line after each n_estimators row

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Show results
print(f"\n📊 RESULTS SUMMARY")
print("=" * 70)

# Best overall
print(f"\n🏆 BEST PARAMETERS:")
print(f"   n_estimators: {best_params['n_estimators']}")
print(f"   random_state: {best_params['random_state']}")
print(f"   Validation score: {best_score:.4f}")

# Compare with previous best (without GPT)
previous_best = 0.8288  # From previous run
improvement = best_score - previous_best
print(f"\n📈 IMPROVEMENT ANALYSIS:")
print(f"   Previous best (no GPT): {previous_best:.4f}")
print(f"   Current best (with GPT): {best_score:.4f}")
print(f"   GPT improvement: {improvement:+.4f} ({improvement*100:+.2f}pp)")

# Top 5 combinations
print(f"\n🔝 TOP 5 COMBINATIONS:")
print("-" * 50)
top5 = results_df.nlargest(5, 'val_score')
for i, (_, row) in enumerate(top5.iterrows()):
    print(f"{i+1}. n_estimators={int(row['n_estimators']):3d}, rs={int(row['random_state']):3d}: {row['val_score']:.4f}")

# Save best model parameters
print(f"\n💾 Saving best parameters...")
with open('best_hyperparameters_with_gpt.txt', 'w') as f:
    f.write(f"Best hyperparameters for feature interactions + GPT model:\n")
    f.write(f"n_estimators: {best_params['n_estimators']}\n")
    f.write(f"random_state: {best_params['random_state']}\n")
    f.write(f"validation_score: {best_score:.4f}\n")
    f.write(f"improvement_over_no_gpt: {improvement:+.4f}\n")
    f.write(f"total_features: {X.shape[1]}\n")

if improvement > 0.002:
    print(f"\n🎉 GPT FEATURES PROVIDE MEANINGFUL IMPROVEMENT!")
    print(f"   Recommend using this configuration for final submission")
elif improvement > 0:
    print(f"\n✅ GPT features provide small improvement")
    print(f"   May be worth using if computational cost is acceptable")
else:
    print(f"\n⚠️  GPT features don't improve performance")
    print(f"   Recommend sticking with geospatial-only model")

print("✅ Done!")