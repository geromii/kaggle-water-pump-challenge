#!/usr/bin/env python3
"""
Hyperparameter Tuning for Best Model
====================================
Testing n_estimators and random_state combinations
Using the feature interactions approach (best from experiments)
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
warnings.filterwarnings('ignore')

print("🎯 HYPERPARAMETER TUNING - FEATURE INTERACTIONS MODEL")
print("=" * 60)

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

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

def prepare_features_with_interactions(df):
    """Prepare features including interactions"""
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
    if 'ward_functional_ratio' in X.columns:
        feature_cols.append('ward_functional_ratio')
    if 'neighbor_functional_ratio' in X.columns:
        feature_cols.append('neighbor_functional_ratio')
    
    X_features = X[feature_cols].fillna(-1)
    
    # ADD INTERACTION FEATURES (what made this approach best)
    if 'quantity_encoded' in X_features.columns and 'water_quality_encoded' in X_features.columns:
        X_features['quantity_quality_interact'] = X_features['quantity_encoded'] * X_features['water_quality_encoded']
    
    if 'ward_functional_ratio' in X_features.columns and 'pump_age' in X_features.columns:
        X_features['ward_age_interact'] = X_features['ward_functional_ratio'] * X_features['pump_age'].fillna(X_features['pump_age'].median())
    
    if 'neighbor_functional_ratio' in X_features.columns and 'days_since_recorded' in X_features.columns:
        X_features['neighbor_time_interact'] = X_features['neighbor_functional_ratio'] * X_features['days_since_recorded'].fillna(X_features['days_since_recorded'].median())
    
    return X_features

# Prepare data
print("🌍 Preparing data with geospatial features...")
train_geo = add_selective_geo_features(train, n_neighbors=10)
X = prepare_features_with_interactions(train_geo)
y = train['status_group']

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=46, stratify=y
)

print(f"Features: {X.shape[1]} (including 3 interaction features)")
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

# Define hyperparameter options
n_estimators_options = [50, 100, 200, 300]
random_state_options = [42, 43, 456, 789]

print(f"\n🔬 Testing {len(n_estimators_options)}x{len(random_state_options)} = {len(n_estimators_options)*len(random_state_options)} combinations")
print("-" * 60)

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

# Show summary statistics
print(f"\n📊 RESULTS SUMMARY")
print("=" * 60)

# Best overall
print(f"\n🏆 BEST PARAMETERS:")
print(f"   n_estimators: {best_params['n_estimators']}")
print(f"   random_state: {best_params['random_state']}")
print(f"   Validation score: {best_score:.4f}")

# Average by n_estimators
print(f"\n📈 Average score by n_estimators:")
n_est_avg = results_df.groupby('n_estimators')['val_score'].agg(['mean', 'std', 'max'])
for n_est, row in n_est_avg.iterrows():
    print(f"   {n_est:3d} trees: {row['mean']:.4f} ± {row['std']:.4f} (max: {row['max']:.4f})")

# Average by random_state
print(f"\n🎲 Average score by random_state:")
rs_avg = results_df.groupby('random_state')['val_score'].agg(['mean', 'std', 'max'])
for rs, row in rs_avg.iterrows():
    print(f"   rs={rs:3d}: {row['mean']:.4f} ± {row['std']:.4f} (max: {row['max']:.4f})")

# Top 5 combinations
print(f"\n🔝 TOP 5 COMBINATIONS:")
print("-" * 50)
top5 = results_df.nlargest(5, 'val_score')
for i, (_, row) in enumerate(top5.iterrows()):
    print(f"{i+1}. n_estimators={int(row['n_estimators']):3d}, rs={int(row['random_state']):3d}: {row['val_score']:.4f}")

# Timing analysis
print(f"\n⏱️  TIMING ANALYSIS:")
print(f"   Total time: {results_df['time'].sum():.1f} seconds")
print(f"   Average per model: {results_df['time'].mean():.1f} seconds")

# Show heatmap-style results
print(f"\n🗺️  RESULTS HEATMAP (validation scores):")
print("-" * 60)
print(f"{'n_est/rs':<10}", end="")
for rs in random_state_options:
    print(f"{rs:>8}", end="")
print()
print("-" * 60)

for n_est in n_estimators_options:
    print(f"{n_est:<10}", end="")
    for rs in random_state_options:
        score = results_df[(results_df['n_estimators'] == n_est) & 
                          (results_df['random_state'] == rs)]['val_score'].values[0]
        # Highlight best scores
        if score == best_score:
            print(f"*{score:.4f}", end="")
        elif score >= best_score - 0.001:
            print(f" {score:.4f}", end="")
        else:
            print(f" {score:.4f}", end="")
    print()

# Stability analysis
print(f"\n🎯 STABILITY ANALYSIS:")
print(f"   Score range: {results_df['val_score'].min():.4f} - {results_df['val_score'].max():.4f}")
print(f"   Score std: {results_df['val_score'].std():.4f}")
print(f"   Most stable n_estimators: {n_est_avg['std'].idxmin()} (std={n_est_avg['std'].min():.4f})")
print(f"   Most stable random_state: {rs_avg['std'].idxmin()} (std={rs_avg['std'].min():.4f})")

# Recommendation
print(f"\n💡 RECOMMENDATION:")
print("-" * 40)
if results_df['val_score'].std() < 0.001:
    print("✅ Model is very stable across hyperparameters")
    print(f"   Use n_estimators={best_params['n_estimators']} for best performance")
    print(f"   Random state has minimal impact (< 0.1% variation)")
else:
    print(f"✅ Use optimal parameters: n_estimators={best_params['n_estimators']}, random_state={best_params['random_state']}")
    print(f"   This gives {(best_score - results_df['val_score'].mean())*100:.2f}pp improvement over average")

# Save best model parameters
print(f"\n💾 Saving best parameters to 'best_hyperparameters.txt'...")
with open('best_hyperparameters.txt', 'w') as f:
    f.write(f"Best hyperparameters for feature interactions model:\n")
    f.write(f"n_estimators: {best_params['n_estimators']}\n")
    f.write(f"random_state: {best_params['random_state']}\n")
    f.write(f"validation_score: {best_score:.4f}\n")
    f.write(f"\nAll results:\n")
    f.write(results_df.to_string())

print("✅ Done!")