#!/usr/bin/env python3
"""
Fast Model Comparison - Streamlined Testing
==========================================
- Focus on Random Forest (fastest and most reliable)
- Test only the most promising feature combinations
- Single CV fold for quick results
- Compare baseline vs best enhancements
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

def add_selective_geo_features(df, reference_df=None, n_neighbors=10):
    """Add only the most valuable geospatial features"""
    print(f"🌍 Adding selective geospatial features...")
    
    data = df.copy()
    
    # 1. WARD-LEVEL STATISTICS (highest impact)
    if 'status_group' in data.columns and 'ward' in data.columns:
        ward_stats = data.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(data['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    elif 'ward' in data.columns and reference_df is not None and 'ward' in reference_df.columns:
        ward_stats = reference_df.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(ward_stats['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    # 2. NEIGHBOR FUNCTIONAL RATIO (second highest impact)
    valid_coords = (data['latitude'] != 0) & (data['longitude'] != 0) & \
                   data['latitude'].notna() & data['longitude'].notna()
    
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
    
    return data

def prepare_features(df, include_gpt=True, include_geo=True):
    """Prepare features for modeling"""
    X = df.copy()
    
    # Missing value indicators (proven high impact)
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
    
    # Add geospatial features
    if include_geo:
        if 'ward_functional_ratio' in X.columns:
            feature_cols.append('ward_functional_ratio')
        if 'neighbor_functional_ratio' in X.columns:
            feature_cols.append('neighbor_functional_ratio')
    
    # Add GPT features
    if include_gpt:
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        for col in gpt_cols:
            if col in X.columns and X[col].notna().any():
                X[col] = X[col].fillna(3.0)  # Default middle value
                feature_cols.append(col)
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, feature_cols

print("🚀 FAST MODEL COMPARISON")
print("=" * 60)

# Load datasets
print("📊 Loading datasets...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Load GPT features
print("🤖 Loading GPT features...")
try:
    gpt_features = pd.read_csv('gpt_features_complete.csv')
    print(f"✅ GPT features loaded: {len(gpt_features):,} rows")
except FileNotFoundError:
    print("⚠️  gpt_features_complete.csv not found, trying progress file...")
    try:
        gpt_features = pd.read_csv('gpt_features_progress.csv')
        print(f"✅ GPT progress features loaded: {len(gpt_features):,} rows")
    except FileNotFoundError:
        print("⚠️  No GPT features found, trying SQLite...")
        import sqlite3
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
            print("❌ No GPT features found, testing baseline only")
            gpt_features = pd.DataFrame()

# Merge datasets
if len(gpt_features) > 0:
    train_with_gpt = train.merge(gpt_features, on='id', how='left')
    gpt_coverage = train_with_gpt['gpt_funder_org_type'].notna().sum()
    print(f"📊 GPT coverage: {gpt_coverage:,}/{len(train):,} ({gpt_coverage/len(train)*100:.1f}%)")
else:
    train_with_gpt = train
    gpt_coverage = 0

# Add geospatial features
print(f"\n🌍 Adding geospatial features...")
start_time = time.time()
train_final = add_selective_geo_features(train_with_gpt, n_neighbors=10)
print(f"⏱️  Geospatial features added in {time.time() - start_time:.1f} seconds")

# Define test configurations (only the most promising)
test_configs = [
    ('1_baseline', False, False),
    ('2_baseline_geo', False, True),
]

if gpt_coverage > 0:
    test_configs.extend([
        ('3_baseline_gpt', True, False),
        ('4_all_features', True, True),
    ])

# Single train/test split for speed
print(f"\n📊 Splitting data...")
y = train_final['status_group']
X_train_df, X_test_df, y_train, y_test = train_test_split(
    train_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train_df):,} | Test: {len(X_test_df):,}")

# Test each configuration
print(f"\n🧪 TESTING {len(test_configs)} CONFIGURATIONS")
print("-" * 60)

results = []

for config_name, use_gpt, use_geo in test_configs:
    print(f"\n🔬 Testing: {config_name}")
    start_time = time.time()
    
    # Prepare features
    X_train, train_features = prepare_features(X_train_df, include_gpt=use_gpt, include_geo=use_geo)
    X_test, _ = prepare_features(X_test_df, include_gpt=use_gpt, include_geo=use_geo)
    
    print(f"   Features: {X_train.shape[1]}")
    gpt_count = len([f for f in train_features if f.startswith('gpt_')])
    geo_count = len([f for f in train_features if 'functional_ratio' in f])
    print(f"   GPT: {gpt_count} | Geo: {geo_count}")
    
    # Train model (with overfitting prevention)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,  # Limit depth to prevent overfitting
        min_samples_split=10,  # Require more samples to split
        min_samples_leaf=2,  # Require more samples in leaves
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    elapsed = time.time() - start_time
    
    result = {
        'name': config_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'overfit_gap': train_acc - test_acc,
        'n_features': X_train.shape[1],
        'time': elapsed,
        'use_gpt': use_gpt,
        'use_geo': use_geo,
        'model': model,
        'features': train_features
    }
    results.append(result)
    
    print(f"   Train: {train_acc:.4f} | Test: {test_acc:.4f} | Gap: {train_acc - test_acc:.4f}")
    print(f"   Time: {elapsed:.1f}s")

# Results summary
print(f"\n🏆 RESULTS SUMMARY")
print("=" * 60)

# Sort by test accuracy
results.sort(key=lambda x: x['test_acc'], reverse=True)

print(f"{'Config':<20} {'Train':<8} {'Test':<8} {'Overfit':<8} {'Features':<10} {'Time':<8}")
print("-" * 60)

baseline_test_acc = None
for i, result in enumerate(results):
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
    print(f"{emoji} {result['name']:<18} {result['train_acc']:.4f}   {result['test_acc']:.4f}   "
          f"{result['overfit_gap']:.4f}   {result['n_features']:<10} {result['time']:.1f}s")
    
    if result['name'] == '1_baseline':
        baseline_test_acc = result['test_acc']

# Improvement analysis
if baseline_test_acc:
    print(f"\n📈 IMPROVEMENTS OVER BASELINE ({baseline_test_acc:.4f})")
    print("-" * 40)
    for result in results:
        if result['name'] != '1_baseline':
            improvement = result['test_acc'] - baseline_test_acc
            improvement_pct = (improvement / baseline_test_acc) * 100
            print(f"{result['name']:<20} {improvement:+.4f} ({improvement_pct:+.2f}%)")

# Feature importance for best model
best_result = results[0]
print(f"\n🔍 TOP FEATURES - {best_result['name']}")
print("-" * 50)

model = best_result['model']
features = best_result['features']
importances = model.feature_importances_

feature_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_df.head(20).iterrows()):
    if row['feature'].startswith('gpt_'):
        emoji = "🤖"
    elif 'functional_ratio' in row['feature']:
        emoji = "🌍"
    elif 'MISSING' in row['feature']:
        emoji = "❓"
    else:
        emoji = "📊"
    
    print(f"{i+1:2d}. {emoji} {row['feature']:<35} {row['importance']:.4f}")

# Final recommendation
print(f"\n💡 RECOMMENDATION")
print("-" * 40)

if best_result['test_acc'] > baseline_test_acc + 0.01:
    print(f"✅ Use {best_result['name']} configuration")
    print(f"   Achieves {best_result['test_acc']:.4f} accuracy")
    print(f"   {(best_result['test_acc'] - baseline_test_acc)*100:.2f}pp improvement")
elif best_result['test_acc'] > baseline_test_acc + 0.005:
    print(f"✅ Moderate improvement with {best_result['name']}")
    print(f"   Consider complexity vs gain trade-off")
else:
    print(f"⚠️  Minimal improvement - consider using baseline")
    print(f"   Added complexity may not be worth it")

print(f"\n🎯 Best test accuracy: {best_result['test_acc']:.4f}")
print(f"📊 Overfit gap: {best_result['overfit_gap']:.4f} (lower is better)")