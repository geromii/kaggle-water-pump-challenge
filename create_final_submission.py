#!/usr/bin/env python3
"""
Create Final Submission
======================
- Uses the best model configuration from testing
- Trains on full dataset
- Creates submission file
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def add_selective_geo_features(df, reference_df=None, n_neighbors=10):
    """Add only the most valuable geospatial features"""
    print(f"🌍 Adding geospatial features for {len(df):,} rows...")
    
    data = df.copy()
    
    # 1. WARD-LEVEL STATISTICS
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
    
    # 2. NEIGHBOR FUNCTIONAL RATIO
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

def prepare_features(df, le_dict=None, include_gpt=True, include_geo=True):
    """Prepare features for modeling"""
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
    
    # Handle label encoding
    if le_dict is None:
        le_dict = {}
        for col in key_categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
    else:
        # Use existing encoders for test data
        for col in key_categorical_cols:
            if col in X.columns and col in le_dict:
                # Handle unseen categories
                le = le_dict[col]
                col_values = X[col].astype(str)
                # Map unseen values to a default
                col_values = col_values.map(lambda x: x if x in le.classes_ else 'unknown')
                X[col+'_encoded'] = le.transform(col_values)
    
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
                X[col] = X[col].fillna(3.0)
                feature_cols.append(col)
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, feature_cols, le_dict

print("🚀 CREATING FINAL SUBMISSION")
print("=" * 60)

# Load datasets
print("📊 Loading datasets...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
test_features = pd.read_csv('ds19-predictive-modeling-challenge/test_features.csv')
train = pd.merge(train_features, train_labels, on='id')

print(f"Train: {len(train):,} | Test: {len(test_features):,}")

# Load GPT features
print("\n🤖 Loading GPT features...")
gpt_features = None
try:
    gpt_features = pd.read_csv('gpt_features_complete.csv')
    print(f"✅ GPT features loaded: {len(gpt_features):,} rows")
except FileNotFoundError:
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
        print("❌ No GPT features found")

# Merge GPT features if available
use_gpt = False
if gpt_features is not None and len(gpt_features) > 0:
    train = train.merge(gpt_features, on='id', how='left')
    gpt_coverage = train['gpt_funder_org_type'].notna().sum()
    print(f"📊 GPT coverage: {gpt_coverage:,}/{len(train):,} ({gpt_coverage/len(train)*100:.1f}%)")
    use_gpt = gpt_coverage > 0

# Add geospatial features
print("\n🌍 Adding geospatial features...")
train_geo = add_selective_geo_features(train, n_neighbors=10)
test_geo = add_selective_geo_features(test_features, reference_df=train_geo, n_neighbors=10)

# Prepare features
print("\n🔧 Preparing features...")
X_train, feature_cols, le_dict = prepare_features(train_geo, include_gpt=use_gpt, include_geo=True)
X_test, _, _ = prepare_features(test_geo, le_dict=le_dict, include_gpt=False, include_geo=True)

print(f"Train features: {X_train.shape}")
print(f"Test features: {X_test.shape}")

# Train final model
print("\n🤖 Training final model...")
model = RandomForestClassifier(
    n_estimators=200,  # More trees for final model
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

y_train = train_geo['status_group']
model.fit(X_train, y_train)

# Make predictions
print("\n🎯 Making predictions...")
predictions = model.predict(X_test)

# Create submission
print("\n📝 Creating submission file...")
submission = pd.DataFrame({
    'id': test_features['id'],
    'status_group': predictions
})

# Save submission
filename = 'submission_enhanced.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ Submission saved: {filename}")
print(f"📊 Predictions distribution:")
print(submission['status_group'].value_counts())

# Feature importance summary
print("\n🔍 TOP 10 FEATURES:")
importances = model.feature_importances_
feature_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_df.head(10).iterrows()):
    if row['feature'].startswith('gpt_'):
        emoji = "🤖"
    elif 'functional_ratio' in row['feature']:
        emoji = "🌍"
    elif 'MISSING' in row['feature']:
        emoji = "❓"
    else:
        emoji = "📊"
    
    print(f"{i+1:2d}. {emoji} {row['feature']:<35} {row['importance']:.4f}")

print("\n🎉 SUBMISSION READY!")
print(f"📤 Upload {filename} to Kaggle")