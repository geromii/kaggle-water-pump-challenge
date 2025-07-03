#!/usr/bin/env python3
"""
Full Dataset Geospatial Model - Fixed
====================================
- Processes train and test separately
- Uses full 59,400 row dataset
- Focuses on high-impact geospatial features
- Creates final improved model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def create_simple_geospatial_features(df, reference_df=None, n_neighbors=20):
    """Create geospatial features with simpler approach"""
    
    print(f"🌍 Creating geospatial features for {len(df):,} rows...")
    
    data = df.copy()
    
    # Remove rows with missing coordinates
    valid_coords = (data['latitude'] != 0) & (data['longitude'] != 0) & \
                   data['latitude'].notna() & data['longitude'].notna()
    
    print(f"📍 Valid coordinates: {valid_coords.sum():,} / {len(data):,} ({valid_coords.mean()*100:.1f}%)")
    
    if valid_coords.sum() < n_neighbors:
        print(f"⚠️  Not enough valid coordinates for neighbor analysis")
        return data
    
    # Use reference data for neighbor analysis if provided (for test set)
    if reference_df is not None:
        print("🔗 Using training data as reference for test neighbors...")
        ref_valid = (reference_df['latitude'] != 0) & (reference_df['longitude'] != 0) & \
                   reference_df['latitude'].notna() & reference_df['longitude'].notna()
        ref_coords = reference_df.loc[ref_valid, ['latitude', 'longitude']].values
        
        # Build neighbor index on reference data
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', 
                               metric='haversine', n_jobs=-1)
        nbrs.fit(np.radians(ref_coords))
        
        # Find neighbors for test data
        test_coords = data.loc[valid_coords, ['latitude', 'longitude']].values
        distances, indices = nbrs.kneighbors(np.radians(test_coords))
        distances = distances * 6371  # Convert to km
        
        # Get reference status for neighbor analysis
        ref_status = reference_df.loc[ref_valid, 'status_group'].fillna('unknown')
        
    else:
        print("🔍 Building self-referencing neighbors...")
        coords = data.loc[valid_coords, ['latitude', 'longitude']].values
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                               metric='haversine', n_jobs=-1)
        nbrs.fit(np.radians(coords))
        distances, indices = nbrs.kneighbors(np.radians(coords))
        distances = distances * 6371
        
        # Remove self from neighbors
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    valid_indices = data.index[valid_coords].tolist()
    
    # Initialize features
    neighbor_features = {}
    
    # 1. DISTANCE-BASED FEATURES
    neighbor_features['nearest_neighbor_distance'] = distances[:, 0]
    neighbor_features['avg_5_neighbor_distance'] = np.mean(distances[:, :5], axis=1)
    neighbor_features['avg_neighbor_distance'] = np.mean(distances, axis=1)
    neighbor_features['neighbor_density'] = 1.0 / (neighbor_features['avg_5_neighbor_distance'] + 0.1)
    
    # 2. STATUS-BASED FEATURES (only if we have status data)
    if reference_df is not None and 'status_group' in reference_df.columns:
        # For test data using training reference
        status_encoder = LabelEncoder()
        status_encoded = status_encoder.fit_transform(ref_status)
        
        neighbor_functional_ratios = []
        for i, neighbor_idx in enumerate(indices):
            valid_neighbor_positions = [idx for idx in neighbor_idx if idx < len(status_encoded)]
            if valid_neighbor_positions:
                neighbor_statuses = status_encoded[valid_neighbor_positions]
                functional_idx = np.where(status_encoder.classes_ == 'functional')[0]
                if len(functional_idx) > 0:
                    functional_ratio = (neighbor_statuses == functional_idx[0]).mean()
                else:
                    functional_ratio = 0.5
            else:
                functional_ratio = 0.5
            neighbor_functional_ratios.append(functional_ratio)
        
        neighbor_features['neighbor_functional_ratio'] = np.array(neighbor_functional_ratios)
    
    elif 'status_group' in data.columns:
        # For training data
        valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
        status_encoder = LabelEncoder()
        status_encoded = status_encoder.fit_transform(valid_status)
        
        neighbor_functional_ratios = []
        for i, neighbor_idx in enumerate(indices):
            valid_neighbor_positions = [idx for idx in neighbor_idx if idx < len(status_encoded)]
            if valid_neighbor_positions:
                neighbor_statuses = status_encoded[valid_neighbor_positions]
                functional_idx = np.where(status_encoder.classes_ == 'functional')[0]
                if len(functional_idx) > 0:
                    functional_ratio = (neighbor_statuses == functional_idx[0]).mean()
                else:
                    functional_ratio = 0.5
            else:
                functional_ratio = 0.5
            neighbor_functional_ratios.append(functional_ratio)
        
        neighbor_features['neighbor_functional_ratio'] = np.array(neighbor_functional_ratios)
    
    # 3. INSTALLER/FUNDER CLUSTERING (using same logic)
    if reference_df is not None:
        # Use reference data for clustering analysis
        ref_data = reference_df.loc[ref_valid]
        for col in ['installer', 'funder']:
            if col in data.columns and col in reference_df.columns:
                ref_values = ref_data[col].fillna('unknown').values
                test_values = data.loc[valid_indices, col].fillna('unknown').values
                
                same_ratios = []
                for i, neighbor_idx in enumerate(indices):
                    if i < len(test_values):
                        current_value = test_values[i]
                        valid_positions = [idx for idx in neighbor_idx if idx < len(ref_values)]
                        if valid_positions:
                            neighbor_values = ref_values[valid_positions]
                            same_ratio = (neighbor_values == current_value).mean()
                        else:
                            same_ratio = 0.0
                        same_ratios.append(same_ratio)
                
                neighbor_features[f'{col}_spatial_clustering'] = np.array(same_ratios)
    else:
        # Self-referencing for training data
        for col in ['installer', 'funder']:
            if col in data.columns:
                col_values = data.loc[valid_indices, col].fillna('unknown').values
                
                same_ratios = []
                for i, neighbor_idx in enumerate(indices):
                    if i < len(col_values):
                        current_value = col_values[i]
                        valid_positions = [idx for idx in neighbor_idx if idx < len(col_values)]
                        if valid_positions:
                            neighbor_values = col_values[valid_positions]
                            same_ratio = (neighbor_values == current_value).mean()
                        else:
                            same_ratio = 0.0
                        same_ratios.append(same_ratio)
                
                neighbor_features[f'{col}_spatial_clustering'] = np.array(same_ratios)
    
    # Map features back to full dataset
    for feature_name, feature_values in neighbor_features.items():
        data[feature_name] = np.nan
        data.loc[valid_indices, feature_name] = feature_values
    
    # Fill missing values
    for feature_name in neighbor_features.keys():
        if data[feature_name].dtype in ['float64', 'int64']:
            data[feature_name] = data[feature_name].fillna(data[feature_name].median())
        else:
            data[feature_name] = data[feature_name].fillna(-1)
    
    # Add ward-level statistics (only for training data with status)
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
        # Use training data ward statistics for test data
        ward_stats = reference_df.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(ward_stats['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    print(f"✅ Created {len(neighbor_features)} neighbor features")
    ward_features = 2 if 'ward_pump_count' in data.columns else 0
    print(f"✅ Created {ward_features} ward features")
    
    return data

print("🚀 FULL DATASET GEOSPATIAL MODEL - FIXED")
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
train_geo = create_simple_geospatial_features(train, n_neighbors=15)

def prepare_features(df):
    """Prepare features for modeling (baseline + geospatial)"""
    X = df.copy()
    
    print(f"🔧 Engineering features for {len(X):,} rows...")
    
    # Original engineered features (simplified)
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
    
    # Key categorical features only (most important ones)
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
    
    # Key encoded categorical features
    encoded_cols = [col+'_encoded' for col in key_categorical_cols]
    for col in encoded_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Geospatial features
    geo_cols = ['nearest_neighbor_distance', 'avg_5_neighbor_distance', 'avg_neighbor_distance',
               'neighbor_density', 'neighbor_functional_ratio', 'installer_spatial_clustering',
               'funder_spatial_clustering', 'ward_pump_count', 'ward_functional_ratio']
    
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

# Train model
print(f"\n🤖 TRAINING GEOSPATIAL MODEL...")
print("-" * 40)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("Training on full training dataset...")
model.fit(X_train_full, y_train_full)

# Cross-validation
print(f"\n🔄 CROSS-VALIDATION (5-fold)...")
print("-" * 40)
cv_scores = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
print(f"\n🔍 TOP 15 FEATURE IMPORTANCE:")
print("-" * 50)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
    if any(geo_term in row['feature'] for geo_term in ['neighbor', 'cluster', 'spatial', 'ward']):
        emoji = "🌍"
    else:
        emoji = "📊"
    
    print(f"{i+1:2d}. {emoji} {row['feature']:<35} {row['importance']:.4f}")

# Create test features
print(f"\n🔮 CREATING TEST PREDICTIONS...")
print("-" * 40)

print("Creating geospatial features for test data...")
test_geo = create_simple_geospatial_features(test_features, reference_df=train_geo, n_neighbors=15)

print("Preparing test features...")
X_test, _ = prepare_features(test_geo)

print("Making predictions...")
test_predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'id': test_features['id'],
    'status_group': test_predictions
})

submission_file = 'improved_geospatial_submission.csv'
submission.to_csv(submission_file, index=False)

print(f"\n🎯 FINAL RESULTS:")
print("=" * 50)
print(f"✅ Cross-validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"✅ Submission file: {submission_file}")
print(f"✅ Predictions: {len(submission):,} rows")

# Show prediction distribution
pred_counts = submission['status_group'].value_counts()
print(f"\n📈 PREDICTION DISTRIBUTION:")
for status, count in pred_counts.items():
    pct = (count / len(submission)) * 100
    print(f"{status:<25} {count:6,} ({pct:5.1f}%)")

print(f"\n🚀 Expected improvement over baseline: ~{(cv_scores.mean() - 0.77)*100:.1f} percentage points")
print(f"🎊 Ready for Kaggle submission!")