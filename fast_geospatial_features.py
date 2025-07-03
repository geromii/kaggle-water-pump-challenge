#!/usr/bin/env python3
"""
Fast Geospatial Feature Engineering
===================================
- Optimized for speed using grid-based clustering
- Vectorized operations instead of loops
- Much faster than K-means for large datasets
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_fast_geospatial_features(df, n_neighbors=10, grid_size=0.01):
    """Create geospatial features optimized for speed"""
    
    print(f"🌍 CREATING FAST GEOSPATIAL FEATURES")
    print("=" * 50)
    
    # Make a copy
    data = df.copy()
    
    # Remove rows with missing coordinates
    valid_coords = (data['latitude'] != 0) & (data['longitude'] != 0) & \
                   data['latitude'].notna() & data['longitude'].notna()
    
    print(f"📍 Valid coordinates: {valid_coords.sum():,} / {len(data):,} ({valid_coords.mean()*100:.1f}%)")
    
    if valid_coords.sum() < n_neighbors:
        print(f"⚠️  Not enough valid coordinates for neighbor analysis")
        return data
    
    # Extract coordinates for valid points
    coords = data.loc[valid_coords, ['latitude', 'longitude']].values
    valid_indices = data.index[valid_coords].tolist()
    
    print(f"🔍 Building k-NN index with {n_neighbors} neighbors...")
    
    # Fit k-nearest neighbors (this is the fast part)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                           metric='haversine', n_jobs=-1)
    nbrs.fit(np.radians(coords))
    
    # Find neighbors for all valid points
    distances, indices = nbrs.kneighbors(np.radians(coords))
    distances = distances * 6371  # Convert to kilometers
    
    print(f"📊 Calculating distance-based features...")
    
    # Initialize feature dictionary
    neighbor_features = {}
    
    # 1. DISTANCE-BASED FEATURES (vectorized)
    neighbor_features['nearest_neighbor_distance'] = distances[:, 1]  # Skip self
    neighbor_features['avg_5_neighbor_distance'] = np.mean(distances[:, 1:6], axis=1)
    neighbor_features['avg_10_neighbor_distance'] = np.mean(distances[:, 1:], axis=1)
    neighbor_features['neighbor_density'] = 1.0 / (neighbor_features['avg_5_neighbor_distance'] + 0.1)
    
    print(f"✅ Distance features: 4")
    
    # 2. STATUS PREDICTION FROM NEIGHBORS (vectorized)
    if 'status_group' in data.columns:
        print(f"🎯 Calculating neighbor status patterns...")
        
        valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
        status_encoder = LabelEncoder()
        status_encoded = status_encoder.fit_transform(valid_status)
        
        # Vectorized neighbor status calculation
        neighbor_statuses = status_encoded[indices[:, 1:]]  # Skip self column
        
        # Find functional class index
        if 'functional' in status_encoder.classes_:
            functional_idx = np.where(status_encoder.classes_ == 'functional')[0][0]
            # Calculate functional ratio for each point's neighbors
            neighbor_features['neighbor_functional_ratio'] = np.mean(
                neighbor_statuses == functional_idx, axis=1
            )
        else:
            neighbor_features['neighbor_functional_ratio'] = np.full(len(indices), 0.5)
        
        print(f"✅ Status prediction: 1")
    
    # 3. INSTALLER/FUNDER CLUSTERING (vectorized)
    print(f"🏗️  Calculating installer/funder spatial patterns...")
    
    for col in ['installer', 'funder']:
        if col in data.columns:
            col_values = data.loc[valid_indices, col].fillna('unknown').values
            
            # Get neighbor values for all points at once
            neighbor_values = col_values[indices[:, 1:]]  # Skip self column
            current_values = col_values[:, np.newaxis]  # Reshape for broadcasting
            
            # Calculate same ratio using broadcasting
            same_ratios = np.mean(neighbor_values == current_values, axis=1)
            neighbor_features[f'{col}_spatial_clustering'] = same_ratios
    
    print(f"✅ Installer/funder clustering: 2")
    
    # 4. FAST GRID-BASED CLUSTERING (instead of K-means)
    print(f"🗺️  Creating fast grid-based clusters...")
    
    # Create grid coordinates
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Calculate grid indices
    lat_grid = ((coords[:, 0] - lat_min) / grid_size).astype(int)
    lon_grid = ((coords[:, 1] - lon_min) / grid_size).astype(int)
    
    # Combine into cluster labels
    cluster_labels = lat_grid * 1000 + lon_grid  # Simple hash
    
    # Reassign to consecutive integers
    unique_clusters = np.unique(cluster_labels)
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    cluster_labels = np.array([cluster_mapping[label] for label in cluster_labels])
    
    neighbor_features['geographic_cluster'] = cluster_labels
    
    # Calculate cluster statistics (vectorized)
    unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    size_mapping = dict(zip(unique_labels, cluster_sizes))
    neighbor_features['cluster_size'] = np.array([size_mapping[label] for label in cluster_labels])
    
    # Simple cluster density based on size and grid area
    neighbor_features['cluster_density'] = neighbor_features['cluster_size'] / (grid_size ** 2 * 111000 ** 2)  # Convert to density per km²
    
    print(f"✅ Grid clustering: 3")
    
    # 5. REGIONAL INFRASTRUCTURE FEATURES (optimized)
    print(f"🏘️  Calculating regional infrastructure patterns...")
    
    if 'ward' in data.columns:
        # Use groupby for efficient aggregation
        ward_stats = data.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean() if len(x) > 0 else 0.5
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        # Merge back efficiently
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(data['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    print(f"✅ Regional features: 2")
    
    # 6. ELEVATION FEATURES (vectorized)
    if 'gps_height' in data.columns:
        print(f"⛰️  Creating elevation-based features...")
        
        gps_heights = data.loc[valid_indices, 'gps_height'].values
        
        # Get neighbor elevations for all points
        neighbor_elevations = gps_heights[indices[:, 1:]]  # Skip self
        current_elevations = gps_heights[:, np.newaxis]  # Reshape for broadcasting
        
        # Calculate elevation differences and standard deviations (vectorized)
        elevation_diffs = np.abs(neighbor_elevations - current_elevations)
        neighbor_features['neighbor_elevation_diff'] = np.mean(elevation_diffs, axis=1)
        neighbor_features['neighbor_elevation_std'] = np.std(neighbor_elevations, axis=1)
        
        print(f"✅ Elevation features: 2")
    
    # Map features back to full dataset (vectorized)
    print(f"🔗 Mapping features back to full dataset...")
    
    for feature_name, feature_values in neighbor_features.items():
        data[feature_name] = np.nan
        data.loc[valid_indices, feature_name] = feature_values
        
        # Fill missing values
        if data[feature_name].dtype in ['float64', 'int64']:
            data[feature_name] = data[feature_name].fillna(data[feature_name].median())
        else:
            data[feature_name] = data[feature_name].fillna(-1)
    
    print(f"\n📊 FAST GEOSPATIAL FEATURES SUMMARY:")
    print("-" * 40)
    total_features = len(neighbor_features) + (2 if 'ward' in df.columns else 0)
    print(f"Total new features created: {total_features}")
    print(f"Distance-based: 4")
    print(f"Neighbor status: {1 if 'status_group' in data.columns else 0}")
    print(f"Spatial clustering: 2")
    print(f"Grid clusters: 3 (fast grid method)")
    print(f"Regional: {2 if 'ward' in df.columns else 0}")
    print(f"Elevation: {2 if 'gps_height' in df.columns else 0}")
    
    print(f"\n⚡ SPEED OPTIMIZATIONS APPLIED:")
    print("-" * 40)
    print("✅ Vectorized neighbor calculations")
    print("✅ Grid-based clustering (vs K-means)")
    print("✅ Broadcasting for same-value comparisons")
    print("✅ Efficient pandas groupby operations")
    print("✅ Batch feature mapping")
    
    return data

# Test the fast implementation
if __name__ == "__main__":
    print("🧪 TESTING FAST GEOSPATIAL FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load data
    print("📊 Loading dataset...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    # Test on larger subset to show speed improvement
    print(f"🎯 Testing on subset of {min(10000, len(train)):,} rows...")
    test_data = train.head(10000).copy()
    
    import time
    start_time = time.time()
    
    # Create fast geospatial features
    enhanced_data = create_fast_geospatial_features(test_data, n_neighbors=15, grid_size=0.01)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ SUCCESS!")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Speed: {len(test_data) / elapsed_time:.0f} rows/second")
    print(f"Original columns: {len(test_data.columns)}")
    print(f"Enhanced columns: {len(enhanced_data.columns)}")
    print(f"New features added: {len(enhanced_data.columns) - len(test_data.columns)}")
    
    # Show feature quality
    new_features = [col for col in enhanced_data.columns if col not in test_data.columns]
    print(f"\n📋 Feature quality check:")
    for feature in new_features[:5]:
        non_null_pct = (enhanced_data[feature].notna()).mean() * 100
        unique_vals = enhanced_data[feature].nunique()
        print(f"  {feature}: {non_null_pct:.1f}% non-null, {unique_vals} unique values")
    
    print(f"\n🚀 Ready for fast integration with models!")