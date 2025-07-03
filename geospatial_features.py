#!/usr/bin/env python3
"""
Advanced Geospatial Feature Engineering
======================================
- K-nearest neighbors analysis
- Installation pattern clustering
- Geographic density features
- Installer/funder spatial patterns
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def create_geospatial_features(df, n_neighbors=10, n_clusters=50):
    """Create comprehensive geospatial features"""
    
    print(f"🌍 CREATING GEOSPATIAL FEATURES")
    print("=" * 50)
    
    # Make a copy to avoid modifying original
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
    
    # Fit k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                           metric='haversine', n_jobs=-1)
    nbrs.fit(np.radians(coords))  # Convert to radians for haversine
    
    # Find neighbors for all valid points
    distances, indices = nbrs.kneighbors(np.radians(coords))
    
    # Convert distances back to kilometers
    distances = distances * 6371  # Earth radius in km
    
    print(f"📊 Calculating neighbor-based features...")
    
    # Initialize feature arrays
    neighbor_features = {}
    
    # 1. DISTANCE-BASED FEATURES
    neighbor_features['nearest_neighbor_distance'] = distances[:, 1]  # Skip self (index 0)
    neighbor_features['avg_5_neighbor_distance'] = np.mean(distances[:, 1:6], axis=1)
    neighbor_features['avg_10_neighbor_distance'] = np.mean(distances[:, 1:], axis=1)
    neighbor_features['neighbor_density'] = 1.0 / (neighbor_features['avg_5_neighbor_distance'] + 0.1)
    
    print(f"✅ Distance features: {len([k for k in neighbor_features.keys() if 'distance' in k or 'density' in k])}")
    
    # 2. STATUS PREDICTION FROM NEIGHBORS
    if 'status_group' in data.columns:
        print(f"🎯 Calculating neighbor status patterns...")
        
        # Create status encoding for valid coordinates only
        valid_status = data.loc[valid_indices, 'status_group'].fillna('unknown')
        status_encoder = LabelEncoder()
        status_encoded = status_encoder.fit_transform(valid_status)
        
        neighbor_functional_ratios = []
        
        for i, neighbor_idx in enumerate(indices):
            if i < len(valid_indices):
                # Get neighbor statuses (skip self at index 0)
                # neighbor_idx contains indices into the valid_indices array
                valid_neighbor_positions = [idx for idx in neighbor_idx[1:] if idx < len(status_encoded)]
                if valid_neighbor_positions:
                    neighbor_statuses = status_encoded[valid_neighbor_positions]
                    
                    # Calculate functional ratio among neighbors
                    status_classes = status_encoder.classes_
                    if 'functional' in status_classes:
                        functional_idx = np.where(status_classes == 'functional')[0][0]
                        functional_ratio = (neighbor_statuses == functional_idx).mean()
                    else:
                        functional_ratio = 0.5  # Default if no functional class
                else:
                    functional_ratio = 0.5
                
                neighbor_functional_ratios.append(functional_ratio)
        
        neighbor_features['neighbor_functional_ratio'] = np.array(neighbor_functional_ratios)
        print(f"✅ Status prediction features: 1")
    
    # 3. INSTALLER/FUNDER CLUSTERING
    print(f"🏗️  Calculating installer/funder spatial patterns...")
    
    for col in ['installer', 'funder']:
        if col in data.columns:
            # Get installer/funder for each valid point
            col_values = data.loc[valid_indices, col].fillna('unknown').values
            
            # Calculate same installer/funder ratios among neighbors
            same_ratios = []
            
            for i, neighbor_idx in enumerate(indices):
                if i < len(valid_indices):
                    current_value = col_values[i]
                    # Map neighbor indices to valid array positions
                    valid_neighbor_positions = [idx for idx in neighbor_idx[1:] if idx < len(col_values)]
                    if valid_neighbor_positions:
                        neighbor_values = col_values[valid_neighbor_positions]
                        same_ratio = (neighbor_values == current_value).mean()
                    else:
                        same_ratio = 0.0
                    same_ratios.append(same_ratio)
            
            neighbor_features[f'{col}_spatial_clustering'] = np.array(same_ratios)
    
    print(f"✅ Installer/funder clustering: {len([k for k in neighbor_features.keys() if 'clustering' in k])}")
    
    # 4. GEOGRAPHIC CLUSTERING
    print(f"🗺️  Creating geographic clusters...")
    
    # K-means clustering on coordinates
    kmeans = KMeans(n_clusters=min(n_clusters, len(coords)//10), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    neighbor_features['geographic_cluster'] = cluster_labels
    
    # Calculate cluster-level statistics
    cluster_sizes = []
    cluster_densities = []
    
    for i, cluster_id in enumerate(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = cluster_mask.sum()
        cluster_sizes.append(cluster_size)
        
        # Calculate average distance within cluster
        if cluster_size > 1:
            cluster_coords = coords[cluster_mask]
            cluster_distances = cdist(cluster_coords, cluster_coords, metric='euclidean')
            # Get upper triangle (exclude diagonal and duplicates)
            upper_tri = np.triu_indices_from(cluster_distances, k=1)
            avg_cluster_distance = cluster_distances[upper_tri].mean()
            cluster_densities.append(1.0 / (avg_cluster_distance + 0.1))
        else:
            cluster_densities.append(0.0)
    
    neighbor_features['cluster_size'] = np.array(cluster_sizes)
    neighbor_features['cluster_density'] = np.array(cluster_densities)
    
    print(f"✅ Geographic clustering: 3")
    
    # 5. REGIONAL INFRASTRUCTURE FEATURES
    print(f"🏘️  Calculating regional infrastructure patterns...")
    
    # Calculate features based on administrative boundaries
    if 'ward' in data.columns:
        ward_stats = data.groupby('ward').agg({
            'latitude': 'count',  # Number of pumps in ward
            'status_group': lambda x: (x == 'functional').mean() if 'functional' in x.values else 0.5
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        # Map back to original data
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        
        # Fill missing values
        data['ward_pump_count'] = data['ward_pump_count'].fillna(data['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    print(f"✅ Regional features: 2")
    
    # 6. ELEVATION AND TERRAIN FEATURES
    if 'gps_height' in data.columns:
        print(f"⛰️  Creating elevation-based features...")
        
        gps_heights = data.loc[valid_indices, 'gps_height'].values
        neighbor_elevation_diffs = []
        neighbor_elevation_stds = []
        
        for i, neighbor_idx in enumerate(indices):
            if i < len(valid_indices):
                current_elevation = gps_heights[i]
                # Map neighbor indices to valid array positions
                valid_neighbor_positions = [idx for idx in neighbor_idx[1:] if idx < len(gps_heights)]
                if valid_neighbor_positions:
                    neighbor_elevations = gps_heights[valid_neighbor_positions]
                    
                    # Calculate elevation differences
                    elevation_diffs = np.abs(neighbor_elevations - current_elevation)
                    neighbor_elevation_diffs.append(elevation_diffs.mean())
                    neighbor_elevation_stds.append(neighbor_elevations.std())
                else:
                    neighbor_elevation_diffs.append(0.0)
                    neighbor_elevation_stds.append(0.0)
        
        neighbor_features['neighbor_elevation_diff'] = np.array(neighbor_elevation_diffs)
        neighbor_features['neighbor_elevation_std'] = np.array(neighbor_elevation_stds)
        
        print(f"✅ Elevation features: 2")
    
    # Map features back to full dataset
    print(f"🔗 Mapping features back to full dataset...")
    
    for feature_name, feature_values in neighbor_features.items():
        data[feature_name] = np.nan
        data.loc[valid_indices, feature_name] = feature_values
    
    # Fill missing values for invalid coordinates
    for feature_name in neighbor_features.keys():
        if data[feature_name].dtype in ['float64', 'int64']:
            data[feature_name] = data[feature_name].fillna(data[feature_name].median())
        else:
            data[feature_name] = data[feature_name].fillna(-1)
    
    print(f"\n📊 GEOSPATIAL FEATURES SUMMARY:")
    print("-" * 40)
    total_features = len(neighbor_features) + (2 if 'ward' in df.columns else 0)
    print(f"Total new features created: {total_features}")
    print(f"Distance-based: 4")
    print(f"Neighbor status: {1 if 'status_group' in data.columns else 0}")
    print(f"Spatial clustering: {len([k for k in neighbor_features.keys() if 'clustering' in k])}")
    print(f"Geographic clusters: 3")
    print(f"Regional: {2 if 'ward' in df.columns else 0}")
    print(f"Elevation: {2 if 'gps_height' in df.columns else 0}")
    
    # Feature quality assessment
    print(f"\n🔍 FEATURE QUALITY ASSESSMENT:")
    print("-" * 40)
    
    for feature_name, feature_values in neighbor_features.items():
        non_null_pct = (data[feature_name].notna()).mean() * 100
        unique_values = data[feature_name].nunique()
        print(f"{feature_name:<30} {non_null_pct:6.1f}% non-null, {unique_values:6d} unique values")
    
    return data

# Test the geospatial feature engineering
if __name__ == "__main__":
    print("🧪 TESTING GEOSPATIAL FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load data
    print("📊 Loading dataset...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    # Test on smaller subset first
    print(f"🎯 Testing on subset of {min(5000, len(train)):,} rows...")
    test_data = train.head(5000).copy()
    
    # Create geospatial features
    enhanced_data = create_geospatial_features(test_data, n_neighbors=10, n_clusters=20)
    
    print(f"\n✅ SUCCESS!")
    print(f"Original columns: {len(test_data.columns)}")
    print(f"Enhanced columns: {len(enhanced_data.columns)}")
    print(f"New features added: {len(enhanced_data.columns) - len(test_data.columns)}")
    
    # Show sample of new features
    new_features = [col for col in enhanced_data.columns if col not in test_data.columns]
    print(f"\n📋 New feature sample:")
    for feature in new_features[:5]:
        sample_values = enhanced_data[feature].dropna().head(3).values
        print(f"  {feature}: {sample_values}")
    
    print(f"\n🚀 Ready to integrate with full model!")