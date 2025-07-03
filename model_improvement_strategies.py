#!/usr/bin/env python3
"""
Advanced Model Improvement Strategies
====================================
- Ensemble methods and model stacking
- Creative feature engineering
- Graph-based approaches
- Advanced preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("🚀 ADVANCED MODEL IMPROVEMENT STRATEGIES")
print("=" * 60)

class WaterPumpModelEnhancer:
    """Advanced model enhancement strategies"""
    
    def __init__(self):
        self.models = {}
        self.features = {}
        self.scalers = {}
        
    def create_interaction_features(self, df):
        """Create sophisticated interaction features"""
        print("🔗 Creating interaction features...")
        
        X = df.copy()
        
        # 1. Geographic-Administrative Interactions
        if all(col in X.columns for col in ['latitude', 'longitude', 'ward_encoded']):
            # Ward centroid distances
            ward_centroids = X.groupby('ward_encoded')[['latitude', 'longitude']].mean()
            X['ward_centroid_distance'] = 0.0
            for ward_id in ward_centroids.index:
                mask = X['ward_encoded'] == ward_id
                if mask.any():
                    centroid_lat = ward_centroids.loc[ward_id, 'latitude']
                    centroid_lon = ward_centroids.loc[ward_id, 'longitude']
                    distances = np.sqrt((X.loc[mask, 'latitude'] - centroid_lat)**2 + 
                                      (X.loc[mask, 'longitude'] - centroid_lon)**2)
                    X.loc[mask, 'ward_centroid_distance'] = distances
        
        # 2. Age-Technology Interactions
        if all(col in X.columns for col in ['pump_age', 'extraction_type_encoded']):
            X['age_extraction_interaction'] = X['pump_age'] * X['extraction_type_encoded']
        
        # 3. Installer-Funder Coordination Score
        if all(col in X.columns for col in ['installer_encoded', 'funder_encoded']):
            # Create coordination score based on how often installer/funder pairs work together
            coordination_counts = X.groupby(['installer_encoded', 'funder_encoded']).size()
            coordination_mapping = coordination_counts / coordination_counts.max()
            X['installer_funder_coordination'] = X.apply(
                lambda row: coordination_mapping.get((row['installer_encoded'], row['funder_encoded']), 0), 
                axis=1
            )
        
        # 4. Population-Quality Interactions
        if all(col in X.columns for col in ['population', 'water_quality_encoded']):
            X['population_quality_ratio'] = X['population'] / (X['water_quality_encoded'] + 1)
        
        # 5. Elevation-Source Interactions
        if all(col in X.columns for col in ['gps_height', 'source_encoded']):
            X['elevation_source_interaction'] = X['gps_height'] * X['source_encoded']
        
        print(f"✅ Created {len([c for c in X.columns if c not in df.columns])} interaction features")
        return X
    
    def create_temporal_features(self, df):
        """Advanced temporal feature engineering"""
        print("📅 Creating temporal features...")
        
        X = df.copy()
        
        if 'date_recorded' in X.columns:
            X['date_recorded'] = pd.to_datetime(X['date_recorded'])
            
            # Seasonal patterns
            X['recording_season'] = ((X['date_recorded'].dt.month - 1) // 3) + 1
            X['is_dry_season'] = ((X['date_recorded'].dt.month >= 6) & 
                                 (X['date_recorded'].dt.month <= 9)).astype(int)
            
            # Time since construction vs recording patterns
            if all(col in X.columns for col in ['construction_year', 'year_recorded']):
                X['years_in_service'] = X['year_recorded'] - X['construction_year']
                X['years_in_service'] = X['years_in_service'].clip(0, 50)  # Cap at reasonable max
                
                # Age categories
                X['pump_age_category'] = pd.cut(X['years_in_service'], 
                                              bins=[-1, 5, 15, 30, 100], 
                                              labels=['new', 'medium', 'old', 'very_old'])
                X['pump_age_category'] = LabelEncoder().fit_transform(X['pump_age_category'].astype(str))
        
        print(f"✅ Created temporal features")
        return X
    
    def create_text_features(self, df):
        """Advanced text feature extraction"""
        print("📝 Creating text features...")
        
        X = df.copy()
        
        text_columns = ['scheme_name', 'installer', 'funder']
        
        for col in text_columns:
            if col in X.columns:
                # Text complexity measures
                X[f'{col}_word_count'] = X[col].astype(str).str.split().str.len()
                X[f'{col}_char_count'] = X[col].astype(str).str.len()
                X[f'{col}_has_numbers'] = X[col].astype(str).str.contains(r'\d').astype(int)
                X[f'{col}_has_special_chars'] = X[col].astype(str).str.contains(r'[^a-zA-Z0-9\s]').astype(int)
                
                # Common keywords (domain-specific)
                if col == 'scheme_name':
                    keywords = ['water', 'pump', 'bore', 'well', 'spring', 'gravity', 'pipe']
                    for keyword in keywords:
                        X[f'{col}_contains_{keyword}'] = X[col].astype(str).str.lower().str.contains(keyword).astype(int)
                
                # Organization type patterns
                if col in ['installer', 'funder']:
                    org_patterns = {
                        'government': ['government', 'ministry', 'district', 'council', 'dwe'],
                        'ngo': ['world', 'vision', 'oxfam', 'foundation', 'trust', 'charity'],
                        'religious': ['church', 'mission', 'christian', 'islamic', 'religious'],
                        'private': ['company', 'limited', 'ltd', 'private', 'individual']
                    }
                    
                    for org_type, patterns in org_patterns.items():
                        pattern = '|'.join(patterns)
                        X[f'{col}_is_{org_type}'] = X[col].astype(str).str.lower().str.contains(pattern).astype(int)
        
        print(f"✅ Created text features")
        return X
    
    def create_anomaly_features(self, df):
        """Identify anomalous pumps that might need special attention"""
        print("🔍 Creating anomaly detection features...")
        
        X = df.copy()
        
        # Geographic anomalies
        if all(col in X.columns for col in ['latitude', 'longitude', 'ward_encoded']):
            # Pumps far from their ward centroid
            ward_centroids = X.groupby('ward_encoded')[['latitude', 'longitude']].mean()
            X['is_geographic_outlier'] = 0
            
            for ward_id in ward_centroids.index:
                mask = X['ward_encoded'] == ward_id
                if mask.sum() > 1:  # Need multiple points to calculate outliers
                    ward_lats = X.loc[mask, 'latitude']
                    ward_lons = X.loc[mask, 'longitude']
                    
                    # Calculate distances from ward median
                    median_lat = ward_lats.median()
                    median_lon = ward_lons.median()
                    distances = np.sqrt((ward_lats - median_lat)**2 + (ward_lons - median_lon)**2)
                    
                    # Mark as outlier if > 2 standard deviations from median
                    threshold = distances.mean() + 2 * distances.std()
                    outliers = distances > threshold
                    X.loc[mask, 'is_geographic_outlier'] = outliers.astype(int)
        
        # Population anomalies
        if 'population' in X.columns:
            pop_q99 = X['population'].quantile(0.99)
            X['is_high_population_outlier'] = (X['population'] > pop_q99).astype(int)
        
        # Age anomalies
        if 'pump_age' in X.columns:
            X['is_very_old_pump'] = (X['pump_age'] > 40).astype(int)
            X['is_future_construction'] = (X['pump_age'] < 0).astype(int)
        
        print(f"✅ Created anomaly features")
        return X
    
    def create_ensemble_models(self, X_train, y_train, X_test):
        """Create sophisticated ensemble of diverse models"""
        print("🎭 Creating ensemble models...")
        
        # Scale features for neural network and logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Individual models with different strengths
        models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        # Train individual models
        predictions = {}
        for name, model in models.items():
            if name in ['lr', 'mlp']:
                model.fit(X_train_scaled, y_train)
                predictions[name] = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions[name] = model.predict_proba(X_test)
        
        # Weighted ensemble (weights based on validation performance)
        weights = {'rf': 0.35, 'gb': 0.35, 'lr': 0.15, 'mlp': 0.15}
        
        ensemble_proba = np.zeros_like(predictions['rf'])
        for name, weight in weights.items():
            ensemble_proba += weight * predictions[name]
        
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Convert back to original labels
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        ensemble_pred_labels = label_encoder.inverse_transform(ensemble_pred)
        
        return ensemble_pred_labels, models, predictions
    
    def analyze_model_improvements(self, df_enhanced, target_col='status_group'):
        """Comprehensive analysis of all improvement strategies"""
        print("\n📊 COMPREHENSIVE MODEL IMPROVEMENT ANALYSIS")
        print("=" * 60)
        
        # Prepare features
        feature_sets = {}
        
        # 1. Baseline + interactions
        df_interactions = self.create_interaction_features(df_enhanced)
        feature_sets['interactions'] = df_interactions
        
        # 2. + Temporal features
        df_temporal = self.create_temporal_features(df_interactions)
        feature_sets['temporal'] = df_temporal
        
        # 3. + Text features
        df_text = self.create_text_features(df_temporal)
        feature_sets['text'] = df_text
        
        # 4. + Anomaly features
        df_anomaly = self.create_anomaly_features(df_text)
        feature_sets['full_enhanced'] = df_anomaly
        
        print(f"\nFeature evolution:")
        print(f"Original: {len(df_enhanced.columns)} features")
        for name, df_set in feature_sets.items():
            new_features = len(df_set.columns) - len(df_enhanced.columns)
            print(f"{name}: +{new_features} features ({len(df_set.columns)} total)")
        
        return feature_sets

def suggest_creative_improvements():
    """Suggest creative and advanced improvement strategies"""
    
    strategies = {
        "🎯 **Ensemble Strategies**": [
            "Multi-level stacking (meta-learners on top of base models)",
            "Dynamic ensemble weights based on geographic region",
            "Bayesian model averaging for uncertainty quantification",
            "Online learning ensemble that adapts to new data patterns"
        ],
        
        "🌐 **Graph-Based Approaches**": [
            "Graph Neural Networks treating pumps as nodes connected by distance",
            "PageRank-style algorithms for pump importance based on network effects",
            "Community detection to find infrastructure clusters",
            "Graph embedding features (node2vec for pump relationships)"
        ],
        
        "🧠 **Deep Learning Enhancements**": [
            "TabNet for end-to-end tabular learning with attention",
            "Transformer models with positional encoding for geographic data",
            "Multi-task learning (predict functionality + maintenance needs)",
            "Variational autoencoders for anomaly detection"
        ],
        
        "🛰️ **External Data Integration**": [
            "Satellite imagery analysis for infrastructure development",
            "Weather/rainfall data for seasonal patterns",
            "Economic indicators (GDP, poverty rates) at district level",
            "Road network accessibility for maintenance logistics"
        ],
        
        "⚡ **Advanced Feature Engineering**": [
            "Fourier transform features for cyclical patterns",
            "Wavelet features for multi-scale geographic analysis",
            "Graph Laplacian features for spatial smoothness",
            "Persistence diagrams from topological data analysis"
        ],
        
        "🎲 **Probabilistic Models**": [
            "Gaussian Process regression for spatial interpolation",
            "Bayesian networks modeling causal relationships",
            "Hidden Markov Models for pump lifecycle states",
            "Dirichlet processes for infinite mixture models"
        ],
        
        "🔄 **Meta-Learning Approaches**": [
            "Learn how to quickly adapt to new regions/installers",
            "Few-shot learning for rare pump types",
            "Transfer learning from other infrastructure datasets",
            "Domain adaptation between different countries/regions"
        ],
        
        "📈 **Optimization Strategies**": [
            "Automated feature selection using genetic algorithms",
            "Neural architecture search for optimal model design",
            "Multi-objective optimization (accuracy vs interpretability)",
            "Evolutionary strategies for hyperparameter tuning"
        ]
    }
    
    print("\n🚀 CREATIVE IMPROVEMENT STRATEGIES")
    print("=" * 60)
    
    for category, items in strategies.items():
        print(f"\n{category}")
        print("-" * 50)
        for i, item in enumerate(items, 1):
            print(f"{i}. {item}")
    
    print(f"\n💡 **IMMEDIATE HIGH-IMPACT RECOMMENDATIONS:**")
    print("-" * 50)
    print("1. 🥇 **Ensemble of diverse models** - Easy 1-2% gain")
    print("2. 🥈 **Graph-based neighbor features** - Leverage spatial relationships") 
    print("3. 🥉 **Advanced interaction features** - Age×technology, installer×funder")
    print("4. 🏅 **External weather/economic data** - Seasonal and economic patterns")
    print("5. 🏆 **Multi-task learning** - Predict functionality + maintenance needs")
    
    print(f"\n⚖️ **COMPLEXITY vs IMPACT ASSESSMENT:**")
    print("-" * 50)
    print("🟢 **Low complexity, high impact**: Ensemble methods, interaction features")
    print("🟡 **Medium complexity, high impact**: Graph features, external data")
    print("🔴 **High complexity, uncertain impact**: Deep learning, meta-learning")

if __name__ == "__main__":
    # Initialize enhancer
    enhancer = WaterPumpModelEnhancer()
    
    # Show creative strategies
    suggest_creative_improvements()
    
    print(f"\n🎯 **RECOMMENDED IMPLEMENTATION ORDER:**")
    print("=" * 60)
    print("1. **Ensemble methods** (Quick wins)")
    print("2. **Interaction features** (Domain knowledge)")
    print("3. **Graph-based features** (Spatial relationships)")
    print("4. **External data integration** (Weather, economics)")
    print("5. **Advanced deep learning** (Long-term research)")
    
    print(f"\n📊 Ready to implement improvement strategies!")