#!/usr/bin/env python3
"""
Fast Ensemble Boost Implementation
==================================
- Optimized for speed while maintaining performance
- Parallel processing and efficient memory usage
- Smart feature selection and model optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class FastFeatureEngineer:
    """Optimized feature engineering with caching"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_cache = {}
    
    def engineer_features(self, df, feature_types=['basic', 'interactions']):
        """Fast feature engineering with selective computation"""
        
        cache_key = f"{len(df)}_{hash(str(sorted(feature_types)))}"
        if cache_key in self.feature_cache:
            print("📋 Using cached features...")
            return self.feature_cache[cache_key]
        
        print(f"🔧 Engineering features: {feature_types}")
        X = df.copy()
        
        # Basic numeric features (always fast)
        if 'basic' in feature_types:
            # Missing value indicators (vectorized)
            zero_cols = ['longitude', 'latitude', 'gps_height', 'population']
            for col in zero_cols:
                if col in X.columns:
                    mask = (X[col] == 0) | X[col].isnull()
                    X[col+'_MISSING'] = mask.astype(np.int8)  # Use int8 for memory
                    X[col] = X[col].where(~mask, np.nan)
            
            # Date features (vectorized)
            if 'date_recorded' in X.columns:
                dates = pd.to_datetime(X['date_recorded'])
                X['year_recorded'] = dates.dt.year.astype(np.int16)
                X['month_recorded'] = dates.dt.month.astype(np.int8)
                X['days_since_recorded'] = (pd.Timestamp('2013-12-03') - dates).dt.days.astype(np.int16)
            
            # Construction year (vectorized)
            if 'construction_year' in X.columns:
                const_year = pd.to_numeric(X['construction_year'], errors='coerce')
                X['pump_age'] = (2013 - const_year).astype(np.int16)
                X['construction_year_missing'] = const_year.isnull().astype(np.int8)
            
            # Text length features (vectorized)
            text_cols = ['scheme_name', 'installer', 'funder']
            for col in text_cols:
                if col in X.columns:
                    lengths = X[col].astype(str).str.len()
                    X[col+'_length'] = lengths.astype(np.int16)
                    X[col+'_is_missing'] = X[col].isnull().astype(np.int8)
        
        # Fast categorical encoding (only for high-cardinality useful features)
        if 'categorical' in feature_types:
            # Select only most important categorical features for speed
            important_cats = ['quantity', 'quantity_group', 'waterpoint_type', 
                             'extraction_type', 'management', 'payment', 'water_quality']
            
            for col in important_cats:
                if col in X.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X[col+'_encoded'] = self.label_encoders[col].fit_transform(X[col].astype(str))
                    else:
                        # Handle unseen categories
                        try:
                            X[col+'_encoded'] = self.label_encoders[col].transform(X[col].astype(str))
                        except ValueError:
                            # Add unseen categories
                            seen_classes = set(self.label_encoders[col].classes_)
                            new_classes = set(X[col].astype(str).unique()) - seen_classes
                            if new_classes:
                                all_classes = list(seen_classes) + list(new_classes)
                                self.label_encoders[col].classes_ = np.array(all_classes)
                            X[col+'_encoded'] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Fast interaction features (only high-impact ones)
        if 'interactions' in feature_types:
            if all(col in X.columns for col in ['pump_age', 'quantity_encoded']):
                X['age_quantity_risk'] = (X['pump_age'] * X['quantity_encoded']).astype(np.float32)
            
            if all(col in X.columns for col in ['population', 'management_encoded']):
                X['pop_mgmt_stress'] = (X['population'] / (X['management_encoded'] + 1)).astype(np.float32)
        
        # Cache the result
        self.feature_cache[cache_key] = X
        return X
    
    def select_features(self, X, y, max_features=50):
        """Fast feature selection using statistical tests"""
        print(f"🎯 Selecting top {max_features} features...")
        
        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(-1)
        
        # Use SelectKBest for speed
        selector = SelectKBest(score_func=f_classif, k=min(max_features, len(numeric_cols)))
        X_selected = selector.fit_transform(X_numeric, y)
        
        selected_features = numeric_cols[selector.get_support()]
        print(f"✅ Selected {len(selected_features)} features")
        
        return X_selected, selected_features

class FastEnsemble:
    """Optimized ensemble with parallel training"""
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.models = {}
        self.scaler = StandardScaler()
        
    def _train_single_model(self, model_config):
        """Train a single model (for parallel execution)"""
        name, model, X_train, y_train = model_config
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        return name, model
    
    def fit(self, X_train, y_train):
        """Train ensemble with parallel model training"""
        print("🎭 Training fast ensemble...")
        
        # Scale features once
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define optimized models (smaller for speed)
        model_configs = [
            ('rf', RandomForestClassifier(
                n_estimators=100,  # Reduced from 300
                max_depth=15,      # Reduced from 20
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=self.n_jobs
            ), X_train, y_train),
            
            ('gb', GradientBoostingClassifier(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.15, # Increased for faster convergence
                max_depth=6,       # Reduced from 8
                min_samples_split=20,
                random_state=42
            ), X_train, y_train),
            
            ('lr', LogisticRegression(
                random_state=42,
                max_iter=500,     # Reduced from 1000
                C=0.1,
                solver='liblinear' # Faster for small datasets
            ), X_train_scaled, y_train)
        ]
        
        # Train models in parallel (use threads for sklearn models)
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self._train_single_model, model_configs))
        
        # Store trained models
        for name, model in results:
            self.models[name] = model
        
        print("✅ Ensemble training complete")
        return self
    
    def predict(self, X_test):
        """Fast ensemble prediction using weighted voting"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions from all models
        predictions = {}
        predictions['rf'] = self.models['rf'].predict_proba(X_test)
        predictions['gb'] = self.models['gb'].predict_proba(X_test)
        predictions['lr'] = self.models['lr'].predict_proba(X_test_scaled)
        
        # Weighted ensemble (tree models get more weight)
        weights = {'rf': 0.4, 'gb': 0.4, 'lr': 0.2}
        
        ensemble_proba = np.zeros_like(predictions['rf'])
        for model_name, proba in predictions.items():
            ensemble_proba += weights[model_name] * proba
        
        return np.argmax(ensemble_proba, axis=1)

def fast_geospatial_features(df, n_neighbors=5):
    """Lightweight geospatial features for speed"""
    print(f"🌍 Creating fast geospatial features...")
    
    X = df.copy()
    
    # Only create the most impactful geospatial features
    valid_coords = (X['latitude'] != 0) & (X['longitude'] != 0) & \
                   X['latitude'].notna() & X['longitude'].notna()
    
    if valid_coords.sum() < n_neighbors:
        return X
    
    from sklearn.neighbors import NearestNeighbors
    
    coords = X.loc[valid_coords, ['latitude', 'longitude']].values
    valid_indices = X.index[valid_coords].tolist()
    
    # Fast k-NN with reduced neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                           metric='haversine', n_jobs=-1)
    nbrs.fit(np.radians(coords))
    
    distances, indices = nbrs.kneighbors(np.radians(coords))
    distances = distances * 6371  # Convert to km
    
    # Only the most important neighbor features
    X['nearest_distance'] = np.nan
    X['neighbor_density'] = np.nan
    X['avg_neighbor_distance'] = np.nan
    
    X.loc[valid_indices, 'nearest_distance'] = distances[:, 1]
    X.loc[valid_indices, 'avg_neighbor_distance'] = np.mean(distances[:, 1:], axis=1)
    X.loc[valid_indices, 'neighbor_density'] = 1.0 / (X.loc[valid_indices, 'avg_neighbor_distance'] + 0.1)
    
    # Fill missing values
    X['nearest_distance'] = X['nearest_distance'].fillna(X['nearest_distance'].median())
    X['avg_neighbor_distance'] = X['avg_neighbor_distance'].fillna(X['avg_neighbor_distance'].median())
    X['neighbor_density'] = X['neighbor_density'].fillna(X['neighbor_density'].median())
    
    print("✅ Fast geospatial features created")
    return X

def run_fast_ensemble_test(df, test_configs=None):
    """Run fast ensemble test with multiple configurations"""
    
    if test_configs is None:
        test_configs = {
            'baseline': {'geo': False, 'interactions': False, 'max_features': 30},
            'with_geo': {'geo': True, 'interactions': False, 'max_features': 40},
            'with_interactions': {'geo': False, 'interactions': True, 'max_features': 35},
            'full_fast': {'geo': True, 'interactions': True, 'max_features': 50}
        }
    
    results = {}
    feature_engineer = FastFeatureEngineer()
    
    print(f"\n🧪 FAST ENSEMBLE TESTING")
    print("=" * 50)
    
    for config_name, config in test_configs.items():
        print(f"\n🔬 Testing {config_name}...")
        
        # Prepare data
        X = df.copy()
        
        # Add geospatial features if requested
        if config['geo']:
            X = fast_geospatial_features(X, n_neighbors=5)
        
        # Engineer features
        feature_types = ['basic', 'categorical']
        if config['interactions']:
            feature_types.append('interactions')
        
        X_engineered = feature_engineer.engineer_features(X, feature_types)
        
        # Feature selection
        y = X_engineered['status_group']
        X_selected, selected_features = feature_engineer.select_features(
            X_engineered, y, max_features=config['max_features']
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train fast ensemble
        ensemble = FastEnsemble(n_jobs=-1)
        ensemble.fit(X_train, y_train)
        
        # Predict
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[config_name] = {
            'accuracy': accuracy,
            'features': len(selected_features),
            'config': config
        }
        
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🎯 Features: {len(selected_features)}")
    
    return results

# Main execution
if __name__ == "__main__":
    print("⚡ FAST ENSEMBLE BOOST IMPLEMENTATION")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    # For speed testing, use a reasonable subset
    print("🎯 Using subset for speed testing...")
    test_size = min(10000, len(train))  # Use up to 10k rows for speed
    train_subset = train.sample(n=test_size, random_state=42)
    print(f"Working with: {len(train_subset):,} rows")
    
    # Test configurations optimized for speed vs performance
    speed_configs = {
        'ultra_fast': {'geo': False, 'interactions': False, 'max_features': 20},
        'balanced': {'geo': True, 'interactions': False, 'max_features': 30},
        'performance': {'geo': True, 'interactions': True, 'max_features': 40}
    }
    
    import time
    start_time = time.time()
    
    # Run tests
    results = run_fast_ensemble_test(train_subset, speed_configs)
    
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n🏆 FAST ENSEMBLE RESULTS")
    print("=" * 50)
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"⚡ Speed: {len(train_subset)/elapsed:.0f} rows/second")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Config':<15} {'Accuracy':<10} {'Features':<10} {'Time/Row':<12}")
    print("-" * 50)
    
    for i, (config_name, result) in enumerate(sorted_results):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        time_per_row = elapsed / len(train_subset) * 1000  # ms per row
        print(f"{emoji} {config_name:<13} {result['accuracy']:.4f}     {result['features']:<10} {time_per_row:.2f}ms")
    
    # Performance analysis
    best_config = sorted_results[0]
    fastest_config = min(results.items(), key=lambda x: x[1]['features'])  # Fewer features = faster
    
    print(f"\n💡 OPTIMIZATION SUMMARY:")
    print("-" * 40)
    print(f"🏆 Best accuracy: {best_config[0]} ({best_config[1]['accuracy']:.4f})")
    print(f"⚡ Fastest config: {fastest_config[0]} ({fastest_config[1]['features']} features)")
    print(f"🔧 Processing speed: {len(train_subset)/elapsed:.0f} rows/second")
    
    if elapsed < 30:
        print("✅ Excellent speed! Ready for larger datasets")
    elif elapsed < 60:
        print("🚀 Good speed! Suitable for iterative development")
    else:
        print("⚠️  Consider further optimization for large datasets")
    
    print(f"\n🎯 SPEED OPTIMIZATIONS APPLIED:")
    print("-" * 40)
    print("✅ Vectorized feature engineering")
    print("✅ Reduced model complexity")
    print("✅ Fast feature selection")
    print("✅ Parallel model training")
    print("✅ Lightweight geospatial features")
    print("✅ Memory-efficient data types")
    print("✅ Cached computations")
    
    print(f"\n📈 NEXT STEPS:")
    print("-" * 40)
    print("1. Run on full dataset with best config")
    print("2. Fine-tune hyperparameters if needed")
    print("3. Consider early stopping for gradient boosting")
    print("4. Profile memory usage for very large datasets")