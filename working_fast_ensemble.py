#!/usr/bin/env python3
"""
Working Fast Ensemble Implementation
===================================
- Verified working approach based on debug results
- Optimized for speed while maintaining accuracy
- Simple, reliable feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

def fast_feature_prep(df):
    """Fast, reliable feature preparation based on debug results"""
    print("🔧 Fast feature preparation...")
    
    X = df.copy()
    feature_cols = []
    
    # 1. Basic numeric features
    numeric_base = ['amount_tsh', 'longitude', 'latitude', 'gps_height', 'population']
    for col in numeric_base:
        if col in X.columns:
            feature_cols.append(col)
    
    # 2. Missing indicators for coordinates (important pattern)
    coord_cols = ['longitude', 'latitude', 'gps_height']
    for col in coord_cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
            missing_col = col + '_MISSING'
            X[missing_col] = X[col].isnull().astype(int)
            feature_cols.append(missing_col)
    
    # 3. Key categorical features (most predictive ones)
    categorical_features = {
        'quantity': 'quantity_encoded',
        'waterpoint_type': 'waterpoint_type_encoded', 
        'payment': 'payment_encoded',
        'water_quality': 'water_quality_encoded',
        'extraction_type': 'extraction_type_encoded'
    }
    
    for orig_col, encoded_col in categorical_features.items():
        if orig_col in X.columns:
            le = LabelEncoder()
            X[encoded_col] = le.fit_transform(X[orig_col].astype(str))
            feature_cols.append(encoded_col)
    
    # 4. Date features (if available)
    if 'date_recorded' in X.columns:
        dates = pd.to_datetime(X['date_recorded'])
        X['year_recorded'] = dates.dt.year
        X['days_since_recorded'] = (pd.Timestamp('2013-12-03') - dates).dt.days
        feature_cols.extend(['year_recorded', 'days_since_recorded'])
    
    # 5. Pump age (if available)
    if 'construction_year' in X.columns:
        const_year = pd.to_numeric(X['construction_year'], errors='coerce')
        X['pump_age'] = 2013 - const_year
        X['pump_age_missing'] = const_year.isnull().astype(int)
        feature_cols.extend(['pump_age', 'pump_age_missing'])
    
    print(f"✅ Prepared {len(feature_cols)} features")
    
    # Return feature matrix with missing values filled
    X_features = X[feature_cols].fillna(-1)  # Use -1 for missing values
    return X_features, feature_cols

class FastEnsemble:
    """Simple, fast ensemble classifier"""
    
    def __init__(self, use_scaling=True):
        self.models = {}
        self.scaler = StandardScaler() if use_scaling else None
        self.use_scaling = use_scaling
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """Train the ensemble quickly"""
        print("🎭 Training fast ensemble...")
        
        # Scale features if needed
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        # 1. Random Forest (tree-based, doesn't need scaling)
        print("  📊 Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        
        # 2. Gradient Boosting (tree-based, doesn't need scaling)
        print("  🚀 Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # 3. Logistic Regression (needs scaling)
        if self.use_scaling:
            print("  📈 Training Logistic Regression...")
            self.models['lr'] = LogisticRegression(
                random_state=42,
                max_iter=500,
                C=1.0
            )
            self.models['lr'].fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        print("✅ Ensemble ready!")
        return self
    
    def predict(self, X_test):
        """Fast ensemble prediction using simple voting"""
        if not self.is_fitted:
            raise ValueError("Must fit ensemble before predicting")
        
        # Get predictions from tree models
        rf_pred = self.models['rf'].predict(X_test)
        gb_pred = self.models['gb'].predict(X_test)
        
        predictions = [rf_pred, gb_pred]
        weights = [2, 2]  # Equal weight for tree models
        
        # Add logistic regression if available
        if 'lr' in self.models:
            if self.use_scaling:
                X_test_scaled = self.scaler.transform(X_test)
                lr_pred = self.models['lr'].predict(X_test_scaled)
                predictions.append(lr_pred)
                weights.append(1)  # Lower weight for LR
        
        # Simple weighted majority voting
        ensemble_pred = []
        for i in range(len(X_test)):
            # Get votes for this sample
            votes = [pred[i] for pred in predictions]
            
            # Count weighted votes
            vote_counts = {}
            for j, vote in enumerate(votes):
                if vote not in vote_counts:
                    vote_counts[vote] = 0
                vote_counts[vote] += weights[j]
            
            # Get majority vote
            best_vote = max(vote_counts.items(), key=lambda x: x[1])[0]
            ensemble_pred.append(best_vote)
        
        return np.array(ensemble_pred)
    
    def get_individual_accuracies(self, X_test, y_test):
        """Get individual model accuracies for analysis"""
        accuracies = {}
        
        # Tree models
        rf_pred = self.models['rf'].predict(X_test)
        gb_pred = self.models['gb'].predict(X_test)
        
        accuracies['rf'] = accuracy_score(y_test, rf_pred)
        accuracies['gb'] = accuracy_score(y_test, gb_pred)
        
        # Logistic regression
        if 'lr' in self.models:
            if self.use_scaling:
                X_test_scaled = self.scaler.transform(X_test)
                lr_pred = self.models['lr'].predict(X_test_scaled)
                accuracies['lr'] = accuracy_score(y_test, lr_pred)
        
        return accuracies

def benchmark_ensemble(df, test_sizes=[2000, 5000], include_gpt=False):
    """Benchmark ensemble performance at different dataset sizes"""
    
    results = {}
    
    for test_size in test_sizes:
        print(f"\n🔬 TESTING WITH {test_size:,} ROWS")
        print("=" * 50)
        
        # Sample data
        if len(df) > test_size:
            df_test = df.sample(n=test_size, random_state=42)
        else:
            df_test = df.copy()
        
        start_time = time.time()
        
        # 1. Feature preparation
        prep_start = time.time()
        X_features, feature_names = fast_feature_prep(df_test)
        y = df_test['status_group']
        prep_time = time.time() - prep_start
        
        print(f"📊 Features: {len(feature_names)} in {prep_time:.2f}s")
        
        # 2. Train-test split
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.25, random_state=42, stratify=y
        )
        split_time = time.time() - split_start
        
        print(f"📊 Split: {len(X_train)} train, {len(X_test)} test in {split_time:.2f}s")
        
        # 3. Train ensemble
        train_start = time.time()
        ensemble = FastEnsemble(use_scaling=True)
        ensemble.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # 4. Predict
        pred_start = time.time()
        ensemble_pred = ensemble.predict(X_test)
        pred_time = time.time() - pred_start
        
        # 5. Evaluate
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        individual_accs = ensemble.get_individual_accuracies(X_test, y_test)
        
        total_time = time.time() - start_time
        
        # Results
        results[test_size] = {
            'ensemble_accuracy': ensemble_acc,
            'individual_accuracies': individual_accs,
            'total_time': total_time,
            'prep_time': prep_time,
            'train_time': train_time,
            'pred_time': pred_time,
            'rows_per_second': test_size / total_time,
            'features': len(feature_names)
        }
        
        print(f"\n📈 RESULTS:")
        print(f"   🎭 Ensemble:     {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
        for model, acc in individual_accs.items():
            print(f"   📊 {model.upper():<12} {acc:.4f} ({acc*100:.2f}%)")
        
        best_individual = max(individual_accs.values())
        boost = ensemble_acc - best_individual
        print(f"   📈 Boost:        +{boost:.4f} ({boost*100:.2f}pp)")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Preparation:  {prep_time:.2f}s")
        print(f"   Training:     {train_time:.2f}s")
        print(f"   Prediction:   {pred_time:.2f}s")
        print(f"   Total:        {total_time:.2f}s")
        print(f"   Speed:        {test_size/total_time:.0f} rows/second")
    
    return results

if __name__ == "__main__":
    print("⚡ WORKING FAST ENSEMBLE IMPLEMENTATION")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    print(f"✅ Loaded {len(train):,} rows")
    
    # Benchmark at different sizes
    test_sizes = [2000, 5000, 10000]
    results = benchmark_ensemble(train, test_sizes)
    
    # Summary
    print(f"\n🏆 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Size':<8} {'Accuracy':<10} {'Time':<8} {'Speed':<12} {'Boost':<10}")
    print("-" * 55)
    
    for size, result in results.items():
        best_individual = max(result['individual_accuracies'].values())
        boost = result['ensemble_accuracy'] - best_individual
        
        print(f"{size:<8} {result['ensemble_accuracy']:<10.4f} {result['total_time']:<8.1f}s "
              f"{result['rows_per_second']:<12.0f} +{boost*100:<9.2f}pp")
    
    # Efficiency analysis
    avg_accuracy = np.mean([r['ensemble_accuracy'] for r in results.values()])
    avg_speed = np.mean([r['rows_per_second'] for r in results.values()])
    avg_boost = np.mean([r['ensemble_accuracy'] - max(r['individual_accuracies'].values()) 
                        for r in results.values()])
    
    print(f"\n💡 PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"🎯 Average accuracy:     {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"⚡ Average speed:        {avg_speed:.0f} rows/second")  
    print(f"🚀 Average ensemble boost: +{avg_boost*100:.2f} percentage points")
    
    # Recommendations
    print(f"\n🔧 OPTIMIZATION STATUS:")
    print("-" * 40)
    if avg_speed > 1000:
        print("🚀 Excellent speed! Ready for production use")
    elif avg_speed > 500:
        print("✅ Good speed! Suitable for large datasets")
    else:
        print("⚠️  Consider further optimization")
    
    if avg_boost > 0.01:
        print("🎭 Ensemble provides meaningful improvement")
    else:
        print("📊 Ensemble provides modest improvement")
    
    print(f"\n🎯 SPEED OPTIMIZATIONS APPLIED:")
    print("-" * 40)
    print("✅ Simplified feature engineering")
    print("✅ Reduced model complexity")
    print("✅ Fast majority voting")
    print("✅ Parallel training where possible")
    print("✅ Efficient data preprocessing")
    
    # Extrapolate to full dataset
    largest_result = results[max(results.keys())]
    full_dataset_time = len(train) / largest_result['rows_per_second']
    
    print(f"\n📊 FULL DATASET PROJECTION:")
    print("-" * 40)
    print(f"Full dataset size: {len(train):,} rows")
    print(f"Estimated time: {full_dataset_time/60:.1f} minutes")
    print(f"Expected accuracy: ~{avg_accuracy:.3f}")
    
    if full_dataset_time < 300:  # 5 minutes
        print("🚀 Ready for full dataset processing!")
    else:
        print("⚠️  Consider using subset for development")