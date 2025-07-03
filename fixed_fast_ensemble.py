#!/usr/bin/env python3
"""
Fixed Fast Ensemble Implementation
==================================
- Fixes label encoding and feature selection issues
- Optimized for speed while maintaining accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def fast_feature_engineering(df):
    """Streamlined feature engineering focused on high-impact features"""
    print("🔧 Fast feature engineering...")
    
    X = df.copy()
    
    # 1. Handle missing coordinates
    coord_cols = ['longitude', 'latitude', 'gps_height']
    for col in coord_cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
            X[col+'_missing'] = X[col].isnull().astype(int)
    
    # 2. Date features (keep simple)
    if 'date_recorded' in X.columns:
        dates = pd.to_datetime(X['date_recorded'])
        X['year_recorded'] = dates.dt.year
        X['days_since_recorded'] = (pd.Timestamp('2013-12-03') - dates).dt.days
    
    # 3. Pump age
    if 'construction_year' in X.columns:
        const_year = pd.to_numeric(X['construction_year'], errors='coerce')
        X['pump_age'] = 2013 - const_year
        X['pump_age_missing'] = const_year.isnull().astype(int)
    
    # 4. Population density
    if 'population' in X.columns:
        X['population'] = X['population'].replace(0, np.nan)
        X['population_missing'] = X['population'].isnull().astype(int)
        X['log_population'] = np.log1p(X['population'].fillna(1))
    
    # 5. Key categorical features only (most important ones)
    key_categoricals = {
        'quantity': 'quantity_cat',
        'quantity_group': 'quantity_group_cat', 
        'waterpoint_type': 'waterpoint_type_cat',
        'extraction_type': 'extraction_type_cat',
        'management': 'management_cat',
        'payment': 'payment_cat',
        'water_quality': 'water_quality_cat',
        'source_type': 'source_type_cat'
    }
    
    # Label encode key categoricals
    for orig_col, new_col in key_categoricals.items():
        if orig_col in X.columns:
            le = LabelEncoder()
            X[new_col] = le.fit_transform(X[orig_col].astype(str))
    
    # 6. Simple interaction features
    if all(col in X.columns for col in ['pump_age', 'quantity_cat']):
        X['age_quantity_interaction'] = X['pump_age'].fillna(0) * X['quantity_cat']
    
    if all(col in X.columns for col in ['log_population', 'management_cat']):
        X['pop_mgmt_interaction'] = X['log_population'] * X['management_cat']
    
    print(f"✅ Feature engineering complete: {len(X.columns)} total columns")
    return X

def select_model_features(X, y, max_features=40):
    """Select the best features for modeling"""
    print(f"🎯 Selecting features for modeling...")
    
    # Define feature categories
    numeric_features = [
        'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
        'year_recorded', 'days_since_recorded', 'pump_age', 'log_population'
    ]
    
    missing_indicators = [
        'longitude_missing', 'latitude_missing', 'gps_height_missing',
        'population_missing', 'pump_age_missing'
    ]
    
    categorical_features = [
        'quantity_cat', 'quantity_group_cat', 'waterpoint_type_cat',
        'extraction_type_cat', 'management_cat', 'payment_cat',
        'water_quality_cat', 'source_type_cat'
    ]
    
    interaction_features = [
        'age_quantity_interaction', 'pop_mgmt_interaction'
    ]
    
    # Combine all feature types
    all_features = numeric_features + missing_indicators + categorical_features + interaction_features
    
    # Keep only features that exist in the dataframe
    available_features = [col for col in all_features if col in X.columns]
    
    print(f"📊 Using {len(available_features)} features")
    
    # Return feature matrix
    X_features = X[available_features].fillna(-1)  # Fill missing with -1
    
    return X_features, available_features

class OptimizedEnsemble:
    """Fast ensemble with optimized models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """Train optimized ensemble"""
        print("🎭 Training optimized ensemble...")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 1. Random Forest (fast version)
        print("  Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        
        # 2. Gradient Boosting (fast version)
        print("  Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=20,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # 3. Logistic Regression (on scaled features)
        print("  Training Logistic Regression...")
        self.models['lr'] = LogisticRegression(
            random_state=42,
            max_iter=500,
            C=1.0,
            solver='lbfgs'
        )
        self.models['lr'].fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        print("✅ Ensemble training complete")
        return self
    
    def predict(self, X_test):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predicting")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions from each model
        rf_pred = self.models['rf'].predict(X_test)
        gb_pred = self.models['gb'].predict(X_test)
        lr_pred = self.models['lr'].predict(X_test_scaled)
        
        # Simple majority voting (faster than probability-based)
        predictions = np.column_stack([rf_pred, gb_pred, lr_pred])
        
        # Weighted voting: tree models get more weight
        weights = np.array([2, 2, 1])  # RF=2, GB=2, LR=1
        
        # For each sample, compute weighted vote
        ensemble_pred = []
        for i in range(len(predictions)):
            votes = predictions[i]
            unique_votes, counts = np.unique(votes, return_counts=True)
            
            # Weight the counts
            weighted_counts = {}
            for j, vote in enumerate(votes):
                if vote not in weighted_counts:
                    weighted_counts[vote] = 0
                weighted_counts[vote] += weights[j]
            
            # Get the vote with highest weight
            best_vote = max(weighted_counts.items(), key=lambda x: x[1])[0]
            ensemble_pred.append(best_vote)
        
        return np.array(ensemble_pred)
    
    def get_individual_predictions(self, X_test):
        """Get predictions from individual models for analysis"""
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'rf': self.models['rf'].predict(X_test),
            'gb': self.models['gb'].predict(X_test),  
            'lr': self.models['lr'].predict(X_test_scaled)
        }

def run_speed_test(df, test_size=5000):
    """Run speed test with different configurations"""
    print(f"⚡ SPEED TEST WITH {test_size:,} ROWS")
    print("=" * 50)
    
    # Use subset for speed testing
    if len(df) > test_size:
        df_test = df.sample(n=test_size, random_state=42)
    else:
        df_test = df.copy()
    
    print(f"Testing with {len(df_test):,} rows...")
    
    import time
    start_time = time.time()
    
    # 1. Feature engineering
    print("\n📊 Step 1: Feature Engineering")
    fe_start = time.time()
    X_engineered = fast_feature_engineering(df_test)
    fe_time = time.time() - fe_start
    print(f"   ⏱️  Time: {fe_time:.2f}s")
    
    # 2. Feature selection
    print("\n🎯 Step 2: Feature Selection")
    fs_start = time.time()
    y = X_engineered['status_group']
    X_features, feature_names = select_model_features(X_engineered, y)
    fs_time = time.time() - fs_start
    print(f"   ⏱️  Time: {fs_time:.2f}s")
    print(f"   📊 Features: {len(feature_names)}")
    
    # 3. Train-test split
    print("\n📊 Step 3: Data Splitting")
    split_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    split_time = time.time() - split_start
    print(f"   ⏱️  Time: {split_time:.2f}s")
    print(f"   📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # 4. Model training
    print("\n🤖 Step 4: Model Training")
    train_start = time.time()
    ensemble = OptimizedEnsemble()
    ensemble.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"   ⏱️  Time: {train_time:.2f}s")
    
    # 5. Prediction
    print("\n🎯 Step 5: Prediction")
    pred_start = time.time()
    ensemble_pred = ensemble.predict(X_test)
    individual_preds = ensemble.get_individual_predictions(X_test)
    pred_time = time.time() - pred_start
    print(f"   ⏱️  Time: {pred_time:.2f}s")
    
    # 6. Evaluation
    print("\n📈 Step 6: Evaluation")
    eval_start = time.time()
    
    # Individual model accuracies
    rf_acc = accuracy_score(y_test, individual_preds['rf'])
    gb_acc = accuracy_score(y_test, individual_preds['gb'])
    lr_acc = accuracy_score(y_test, individual_preds['lr'])
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    eval_time = time.time() - eval_start
    print(f"   ⏱️  Time: {eval_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Results
    print(f"\n🏆 RESULTS")
    print("=" * 50)
    print(f"📊 Individual Models:")
    print(f"   Random Forest:     {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"   Gradient Boosting: {gb_acc:.4f} ({gb_acc*100:.2f}%)")
    print(f"   Logistic Reg:      {lr_acc:.4f} ({lr_acc*100:.2f}%)")
    print(f"🎭 Ensemble:          {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    
    best_individual = max(rf_acc, gb_acc, lr_acc)
    ensemble_boost = ensemble_acc - best_individual
    print(f"📈 Ensemble boost:    +{ensemble_boost:.4f} ({ensemble_boost*100:.2f}pp)")
    
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"   Feature Engineering: {fe_time:.2f}s ({fe_time/total_time*100:.1f}%)")
    print(f"   Feature Selection:   {fs_time:.2f}s ({fs_time/total_time*100:.1f}%)")
    print(f"   Data Splitting:      {split_time:.2f}s ({split_time/total_time*100:.1f}%)")
    print(f"   Model Training:      {train_time:.2f}s ({train_time/total_time*100:.1f}%)")
    print(f"   Prediction:          {pred_time:.2f}s ({pred_time/total_time*100:.1f}%)")
    print(f"   Evaluation:          {eval_time:.2f}s ({eval_time/total_time*100:.1f}%)")
    print(f"   📊 TOTAL:            {total_time:.2f}s")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Processing speed:    {len(df_test)/total_time:.0f} rows/second")
    print(f"   Training speed:      {len(X_train)/train_time:.0f} rows/second")
    print(f"   Prediction speed:    {len(X_test)/pred_time:.0f} rows/second")
    
    return {
        'ensemble_accuracy': ensemble_acc,
        'individual_accuracies': {'rf': rf_acc, 'gb': gb_acc, 'lr': lr_acc},
        'total_time': total_time,
        'rows_per_second': len(df_test)/total_time,
        'ensemble_boost': ensemble_boost
    }

if __name__ == "__main__":
    print("⚡ FIXED FAST ENSEMBLE IMPLEMENTATION")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    print(f"✅ Loaded {len(train):,} total rows")
    
    # Run speed tests with different sizes
    test_sizes = [2000, 5000]
    
    results = {}
    for test_size in test_sizes:
        print(f"\n{'='*60}")
        result = run_speed_test(train, test_size=test_size)
        results[test_size] = result
    
    # Summary comparison
    print(f"\n🎯 SPEED TEST SUMMARY")
    print("=" * 60)
    print(f"{'Size':<8} {'Accuracy':<10} {'Time':<8} {'Speed':<12} {'Boost':<8}")
    print("-" * 50)
    
    for size, result in results.items():
        print(f"{size:<8} {result['ensemble_accuracy']:<10.4f} {result['total_time']:<8.1f}s "
              f"{result['rows_per_second']:<12.0f} +{result['ensemble_boost']*100:<7.2f}pp")
    
    print(f"\n💡 OPTIMIZATION EFFECTIVENESS:")
    print("-" * 40)
    avg_speed = np.mean([r['rows_per_second'] for r in results.values()])
    avg_boost = np.mean([r['ensemble_boost'] for r in results.values()])
    
    print(f"✅ Average processing speed: {avg_speed:.0f} rows/second")
    print(f"🎭 Average ensemble boost: +{avg_boost*100:.2f} percentage points")
    
    if avg_speed > 1000:
        print("🚀 Excellent speed! Ready for large datasets")
    elif avg_speed > 500:
        print("✅ Good speed! Suitable for production use")
    else:
        print("⚠️  Consider further optimization")
    
    print(f"\n🔧 KEY OPTIMIZATIONS:")
    print("-" * 40)
    print("✅ Streamlined feature engineering")
    print("✅ Focused on high-impact features only")
    print("✅ Efficient categorical encoding")
    print("✅ Reduced model complexity")
    print("✅ Fast majority voting ensemble")
    print("✅ Minimal memory allocations")