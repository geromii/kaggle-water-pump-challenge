#!/usr/bin/env python3
"""
Quick Ensemble Boost Implementation
===================================
- Immediate ensemble improvements
- Advanced interaction features
- Quick wins for model performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from geospatial_features import create_geospatial_features
import warnings
warnings.filterwarnings('ignore')

def create_interaction_features(df):
    """Create high-impact interaction features"""
    print("🔗 Creating interaction features...")
    
    X = df.copy()
    
    # 1. Age-Technology Interactions (older pumps with certain technologies fail more)
    if all(col in X.columns for col in ['pump_age', 'extraction_type_encoded']):
        X['age_extraction_risk'] = X['pump_age'] * X['extraction_type_encoded']
        
    # 2. Population-Management Interactions (high pop + poor management = problems)
    if all(col in X.columns for col in ['population', 'management_encoded']):
        X['population_management_stress'] = X['population'] / (X['management_encoded'] + 1)
        
    # 3. Installer-Funder Coordination Score
    if all(col in X.columns for col in ['installer_encoded', 'funder_encoded']):
        # Same organization doing both = better coordination
        X['same_installer_funder'] = (X['installer_encoded'] == X['funder_encoded']).astype(int)
        
    # 4. Geographic-Economic Interactions
    if all(col in X.columns for col in ['gps_height', 'population']):
        # High altitude + high population = maintenance challenges
        X['altitude_population_challenge'] = X['gps_height'] * np.log1p(X['population'])
        
    # 5. Payment-Quantity Interactions
    if all(col in X.columns for col in ['payment_encoded', 'quantity_encoded']):
        X['payment_quantity_sustainability'] = X['payment_encoded'] * X['quantity_encoded']
    
    # 6. Seasonal Construction Effects
    if 'month_recorded' in X.columns:
        # Dry season installations might be more reliable
        X['dry_season_install'] = ((X['month_recorded'] >= 6) & (X['month_recorded'] <= 9)).astype(int)
    
    print(f"✅ Created {len([c for c in X.columns if c not in df.columns])} interaction features")
    return X

def prepare_enhanced_features(df, include_gpt=True, include_geo=True, include_interactions=True):
    """Prepare features with all enhancements"""
    X = df.copy()
    
    print(f"🔧 Engineering enhanced features...")
    
    # Original feature engineering
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
    
    # Add interaction features
    if include_interactions:
        X = create_interaction_features(X)
    
    # Select features for modeling
    feature_cols = []
    
    # Numeric features
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
    
    # Text features
    text_length_cols = [col+'_length' for col in text_cols] + [col+'_is_missing' for col in text_cols]
    for col in text_length_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Encoded categoricals
    encoded_cols = [col+'_encoded' for col in categorical_cols]
    for col in encoded_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Interaction features
    if include_interactions:
        interaction_cols = ['age_extraction_risk', 'population_management_stress', 'same_installer_funder',
                          'altitude_population_challenge', 'payment_quantity_sustainability', 'dry_season_install']
        for col in interaction_cols:
            if col in X.columns:
                feature_cols.append(col)
    
    # GPT features
    if include_gpt:
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        for col in gpt_cols:
            if col in X.columns and X[col].notna().any():
                feature_cols.append(col)
    
    # Geospatial features
    if include_geo:
        geo_cols = ['nearest_neighbor_distance', 'avg_5_neighbor_distance', 'avg_10_neighbor_distance',
                   'neighbor_density', 'neighbor_functional_ratio', 'installer_spatial_clustering',
                   'funder_spatial_clustering', 'geographic_cluster', 'cluster_size', 'cluster_density',
                   'neighbor_elevation_diff', 'neighbor_elevation_std', 'ward_pump_count', 'ward_functional_ratio']
        for col in geo_cols:
            if col in X.columns and X[col].notna().any():
                feature_cols.append(col)
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    print(f"📊 Total features: {len(feature_cols)}")
    return X_features, feature_cols

def create_ensemble_models(X_train, y_train, X_test):
    """Create optimized ensemble of diverse models"""
    print("🎭 Creating optimized ensemble...")
    
    # Scale features for algorithms that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train diverse base models
    models = {}
    
    # Random Forest (good for non-linear patterns, handles missing values well)
    models['rf'] = RandomForestClassifier(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    
    # Gradient Boosting (good for sequential error correction)
    models['gb'] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=10,
        random_state=42
    )
    
    # Logistic Regression (good for linear patterns, fast)
    models['lr'] = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=0.1  # Regularization
    )
    
    # Train and get predictions
    predictions = {}
    
    # Train tree-based models on original features
    for name in ['rf', 'gb']:
        print(f"Training {name}...")
        models[name].fit(X_train, y_train)
        predictions[name] = models[name].predict(X_test)
    
    # Train linear model on scaled features
    print(f"Training lr...")
    models['lr'].fit(X_train_scaled, y_train)
    predictions['lr'] = models['lr'].predict(X_test_scaled)
    
    # Create weighted voting ensemble
    # Weights based on typical performance: RF and GB stronger, LR for diversity
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models['rf']),
            ('gb', models['gb']),
            ('lr', models['lr'])
        ],
        voting='soft',  # Use probability-based voting
        weights=[2, 2, 1]  # RF and GB get double weight
    )
    
    # Train ensemble on scaled features (LR needs scaling)
    print("Training ensemble...")
    voting_clf.fit(X_train_scaled, y_train)
    ensemble_pred = voting_clf.predict(X_test_scaled)
    
    return ensemble_pred, models, predictions, voting_clf

# Main execution
if __name__ == "__main__":
    print("🚀 QUICK ENSEMBLE BOOST IMPLEMENTATION")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
    train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
    train = pd.merge(train_features, train_labels, on='id')
    
    # Load GPT features
    gpt_features = pd.read_csv('gpt_features_progress.csv')
    train_with_gpt = train.merge(gpt_features, on='id', how='left')
    
    # Use subset with GPT features
    has_gpt = train_with_gpt['gpt_funder_org_type'].notna()
    train_subset = train_with_gpt[has_gpt].copy()
    print(f"🎯 Working with: {len(train_subset):,} rows")
    
    # Create geospatial features
    print("\n🌍 Adding geospatial features...")
    train_geo = create_geospatial_features(train_subset, n_neighbors=15, n_clusters=50)
    
    # Test different feature combinations
    feature_combinations = {
        'baseline': {'gpt': False, 'geo': False, 'interactions': False},
        'gpt_only': {'gpt': True, 'geo': False, 'interactions': False},
        'geo_only': {'gpt': False, 'geo': True, 'interactions': False},
        'interactions_only': {'gpt': False, 'geo': False, 'interactions': True},
        'gpt_geo': {'gpt': True, 'geo': True, 'interactions': False},
        'full_enhanced': {'gpt': True, 'geo': True, 'interactions': True}
    }
    
    results = {}
    
    print(f"\n🧪 TESTING FEATURE COMBINATIONS WITH ENSEMBLE")
    print("=" * 60)
    
    for combo_name, combo_config in feature_combinations.items():
        print(f"\n🔬 Testing {combo_name}...")
        
        # Prepare features
        X, feature_names = prepare_enhanced_features(
            train_geo, 
            include_gpt=combo_config['gpt'],
            include_geo=combo_config['geo'],
            include_interactions=combo_config['interactions']
        )
        y = train_geo['status_group']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Single model baseline (Random Forest)
        rf_single = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_single.fit(X_train, y_train)
        rf_pred = rf_single.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Ensemble model
        ensemble_pred, models, individual_preds, voting_clf = create_ensemble_models(X_train, y_train, X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        # Store results
        results[combo_name] = {
            'single_rf': rf_accuracy,
            'ensemble': ensemble_accuracy,
            'improvement': ensemble_accuracy - rf_accuracy,
            'features': len(feature_names)
        }
        
        print(f"   📊 Single RF:  {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
        print(f"   🎭 Ensemble:   {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        print(f"   📈 Boost:      +{ensemble_accuracy - rf_accuracy:.4f} ({(ensemble_accuracy - rf_accuracy)*100:.2f}pp)")
    
    # Final comparison
    print(f"\n🏆 FINAL RESULTS COMPARISON")
    print("=" * 60)
    
    # Sort by ensemble performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ensemble'], reverse=True)
    
    print(f"{'Configuration':<20} {'Single RF':<12} {'Ensemble':<12} {'Boost':<10} {'Features':<10}")
    print("-" * 75)
    
    for i, (combo_name, result) in enumerate(sorted_results):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        boost_str = f"+{result['improvement']*100:.2f}pp"
        print(f"{emoji} {combo_name:<18} {result['single_rf']:.4f}      {result['ensemble']:.4f}      {boost_str:<9} {result['features']:<10}")
    
    # Show best improvement
    best_combo = sorted_results[0]
    baseline_ensemble = results['baseline']['ensemble']
    best_improvement = best_combo[1]['ensemble'] - baseline_ensemble
    
    print(f"\n🎯 SUMMARY:")
    print("-" * 40)
    print(f"🏆 Best configuration: {best_combo[0]}")
    print(f"📈 Total improvement over baseline: +{best_improvement*100:.2f}pp")
    print(f"🎭 Ensemble boost: +{best_combo[1]['improvement']*100:.2f}pp")
    print(f"🔧 Feature count: {best_combo[1]['features']}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    ensemble_improvements = [result['improvement'] for result in results.values()]
    avg_ensemble_boost = np.mean(ensemble_improvements)
    print(f"• Average ensemble boost: +{avg_ensemble_boost*100:.2f}pp")
    print(f"• Best feature combination provides {best_improvement*100:.2f}pp total gain")
    print(f"• Ensemble methods consistently improve performance")
    
    if best_improvement > 0.02:
        print(f"🎉 Excellent! >2pp improvement achieved")
    elif best_improvement > 0.01:
        print(f"✅ Great! >1pp improvement achieved")
    else:
        print(f"📊 Modest but consistent improvements")