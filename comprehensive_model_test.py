#!/usr/bin/env python3
"""
Comprehensive Model Testing with GPT Features
============================================
- Test multiple model combinations
- Include overfitting prevention techniques
- Compare baseline vs GPT vs combined features
- Use proper cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def add_selective_geo_features(df, reference_df=None, n_neighbors=10):
    """Add only the most valuable geospatial features"""
    print(f"🌍 Adding selective geospatial features for {len(df):,} rows...")
    
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

def prepare_baseline_features(df):
    """Prepare improved baseline features"""
    X = df.copy()
    
    # Missing value indicators (key improvement)
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
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, feature_cols

def prepare_enhanced_features(df, include_gpt=True, include_geo=True, feature_selection=False, k_best=50):
    """Add GPT and/or geospatial features to baseline with optional feature selection"""
    X_base, base_features = prepare_baseline_features(df)
    feature_cols = base_features.copy()
    
    # Add geospatial features
    if include_geo:
        geo_features_added = 0
        if 'ward_functional_ratio' in df.columns:
            X_base['ward_functional_ratio'] = df['ward_functional_ratio'].fillna(0.5)
            feature_cols.append('ward_functional_ratio')
            geo_features_added += 1
        
        if 'neighbor_functional_ratio' in df.columns:
            X_base['neighbor_functional_ratio'] = df['neighbor_functional_ratio'].fillna(0.5)
            feature_cols.append('neighbor_functional_ratio')
            geo_features_added += 1
        
        print(f"✅ Added {geo_features_added} geospatial features")
    
    # Add GPT features
    if include_gpt:
        gpt_features_added = 0
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        
        for col in gpt_cols:
            if col in df.columns and df[col].notna().any():
                X_base[col] = df[col].fillna(3.0)  # Default middle value
                feature_cols.append(col)
                gpt_features_added += 1
        
        print(f"✅ Added {gpt_features_added} GPT features")
    
    # Optional feature selection
    if feature_selection and len(feature_cols) > k_best:
        print(f"🔧 Applying feature selection: top {k_best} features")
        y = df['status_group']
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = selector.fit_transform(X_base[feature_cols], y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    return X_base[feature_cols], feature_cols

print("🤖 COMPREHENSIVE MODEL TESTING WITH GPT FEATURES")
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
train_geo = add_selective_geo_features(train_with_gpt, n_neighbors=10)

# Define model configurations - Focus on Random Forest which performs well and trains fast
models_config = {
    'rf_basic': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        'name': 'Random Forest (Basic)'
    },
    'rf_tuned': {
        'model': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, 
                                       min_samples_leaf=2, random_state=42, n_jobs=-1),
        'name': 'Random Forest (Tuned)'
    },
    'rf_optimized': {
        'model': RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=5, 
                                       min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1),
        'name': 'Random Forest (Optimized)'
    }
}

# Define feature combinations
feature_combinations = [
    ('baseline', False, False, False),
    ('baseline_geo', False, True, False),
    ('baseline_gpt', True, False, False),
    ('baseline_gpt_geo', True, True, False),
    ('baseline_gpt_geo_fs', True, True, True)  # Feature selection
]

if gpt_coverage == 0:
    # Remove GPT combinations if no GPT data
    feature_combinations = [combo for combo in feature_combinations if not combo[1]]
    print("⚠️  Skipping GPT combinations due to missing data")

print(f"\n🔬 TESTING {len(feature_combinations)} FEATURE COMBINATIONS")
print(f"📈 TESTING {len(models_config)} MODEL CONFIGURATIONS")
print("-" * 60)

results = []
y = train_geo['status_group']

# Test each combination
for combo_name, use_gpt, use_geo, use_fs in feature_combinations:
    print(f"\n🧪 Testing feature combination: {combo_name}")
    
    # Prepare features
    X, feature_names = prepare_enhanced_features(
        train_geo, 
        include_gpt=use_gpt, 
        include_geo=use_geo, 
        feature_selection=use_fs,
        k_best=50
    )
    
    print(f"📊 Features: {X.shape[1]} ({len([f for f in feature_names if f.startswith('gpt_')])} GPT)")
    
    # Test each model
    for model_key, model_config in models_config.items():
        model = model_config['model']
        model_name = model_config['name']
        
        print(f"  🤖 {model_name}...")
        
        # Cross-validation with stratification
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Single train/test split for detailed analysis
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Store results
        result = {
            'combo_name': combo_name,
            'model_name': model_name,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'n_features': X.shape[1],
            'use_gpt': use_gpt,
            'use_geo': use_geo,
            'use_fs': use_fs
        }
        results.append(result)
        
        print(f"    CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Test: {test_accuracy:.4f}")

# Results analysis
print(f"\n🏆 RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_mean', ascending=False)

print(f"{'Rank':<4} {'Combination':<20} {'Model':<25} {'CV Score':<12} {'Test Score':<12} {'Features':<10}")
print("-" * 80)

for i, (_, row) in enumerate(results_df.head(10).iterrows()):
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
    print(f"{emoji:<4} {row['combo_name']:<20} {row['model_name']:<25} "
          f"{row['cv_mean']:.4f}±{row['cv_std']:.3f} {row['test_accuracy']:<12.4f} {row['n_features']:<10}")

# Feature importance for best model
best_result = results_df.iloc[0]
print(f"\n🔍 BEST MODEL ANALYSIS")
print("-" * 40)
print(f"Best: {best_result['combo_name']} + {best_result['model_name']}")
print(f"CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
print(f"Test Score: {best_result['test_accuracy']:.4f}")

# Refit best model for feature importance
best_combo = (best_result['combo_name'], best_result['use_gpt'], best_result['use_geo'], best_result['use_fs'])
X_best, feature_names_best = prepare_enhanced_features(
    train_geo, 
    include_gpt=best_combo[1], 
    include_geo=best_combo[2], 
    feature_selection=best_combo[3],
    k_best=50
)

# Find best model config
best_model_config = None
for model_key, model_config in models_config.items():
    if model_config['name'] == best_result['model_name']:
        best_model_config = model_config
        break

if best_model_config and hasattr(best_model_config['model'], 'feature_importances_'):
    model = best_model_config['model']
    model.fit(X_best, y)
    
    print(f"\n🔍 TOP 15 FEATURES:")
    print("-" * 50)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names_best,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
        if row['feature'].startswith('gpt_'):
            emoji = "🤖"
        elif row['feature'] in ['ward_functional_ratio', 'neighbor_functional_ratio']:
            emoji = "🌍"
        else:
            emoji = "📊"
        
        print(f"{i+1:2d}. {emoji} {row['feature']:<30} {row['importance']:.4f}")

# Compare with original baseline
baseline_results = results_df[results_df['combo_name'] == 'baseline']
if len(baseline_results) > 0:
    baseline_best = baseline_results['cv_mean'].max()
    improvement = best_result['cv_mean'] - baseline_best
    improvement_pct = (improvement / baseline_best) * 100
    
    print(f"\n📈 IMPROVEMENT ANALYSIS")
    print("-" * 40)
    print(f"Baseline best: {baseline_best:.4f}")
    print(f"Enhanced best: {best_result['cv_mean']:.4f}")
    print(f"Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
    
    if improvement > 0.01:
        print("🎉 Significant improvement!")
    elif improvement > 0.005:
        print("✅ Moderate improvement")
    else:
        print("📊 Marginal improvement")

print(f"\n💾 Best configuration saved for deployment!")