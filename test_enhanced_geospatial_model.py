#!/usr/bin/env python3
"""
Enhanced Model with GPT + Geospatial Features
=============================================
- Combines GPT features with advanced geospatial analysis
- Tests neighbor-based predictions
- Spatial clustering and density features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from geospatial_features import create_geospatial_features
import warnings
warnings.filterwarnings('ignore')

print("🚀 ENHANCED MODEL: GPT + GEOSPATIAL FEATURES")
print("=" * 60)

# Load the original data
print("📊 Loading original dataset...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Load partial GPT features
print("🤖 Loading partial GPT features...")
gpt_features = pd.read_csv('gpt_features_progress.csv')
print(f"✅ Loaded {len(gpt_features):,} GPT feature rows (~{len(gpt_features)/len(train)*100:.1f}% of dataset)")

# Merge GPT features with training data
print("🔗 Merging GPT features with training data...")
train_with_gpt = train.merge(gpt_features, on='id', how='left')

# Use only rows that have GPT features for testing
has_gpt = train_with_gpt['gpt_funder_org_type'].notna()
train_subset = train_with_gpt[has_gpt].copy()
print(f"🎯 Working with subset: {len(train_subset):,} rows")

# Create geospatial features
print("\n🌍 CREATING GEOSPATIAL FEATURES...")
print("-" * 50)
train_geo = create_geospatial_features(train_subset, n_neighbors=15, n_clusters=50)

def prepare_features(df, include_gpt=True, include_geo=True):
    """Prepare features for modeling"""
    X = df.copy()
    
    print(f"🔧 Engineering features (GPT: {include_gpt}, Geo: {include_geo})...")
    
    # Original engineered features
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
    
    # Select features
    feature_cols = []
    
    # Original numeric features
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
    
    # Text length features
    text_length_cols = [col+'_length' for col in text_cols] + [col+'_is_missing' for col in text_cols]
    for col in text_length_cols:
        if col in X.columns:
            feature_cols.append(col)
    
    # Encoded categorical features
    encoded_cols = [col+'_encoded' for col in categorical_cols]
    for col in encoded_cols:
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
                if include_geo and col not in ['geographic_cluster']:  # Skip categorical for first run
                    print(f"✅ Including geo feature: {col}")
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    print(f"📊 Total features: {len(feature_cols)}")
    return X_features, feature_cols

# Prepare different feature combinations
print("\n🔧 PREPARING FEATURE COMBINATIONS...")
print("-" * 50)

X_baseline, features_baseline = prepare_features(train_geo, include_gpt=False, include_geo=False)
X_gpt_only, features_gpt_only = prepare_features(train_geo, include_gpt=True, include_geo=False)
X_geo_only, features_geo_only = prepare_features(train_geo, include_gpt=False, include_geo=True)
X_combined, features_combined = prepare_features(train_geo, include_gpt=True, include_geo=True)

y = train_geo['status_group']

print(f"Baseline features: {len(features_baseline)}")
print(f"GPT only features: {len(features_gpt_only)}")
print(f"Geo only features: {len(features_geo_only)}")
print(f"Combined features: {len(features_combined)}")

# Split data
print("\n📊 SPLITTING DATA...")
print("-" * 40)
X_train_base, X_test_base, X_train_gpt, X_test_gpt, X_train_geo, X_test_geo, X_train_comb, X_test_comb, y_train, y_test = train_test_split(
    X_baseline, X_gpt_only, X_geo_only, X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train_base):,} rows")
print(f"Test set: {len(X_test_base):,} rows")

# Train models
print("\n🤖 TRAINING MODELS...")
print("-" * 40)

models = {}
predictions = {}

# Baseline model
print("Training baseline model...")
models['baseline'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
models['baseline'].fit(X_train_base, y_train)
predictions['baseline'] = models['baseline'].predict(X_test_base)

# GPT only model
print("Training GPT-only model...")
models['gpt_only'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
models['gpt_only'].fit(X_train_gpt, y_train)
predictions['gpt_only'] = models['gpt_only'].predict(X_test_gpt)

# Geospatial only model
print("Training geospatial-only model...")
models['geo_only'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
models['geo_only'].fit(X_train_geo, y_train)
predictions['geo_only'] = models['geo_only'].predict(X_test_geo)

# Combined model
print("Training combined model...")
models['combined'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
models['combined'].fit(X_train_comb, y_train)
predictions['combined'] = models['combined'].predict(X_test_comb)

# Calculate accuracies
print(f"\n🎯 RESULTS COMPARISON:")
print("=" * 60)

accuracies = {}
for model_name in models.keys():
    accuracies[model_name] = accuracy_score(y_test, predictions[model_name])

# Sort by accuracy
sorted_results = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

for i, (model_name, accuracy) in enumerate(sorted_results):
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
    print(f"{emoji} {model_name:15} {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculate improvements
baseline_acc = accuracies['baseline']
print(f"\n📈 IMPROVEMENTS OVER BASELINE:")
print("-" * 40)
for model_name, accuracy in sorted_results:
    if model_name != 'baseline':
        improvement = accuracy - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        print(f"{model_name:15} +{improvement:.4f} ({improvement_pct:+.2f}%)")

# Feature importance analysis for best model
best_model_name = sorted_results[0][0]
best_model = models[best_model_name]

if best_model_name == 'combined':
    feature_names = features_combined
elif best_model_name == 'geo_only':
    feature_names = features_geo_only
elif best_model_name == 'gpt_only':
    feature_names = features_gpt_only
else:
    feature_names = features_baseline

print(f"\n🔍 TOP 15 FEATURES ({best_model_name.upper()} MODEL):")
print("-" * 60)

importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
    # Categorize features
    if row['feature'].startswith('gpt_'):
        emoji = "🤖"
    elif any(geo_term in row['feature'] for geo_term in ['neighbor', 'cluster', 'spatial', 'ward', 'elevation']):
        emoji = "🌍"
    else:
        emoji = "📊"
    
    print(f"{i+1:2d}. {emoji} {row['feature']:<30} {row['importance']:.4f}")

# Analyze feature type contributions
if best_model_name == 'combined':
    gpt_importance = feature_importance_df[feature_importance_df['feature'].str.startswith('gpt_')]['importance'].sum()
    geo_importance = feature_importance_df[feature_importance_df['feature'].str.contains('neighbor|cluster|spatial|ward|elevation')]['importance'].sum()
    base_importance = 1.0 - gpt_importance - geo_importance
    
    print(f"\n📊 FEATURE TYPE CONTRIBUTIONS:")
    print("-" * 40)
    print(f"🤖 GPT Features:        {gpt_importance:.3f} ({gpt_importance*100:.1f}%)")
    print(f"🌍 Geospatial Features: {geo_importance:.3f} ({geo_importance*100:.1f}%)")
    print(f"📊 Baseline Features:   {base_importance:.3f} ({base_importance*100:.1f}%)")

# Cross-validation
print(f"\n🔄 CROSS-VALIDATION COMPARISON:")
print("-" * 40)

cv_scores = {}
for model_name, model in models.items():
    if model_name == 'baseline':
        X_cv = X_combined.iloc[:, :len(features_baseline)]
    elif model_name == 'gpt_only':
        X_cv = X_combined.iloc[:, :len(features_gpt_only)]
    elif model_name == 'geo_only':
        X_cv = X_combined.iloc[:, :len(features_geo_only)]
    else:
        X_cv = X_combined
    
    cv_score = cross_val_score(model, X_cv, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_scores[model_name] = cv_score
    print(f"{model_name:15} {cv_score.mean():.4f} ± {cv_score.std():.4f}")

print(f"\n🎯 CONCLUSION:")
print("-" * 40)
best_accuracy = sorted_results[0][1]
best_improvement = best_accuracy - baseline_acc

if best_improvement > 0.01:  # > 1 percentage point
    print(f"🎉 Excellent improvement! {best_model_name} model achieves {best_improvement*100:.2f}pp gain")
    print(f"🚀 Geospatial features are providing significant value")
elif best_improvement > 0.005:  # > 0.5 percentage point
    print(f"✅ Good improvement! {best_model_name} model achieves {best_improvement*100:.2f}pp gain")
    print(f"📈 Features are contributing meaningfully")
else:
    print(f"📊 Modest improvement: {best_improvement*100:.2f}pp gain")
    print(f"💡 May need feature tuning or more data")

print(f"\n💾 Best configuration: {best_model_name} with {best_accuracy*100:.2f}% accuracy")