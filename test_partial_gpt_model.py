#!/usr/bin/env python3
"""
Test Improved Model with Partial GPT Features
=============================================
- Uses ~25% of GPT features we have so far
- Tests model performance improvement
- Compares against baseline without GPT features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 TESTING IMPROVED MODEL WITH PARTIAL GPT FEATURES")
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

# Check how many rows have GPT features
has_gpt = train_with_gpt['gpt_funder_org_type'].notna()
print(f"📈 Rows with GPT features: {has_gpt.sum():,} / {len(train_with_gpt):,} ({has_gpt.mean()*100:.1f}%)")

# Use only rows that have GPT features for testing
train_subset = train_with_gpt[has_gpt].copy()
print(f"🎯 Testing on subset: {len(train_subset):,} rows")

def prepare_features(df, include_gpt=True):
    """Prepare features for modeling"""
    X = df.copy()
    
    # Original engineered features (from our improved model)
    print("🔧 Engineering original features...")
    
    # Missing value indicators for zero-filled columns
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
    
    # Select numeric features for modeling
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
    
    # GPT features (if available and requested)
    if include_gpt:
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        for col in gpt_cols:
            if col in X.columns and X[col].notna().any():
                feature_cols.append(col)
                print(f"✅ Including GPT feature: {col}")
    
    # Fill missing values
    X_features = X[feature_cols].fillna(-1)
    
    print(f"📊 Total features: {len(feature_cols)}")
    return X_features, feature_cols

# Prepare features with and without GPT
print("\n🔧 PREPARING FEATURES...")
print("-" * 40)

X_with_gpt, features_with_gpt = prepare_features(train_subset, include_gpt=True)
X_without_gpt, features_without_gpt = prepare_features(train_subset, include_gpt=False)
y = train_subset['status_group']

print(f"Features with GPT: {len(features_with_gpt)}")
print(f"Features without GPT: {len(features_without_gpt)}")
print(f"GPT features added: {len(features_with_gpt) - len(features_without_gpt)}")

# Split data
print("\n📊 SPLITTING DATA...")
print("-" * 40)
X_train_gpt, X_test_gpt, X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_with_gpt, X_without_gpt, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train_gpt):,} rows")
print(f"Test set: {len(X_test_gpt):,} rows")

# Train models
print("\n🤖 TRAINING MODELS...")
print("-" * 40)

# Baseline model (without GPT features)
print("Training baseline model (no GPT features)...")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_base, y_train)

# Enhanced model (with GPT features)
print("Training enhanced model (with GPT features)...")
rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_enhanced.fit(X_train_gpt, y_train)

# Make predictions
print("\n📈 MAKING PREDICTIONS...")
print("-" * 40)

y_pred_baseline = rf_baseline.predict(X_test_base)
y_pred_enhanced = rf_enhanced.predict(X_test_gpt)

# Calculate accuracies
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
enhanced_accuracy = accuracy_score(y_test, y_pred_enhanced)

print(f"🎯 RESULTS ON {len(X_test_gpt):,} TEST SAMPLES:")
print("=" * 60)
print(f"📊 Baseline Model (no GPT):     {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"🚀 Enhanced Model (with GPT):   {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
print(f"📈 Improvement:                 {enhanced_accuracy - baseline_accuracy:.4f} ({(enhanced_accuracy - baseline_accuracy)*100:.2f} percentage points)")

if enhanced_accuracy > baseline_accuracy:
    improvement_pct = ((enhanced_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    print(f"🎉 Relative improvement:        {improvement_pct:.2f}%")
else:
    print(f"⚠️  No improvement detected")

# Cross-validation for more robust evaluation
print(f"\n🔄 CROSS-VALIDATION (5-fold):")
print("-" * 40)

cv_baseline = cross_val_score(rf_baseline, X_with_gpt.iloc[:, :-6], y, cv=5, scoring='accuracy')  # Exclude GPT features
cv_enhanced = cross_val_score(rf_enhanced, X_with_gpt, y, cv=5, scoring='accuracy')

print(f"Baseline CV Score:  {cv_baseline.mean():.4f} ± {cv_baseline.std():.4f}")
print(f"Enhanced CV Score:  {cv_enhanced.mean():.4f} ± {cv_enhanced.std():.4f}")
print(f"CV Improvement:     {cv_enhanced.mean() - cv_baseline.mean():.4f}")

# Feature importance analysis
print(f"\n🔍 TOP GPT FEATURE IMPORTANCE:")
print("-" * 40)

feature_names = features_with_gpt
importances = rf_enhanced.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Show top 10 features overall
print("Top 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
    gpt_indicator = "🤖" if row['feature'].startswith('gpt_') else "📊"
    print(f"{i+1:2d}. {gpt_indicator} {row['feature']:<25} {row['importance']:.4f}")

# Show GPT feature importance specifically
gpt_features_importance = feature_importance_df[feature_importance_df['feature'].str.startswith('gpt_')]
if not gpt_features_importance.empty:
    print(f"\nGPT Features Ranking:")
    for i, (_, row) in enumerate(gpt_features_importance.iterrows()):
        print(f"{i+1}. {row['feature']:<25} {row['importance']:.4f}")
    
    total_gpt_importance = gpt_features_importance['importance'].sum()
    print(f"\nTotal GPT feature importance: {total_gpt_importance:.4f} ({total_gpt_importance*100:.1f}% of model)")

# Detailed classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
print("-" * 60)
print("Enhanced Model Performance:")
print(classification_report(y_test, y_pred_enhanced))

print(f"\n🎯 CONCLUSION:")
print("-" * 40)
if enhanced_accuracy > baseline_accuracy:
    print(f"✅ GPT features are providing value!")
    print(f"📈 Even with only ~25% of data, we see {(enhanced_accuracy - baseline_accuracy)*100:.2f}pp improvement")
    print(f"🚀 Full dataset should provide even better results")
else:
    print(f"⚠️  GPT features not showing improvement yet")
    print(f"💡 May need more data or feature engineering")

print(f"\n💾 Recommendation: {'Continue full dataset generation!' if enhanced_accuracy > baseline_accuracy else 'Analyze feature quality and continue generation'}")