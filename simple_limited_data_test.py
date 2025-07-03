#!/usr/bin/env python3
"""
Simple Limited Data Test
========================
Test if GPT features help with small training sets
Focus on core question without complex geospatial features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
import sqlite3
warnings.filterwarnings('ignore')

print("🧪 SIMPLE TEST: GPT FEATURES WITH LIMITED DATA")
print("=" * 60)

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Load GPT features
print("🤖 Loading GPT features...")
try:
    conn = sqlite3.connect('gpt_features_progress.db')
    gpt_features = pd.read_sql_query('''
        SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
               gpt_scheme_pattern, gpt_text_language, gpt_coordination 
        FROM gpt_features ORDER BY id
    ''', conn)
    conn.close()
    print(f"✅ GPT features loaded: {len(gpt_features):,} rows")
except:
    print("❌ No GPT features found")
    exit(1)

# Merge GPT features
train = train.merge(gpt_features, on='id', how='left')

def prepare_simple_features(df, include_gpt=False, le_dict=None):
    """Prepare simplified features"""
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
    
    # Construction year features
    if 'construction_year' in X.columns:
        X['construction_year'] = pd.to_numeric(X['construction_year'], errors='coerce')
        X['pump_age'] = 2013 - X['construction_year']
        X['construction_year_missing'] = X['construction_year'].isnull().astype(int)
    
    # Key categorical features (most important ones)
    key_categorical_cols = ['basin', 'region', 'ward', 'public_meeting', 'permit',
                           'extraction_type', 'management', 'payment', 'water_quality', 
                           'quantity', 'source', 'waterpoint_type']
    
    if le_dict is None:
        le_dict = {}
        for col in key_categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
    else:
        for col in key_categorical_cols:
            if col in X.columns and col in le_dict:
                le = le_dict[col]
                col_values = X[col].astype(str)
                encoded_values = []
                for val in col_values:
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        encoded_values.append(0)  # Map unknown to 0
                X[col+'_encoded'] = encoded_values
    
    # Select features
    feature_cols = []
    
    # Numeric features
    numeric_cols = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
                   'year_recorded', 'month_recorded', 'pump_age']
    feature_cols.extend([col for col in numeric_cols if col in X.columns])
    
    # Missing indicators
    missing_cols = [col+'_MISSING' for col in cols_with_zeros] + ['construction_year_missing']
    feature_cols.extend([col for col in missing_cols if col in X.columns])
    
    # Encoded categorical
    encoded_cols = [col+'_encoded' for col in key_categorical_cols]
    feature_cols.extend([col for col in encoded_cols if col in X.columns])
    
    # GPT features if requested
    gpt_added = 0
    if include_gpt:
        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
        for col in gpt_cols:
            if col in X.columns and X[col].notna().any():
                X[col] = X[col].fillna(3.0)
                feature_cols.append(col)
                gpt_added += 1
    
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, le_dict, gpt_added

# Test different training set sizes with multiple samples each
training_sizes = [0.01, 0.02, 0.05, 0.10, 0.20]  # 1%, 2%, 5%, 10%, 20%
n_samples = 10  # Test 3 different samples for each percentage

print(f"\n📊 TESTING TRAINING SET SIZES (3 samples each)")
print("=" * 60)

y = train['status_group']
all_results = []

for train_size in training_sizes:
    print(f"\n🎯 Training with {train_size*100:.0f}% of data ({int(len(train)*train_size):,} samples)")
    print("-" * 50)
    
    size_results = []
    
    for sample_num in range(n_samples):
        print(f"   Sample {sample_num + 1}:")
        
        # Create different training set for each sample
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=42 + sample_num)
        train_idx, remaining_idx = next(splitter.split(train, y))
        
        # Use fixed validation set from remaining data
        np.random.seed(42 + sample_num)
        val_idx = np.random.choice(remaining_idx, min(5000, len(remaining_idx)), replace=False)
        
        train_subset = train.iloc[train_idx]
        val_subset = train.iloc[val_idx]
        y_train_subset = y.iloc[train_idx]
        y_val_subset = y.iloc[val_idx]
        
        # Test baseline (no GPT)
        X_train_base, le_dict, _ = prepare_simple_features(train_subset, include_gpt=False)
        X_val_base, _, _ = prepare_simple_features(val_subset, include_gpt=False, le_dict=le_dict)
        
        model_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_base.fit(X_train_base, y_train_subset)
        pred_base = model_base.predict(X_val_base)
        score_base = accuracy_score(y_val_subset, pred_base)
        
        # Test with GPT features
        X_train_gpt, le_dict_gpt, gpt_count = prepare_simple_features(train_subset, include_gpt=True)
        X_val_gpt, _, _ = prepare_simple_features(val_subset, include_gpt=True, le_dict=le_dict_gpt)
        
        model_gpt = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_gpt.fit(X_train_gpt, y_train_subset)
        pred_gpt = model_gpt.predict(X_val_gpt)
        score_gpt = accuracy_score(y_val_subset, pred_gpt)
        
        improvement = score_gpt - score_base
        
        print(f"     Baseline: {score_base:.4f} | GPT: {score_gpt:.4f} | Δ: {improvement:+.4f}")
        
        result = {
            'train_size': train_size,
            'sample_num': sample_num + 1,
            'train_samples': len(train_subset),
            'baseline_score': score_base,
            'gpt_score': score_gpt,
            'improvement': improvement,
            'baseline_features': X_train_base.shape[1],
            'gpt_features': X_train_gpt.shape[1],
            'gpt_count': gpt_count
        }
        
        size_results.append(result)
        all_results.append(result)
    
    # Calculate average for this size
    avg_baseline = np.mean([r['baseline_score'] for r in size_results])
    avg_gpt = np.mean([r['gpt_score'] for r in size_results])
    avg_improvement = np.mean([r['improvement'] for r in size_results])
    std_improvement = np.std([r['improvement'] for r in size_results])
    
    print(f"   📊 Average: Baseline={avg_baseline:.4f}, GPT={avg_gpt:.4f}")
    print(f"   📊 Improvement: {avg_improvement:+.4f} ± {std_improvement:.4f}")
    
    if avg_improvement > 0.003:
        print(f"   🎉 GPT consistently helps!")
    elif avg_improvement > 0.001:
        print(f"   ⚠️  GPT provides small improvement")
    else:
        print(f"   ❌ GPT doesn't help")

# Add 100% test for comparison
print(f"\n🎯 Training with 100% of data (for comparison)")
print("-" * 50)
train_subset, val_subset, y_train_subset, y_val_subset = train_test_split(
    train, y, test_size=0.2, random_state=42, stratify=y
)

X_train_base, le_dict, _ = prepare_simple_features(train_subset, include_gpt=False)
X_val_base, _, _ = prepare_simple_features(val_subset, include_gpt=False, le_dict=le_dict)

model_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_base.fit(X_train_base, y_train_subset)
pred_base = model_base.predict(X_val_base)
score_base_100 = accuracy_score(y_val_subset, pred_base)

X_train_gpt, le_dict_gpt, gpt_count = prepare_simple_features(train_subset, include_gpt=True)
X_val_gpt, _, _ = prepare_simple_features(val_subset, include_gpt=True, le_dict=le_dict_gpt)

model_gpt = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_gpt.fit(X_train_gpt, y_train_subset)
pred_gpt = model_gpt.predict(X_val_gpt)
score_gpt_100 = accuracy_score(y_val_subset, pred_gpt)

improvement_100 = score_gpt_100 - score_base_100
print(f"   Baseline: {score_base_100:.4f} | GPT: {score_gpt_100:.4f} | Δ: {improvement_100:+.4f}")

all_results.append({
    'train_size': 1.0,
    'sample_num': 1,
    'train_samples': len(train_subset),
    'baseline_score': score_base_100,
    'gpt_score': score_gpt_100,
    'improvement': improvement_100,
    'baseline_features': X_train_base.shape[1],
    'gpt_features': X_train_gpt.shape[1],
    'gpt_count': gpt_count
})

# Summary - Group by training size and average
print(f"\n📈 SUMMARY BY TRAINING SIZE")
print("=" * 70)
print(f"{'Size':<6} {'Samples':<8} {'Baseline':<12} {'GPT':<12} {'Improvement':<15} {'Verdict'}")
print("-" * 70)

# Group results by training size
size_groups = {}
for result in all_results:
    size = result['train_size']
    if size not in size_groups:
        size_groups[size] = []
    size_groups[size].append(result)

# Calculate averages for each size
for size in sorted(size_groups.keys()):
    group = size_groups[size]
    
    avg_baseline = np.mean([r['baseline_score'] for r in group])
    avg_gpt = np.mean([r['gpt_score'] for r in group])
    avg_improvement = np.mean([r['improvement'] for r in group])
    std_improvement = np.std([r['improvement'] for r in group]) if len(group) > 1 else 0
    
    size_str = f"{size*100:.0f}%"
    samples_str = f"{group[0]['train_samples']:,}"
    baseline_str = f"{avg_baseline:.4f}"
    gpt_str = f"{avg_gpt:.4f}"
    
    if len(group) > 1:
        improvement_str = f"{avg_improvement:+.4f}±{std_improvement:.3f}"
    else:
        improvement_str = f"{avg_improvement:+.4f}"
    
    if avg_improvement > 0.003:
        verdict = "🎉 Strong"
    elif avg_improvement > 0.001:
        verdict = "⚠️ Weak"
    else:
        verdict = "❌ None"
    
    print(f"{size_str:<6} {samples_str:<8} {baseline_str:<12} {gpt_str:<12} {improvement_str:<15} {verdict}")

# Analysis
positive_results = []
for size, group in size_groups.items():
    avg_improvement = np.mean([r['improvement'] for r in group])
    if avg_improvement > 0.001:
        positive_results.append((size, avg_improvement))
if positive_results:
    best_size, best_improvement = max(positive_results, key=lambda x: x[1])
    print(f"\n🎯 FINDINGS:")
    print(f"✅ GPT features help most with {best_size*100:.0f}% data")
    print(f"   Best improvement: {best_improvement*100:+.2f}pp")
    
    # Find crossover point
    negative_sizes = [size for size, group in size_groups.items() 
                     if np.mean([r['improvement'] for r in group]) <= 0]
    if negative_sizes:
        crossover = min(negative_sizes)
        print(f"   GPT stops helping at ≥{crossover*100:.0f}% training data")
    
    # Show the pattern
    print(f"\n📊 PATTERN:")
    for size, improvement in sorted(positive_results):
        print(f"   {size*100:2.0f}% data: {improvement*100:+.2f}pp improvement")
        
else:
    print(f"\n🎯 FINDINGS:")
    print(f"❌ GPT features never provide meaningful improvement")
    print(f"   Even with very limited training data")

# Data efficiency insight
print(f"\n💡 INSIGHT:")
if positive_results:
    print(f"✅ GPT features act as useful priors when training data is very scarce")
    print(f"   They help overcome sparse categorical encodings with limited data")
    print(f"   But become noise when sufficient data makes categorical encodings reliable")
    
    # Calculate the efficiency difference
    small_data_sizes = [size for size, _ in positive_results if size <= 0.05]
    if small_data_sizes:
        print(f"   Sweet spot: ≤{max(small_data_sizes)*100:.0f}% training data")
else:
    print(f"The categorical encodings are so effective that GPT features never help")
    print(f"This suggests the original features capture the information very well")

print("\n✅ Limited data analysis complete!")