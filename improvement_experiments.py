#!/usr/bin/env python3
"""
Improvement Experiments
======================
Testing 5 theories to improve on 82.27% baseline
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

print("🧪 TESTING 5 IMPROVEMENT THEORIES")
print("=" * 60)

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
test_features = pd.read_csv('ds19-predictive-modeling-challenge/test_features.csv')
train = pd.merge(train_features, train_labels, on='id')

# First, recreate the baseline selective geo model functions
def add_selective_geo_features(df, reference_df=None, n_neighbors=10):
    """Add selective geospatial features"""
    data = df.copy()
    
    # Ward-level statistics
    if 'status_group' in data.columns and 'ward' in data.columns:
        ward_stats = data.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(data['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    elif 'ward' in data.columns and reference_df is not None:
        ward_stats = reference_df.groupby('ward').agg({
            'latitude': 'count',
            'status_group': lambda x: (x == 'functional').mean()
        }).rename(columns={'latitude': 'ward_pump_count', 'status_group': 'ward_functional_ratio'})
        
        data = data.merge(ward_stats, left_on='ward', right_index=True, how='left')
        data['ward_pump_count'] = data['ward_pump_count'].fillna(ward_stats['ward_pump_count'].median())
        data['ward_functional_ratio'] = data['ward_functional_ratio'].fillna(0.5)
    
    # Neighbor functional ratio
    valid_coords = (data['latitude'] != 0) & (data['longitude'] != 0) & \
                   data['latitude'].notna() & data['longitude'].notna()
    
    if valid_coords.sum() >= n_neighbors:
        if reference_df is not None:
            ref_valid = (reference_df['latitude'] != 0) & (reference_df['longitude'] != 0)
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
            coords = data.loc[valid_coords, ['latitude', 'longitude']].values
            valid_indices = data.index[valid_coords].tolist()
            
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree', 
                                   metric='haversine', n_jobs=-1)
            nbrs.fit(np.radians(coords))
            _, indices = nbrs.kneighbors(np.radians(coords))
            indices = indices[:, 1:]
            
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

def prepare_baseline_features(df, le_dict=None):
    """Prepare baseline features"""
    X = df.copy()
    
    # Missing value indicators
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
    
    if le_dict is None:
        le_dict = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col+'_encoded'] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
    else:
        for col in categorical_cols:
            if col in X.columns and col in le_dict:
                le = le_dict[col]
                col_values = X[col].astype(str)
                # Handle unseen values by mapping to the most common class
                most_common = le.classes_[0]  # First class is usually most common
                encoded_values = []
                for val in col_values:
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        encoded_values.append(le.transform([most_common])[0])
                X[col+'_encoded'] = encoded_values
    
    # Select features
    feature_cols = []
    
    # All numeric features
    numeric_cols = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
                   'year_recorded', 'month_recorded', 'days_since_recorded', 'pump_age']
    feature_cols.extend([col for col in numeric_cols if col in X.columns])
    
    # Missing indicators
    missing_cols = [col+'_MISSING' for col in cols_with_zeros] + ['construction_year_missing']
    feature_cols.extend([col for col in missing_cols if col in X.columns])
    
    # Text features
    text_feature_cols = [col+'_length' for col in text_cols] + [col+'_is_missing' for col in text_cols]
    feature_cols.extend([col for col in text_feature_cols if col in X.columns])
    
    # Encoded categorical
    encoded_cols = [col+'_encoded' for col in categorical_cols]
    feature_cols.extend([col for col in encoded_cols if col in X.columns])
    
    # Geo features if present
    if 'ward_functional_ratio' in X.columns:
        feature_cols.append('ward_functional_ratio')
    if 'neighbor_functional_ratio' in X.columns:
        feature_cols.append('neighbor_functional_ratio')
    
    X_features = X[feature_cols].fillna(-1)
    
    return X_features, feature_cols, le_dict

# Add geo features
print("🌍 Adding geospatial features...")
train_geo = add_selective_geo_features(train, n_neighbors=10)
test_geo = add_selective_geo_features(test_features, reference_df=train_geo, n_neighbors=10)

# Prepare base features
X_base, feature_cols, le_dict = prepare_baseline_features(train_geo)
X_test_base, _, _ = prepare_baseline_features(test_geo, le_dict=le_dict)
y = train['status_group']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Features: {X_base.shape[1]}")
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

# BASELINE MODEL (for comparison)
print("\n📊 BASELINE: Selective Geo Model")
baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline_rf.fit(X_train, y_train)
baseline_score = accuracy_score(y_val, baseline_rf.predict(X_val))
print(f"Baseline validation: {baseline_score:.4f}")

# Store all experiments
experiments = []

# THEORY 1: TEMPORAL DECAY WEIGHTING
print("\n🧪 THEORY 1: Temporal Decay Weighting")
print("Hypothesis: More recent data is more reliable")

# Create days_since_recorded if not exists
if 'date_recorded' in train_geo.columns:
    train_geo['date_recorded'] = pd.to_datetime(train_geo['date_recorded'])
    train_geo['days_since_recorded'] = (pd.Timestamp('2013-12-03') - train_geo['date_recorded']).dt.days

# Create sample weights based on recency
days_since = train_geo['days_since_recorded'].fillna(train_geo['days_since_recorded'].median())
# Normalize to 0-1 and invert (so recent = higher weight)
temporal_weights = 1 - (days_since - days_since.min()) / (days_since.max() - days_since.min())
# Apply mild exponential decay
temporal_weights = np.exp(temporal_weights) / np.exp(1)

# Split weights
train_weights = temporal_weights.iloc[X_train.index]

model1 = RandomForestClassifier(n_estimators=100, random_state=43, n_jobs=-1)
model1.fit(X_train, y_train, sample_weight=train_weights)
score1 = accuracy_score(y_val, model1.predict(X_val))
print(f"Temporal weighting score: {score1:.4f} ({score1-baseline_score:+.4f})")
experiments.append(('temporal_weighting', model1, score1))

# THEORY 2: FEATURE INTERACTIONS
print("\n🧪 THEORY 2: Smart Feature Interactions")
print("Hypothesis: Key features interact in meaningful ways")

# Create interaction features for top importance features
X_interact = X_base.copy()

# Quantity × Water Quality interaction
if 'quantity_encoded' in X_interact.columns and 'water_quality_encoded' in X_interact.columns:
    X_interact['quantity_quality_interact'] = X_interact['quantity_encoded'] * X_interact['water_quality_encoded']

# Ward functional ratio × Pump age
if 'ward_functional_ratio' in X_interact.columns and 'pump_age' in X_interact.columns:
    X_interact['ward_age_interact'] = X_interact['ward_functional_ratio'] * X_interact['pump_age'].fillna(X_interact['pump_age'].median())

# Neighbor ratio × Days since recorded
if 'neighbor_functional_ratio' in X_interact.columns and 'days_since_recorded' in X_interact.columns:
    X_interact['neighbor_time_interact'] = X_interact['neighbor_functional_ratio'] * X_interact['days_since_recorded'].fillna(X_interact['days_since_recorded'].median())

X_train_int = X_interact.iloc[X_train.index]
X_val_int = X_interact.iloc[X_val.index]

model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model2.fit(X_train_int, y_train)
score2 = accuracy_score(y_val, model2.predict(X_val_int))
print(f"Feature interactions score: {score2:.4f} ({score2-baseline_score:+.4f})")
experiments.append(('feature_interactions', model2, score2))

# THEORY 3: EXTRA TREES ENSEMBLE
print("\n🧪 THEORY 3: Extra Trees Ensemble")
print("Hypothesis: More randomness reduces overfitting")

model3 = ExtraTreesClassifier(n_estimators=150, max_features='sqrt', random_state=42, n_jobs=-1)
model3.fit(X_train, y_train)
score3 = accuracy_score(y_val, model3.predict(X_val))
print(f"Extra Trees score: {score3:.4f} ({score3-baseline_score:+.4f})")
experiments.append(('extra_trees', model3, score3))

# THEORY 4: REGION-SPECIFIC PATTERNS
print("\n🧪 THEORY 4: Region-Specific Failure Patterns")
print("Hypothesis: Failure patterns vary by region")

# Add region-specific failure rates as features
X_region = X_base.copy()

if 'region_encoded' in train_geo.columns:
    region_stats = train_geo.groupby('region_encoded')['status_group'].apply(
        lambda x: pd.Series({
            'region_functional_rate': (x == 'functional').mean(),
            'region_repair_rate': (x == 'functional needs repair').mean()
        })
    ).reset_index()
    
    # Merge back
    region_map = dict(zip(region_stats['region_encoded'], region_stats['region_functional_rate']))
    X_region['region_functional_rate'] = train_geo['region_encoded'].map(region_map).fillna(0.5)

X_train_reg = X_region.iloc[X_train.index]
X_val_reg = X_region.iloc[X_val.index]

model4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model4.fit(X_train_reg, y_train)
score4 = accuracy_score(y_val, model4.predict(X_val_reg))
print(f"Region patterns score: {score4:.4f} ({score4-baseline_score:+.4f})")
experiments.append(('region_patterns', model4, score4))

# THEORY 5: BALANCED CLASS WEIGHTS
print("\n🧪 THEORY 5: Balanced Class Weights")
print("Hypothesis: Better handling of minority classes")

model5 = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', 
                                random_state=42, n_jobs=-1)
model5.fit(X_train, y_train)
score5 = accuracy_score(y_val, model5.predict(X_val))
print(f"Balanced weights score: {score5:.4f} ({score5-baseline_score:+.4f})")
experiments.append(('balanced_weights', model5, score5))

# Get top 3 experiments
experiments.sort(key=lambda x: x[2], reverse=True)
top3 = experiments[:3]

print("\n🏆 TOP 3 APPROACHES:")
print("-" * 40)
for i, (name, model, score) in enumerate(top3):
    print(f"{i+1}. {name}: {score:.4f}")

# ENSEMBLE THE TOP 3
print("\n🎭 ENSEMBLE METHODS FOR TOP 3:")
print("-" * 40)

# Get predictions from top 3
preds = []
for name, model, _ in top3:
    if name == 'temporal_weighting':
        pred = model.predict(X_val)
    elif name == 'feature_interactions':
        pred = model.predict(X_val_int)
    elif name == 'region_patterns':
        pred = model.predict(X_val_reg)
    else:
        pred = model.predict(X_val)
    preds.append(pred)

# Method 1: Simple Majority Voting
from collections import Counter
pred_majority = []
for votes in zip(*preds):
    vote_counts = Counter(votes)
    majority_vote = vote_counts.most_common(1)[0][0]
    pred_majority.append(majority_vote)
pred_majority = np.array(pred_majority)
score_majority = accuracy_score(y_val, pred_majority)
print(f"1. Majority voting: {score_majority:.4f} ({score_majority-baseline_score:+.4f})")

# Method 2: Weighted Voting (by accuracy)
weights = [score for _, _, score in top3]
weights = np.array(weights) / sum(weights)

# Convert predictions to numeric for weighted average
label_map = {'functional': 0, 'functional needs repair': 1, 'non functional': 2}
reverse_map = {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}

preds_numeric = []
for pred in preds:
    pred_num = np.array([label_map[p] for p in pred])
    preds_numeric.append(pred_num)

weighted_avg = np.average(preds_numeric, axis=0, weights=weights)
pred_weighted = [reverse_map[int(round(w))] for w in weighted_avg]
score_weighted = accuracy_score(y_val, pred_weighted)
print(f"2. Weighted voting: {score_weighted:.4f} ({score_weighted-baseline_score:+.4f})")

# Method 3: Probability-based ensemble (if we have probabilities)
print("\n🔮 Getting probability predictions...")
probs = []
for name, model, _ in top3:
    if name == 'temporal_weighting':
        prob = model.predict_proba(X_val)
    elif name == 'feature_interactions':
        prob = model.predict_proba(X_val_int)
    elif name == 'region_patterns':
        prob = model.predict_proba(X_val_reg)
    else:
        prob = model.predict_proba(X_val)
    probs.append(prob)

# Average probabilities
avg_probs = np.mean(probs, axis=0)
pred_prob = [model.classes_[i] for i in np.argmax(avg_probs, axis=1)]
score_prob = accuracy_score(y_val, pred_prob)
print(f"3. Probability averaging: {score_prob:.4f} ({score_prob-baseline_score:+.4f})")

# Find best ensemble method
best_ensemble_score = max(score_majority, score_weighted, score_prob)
best_ensemble_name = "Majority" if score_majority == best_ensemble_score else \
                    "Weighted" if score_weighted == best_ensemble_score else "Probability"

print(f"\n✨ BEST ENSEMBLE: {best_ensemble_name} voting with {best_ensemble_score:.4f}")
print(f"📈 Total improvement: {best_ensemble_score-baseline_score:+.4f} ({(best_ensemble_score-baseline_score)*100:+.2f}pp)")

# If improvement is significant, create final submission
if best_ensemble_score > baseline_score + 0.002:
    print("\n🚀 CREATING ENHANCED SUBMISSION...")
    
    # Train top 3 models on full data
    final_models = []
    
    for name, _, _ in top3:
        print(f"Training {name} on full dataset...")
        
        if name == 'temporal_weighting':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_base, y, sample_weight=temporal_weights)
            final_models.append((name, model, 'base'))
            
        elif name == 'feature_interactions':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_interact, y)
            final_models.append((name, model, 'interact'))
            
        elif name == 'extra_trees':
            model = ExtraTreesClassifier(n_estimators=150, max_features='sqrt', random_state=42, n_jobs=-1)
            model.fit(X_base, y)
            final_models.append((name, model, 'base'))
            
        elif name == 'region_patterns':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_region, y)
            final_models.append((name, model, 'region'))
            
        else:  # balanced_weights
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', 
                                          random_state=42, n_jobs=-1)
            model.fit(X_base, y)
            final_models.append((name, model, 'base'))
    
    # Get test predictions
    test_probs = []
    for name, model, feat_type in final_models:
        print(f"Predicting with {name}...")
        
        if feat_type == 'interact':
            X_test_interact = X_test_base.copy()
            if 'quantity_encoded' in X_test_interact.columns and 'water_quality_encoded' in X_test_interact.columns:
                X_test_interact['quantity_quality_interact'] = X_test_interact['quantity_encoded'] * X_test_interact['water_quality_encoded']
            if 'ward_functional_ratio' in X_test_interact.columns and 'pump_age' in X_test_interact.columns:
                X_test_interact['ward_age_interact'] = X_test_interact['ward_functional_ratio'] * X_test_interact['pump_age'].fillna(X_test_interact['pump_age'].median())
            if 'neighbor_functional_ratio' in X_test_interact.columns and 'days_since_recorded' in X_test_interact.columns:
                X_test_interact['neighbor_time_interact'] = X_test_interact['neighbor_functional_ratio'] * X_test_interact['days_since_recorded'].fillna(X_test_interact['days_since_recorded'].median())
            prob = model.predict_proba(X_test_interact)
            
        elif feat_type == 'region':
            X_test_region = X_test_base.copy()
            if 'region_encoded' in test_geo.columns:
                X_test_region['region_functional_rate'] = test_geo['region_encoded'].map(region_map).fillna(0.5)
            prob = model.predict_proba(X_test_region)
            
        else:
            prob = model.predict_proba(X_test_base)
            
        test_probs.append(prob)
    
    # Ensemble predictions
    if best_ensemble_name == "Probability":
        avg_test_probs = np.mean(test_probs, axis=0)
        test_predictions = [final_models[0][1].classes_[i] for i in np.argmax(avg_test_probs, axis=1)]
    else:
        # Get individual predictions for voting
        test_preds = []
        for prob, (_, model, _) in zip(test_probs, final_models):
            pred = [model.classes_[i] for i in np.argmax(prob, axis=1)]
            test_preds.append(pred)
        
        if best_ensemble_name == "Majority":
            test_predictions = []
            for votes in zip(*test_preds):
                vote_counts = Counter(votes)
                majority_vote = vote_counts.most_common(1)[0][0]
                test_predictions.append(majority_vote)
        else:  # Weighted
            # Use validation scores as weights
            test_preds_numeric = []
            for pred in test_preds:
                pred_num = np.array([label_map[p] for p in pred])
                test_preds_numeric.append(pred_num)
            
            weighted_test_avg = np.average(test_preds_numeric, axis=0, weights=weights)
            test_predictions = [reverse_map[int(round(w))] for w in weighted_test_avg]
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_features['id'],
        'status_group': test_predictions
    })
    
    submission.to_csv('enhanced_ensemble_submission.csv', index=False)
    print(f"✅ Enhanced submission saved: enhanced_ensemble_submission.csv")
    print(f"📊 Expected improvement: ~{(best_ensemble_score-baseline_score)*100:.2f}pp")
else:
    print("\n⚠️  Ensemble improvement too small to justify complexity")