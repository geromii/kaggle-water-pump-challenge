import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import os

# Import the feature engineering function from our previous model
from improved_water_pump_model import engineer_features, label_encoder

print("Loading data...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
test_features = pd.read_csv('ds19-predictive-modeling-challenge/test_features.csv')

# Merge training data
train = pd.merge(train_features, train_labels, on='id')

# Check if GPT features exist
gpt_features_path = 'gpt_features_full.csv'
if os.path.exists(gpt_features_path):
    print(f"\nLoading GPT features from {gpt_features_path}")
    gpt_features = pd.read_csv(gpt_features_path)
    
    # Merge GPT features with training data
    train = train.merge(gpt_features, on='id', how='left')
    
    # For test data, we'll need to generate GPT features too
    # For now, we'll use median values as placeholders
    gpt_cols = [col for col in gpt_features.columns if col.startswith('gpt_')]
    for col in gpt_cols:
        test_features[col] = gpt_features[col].median()
    
    print(f"Added {len(gpt_cols)} GPT features: {gpt_cols}")
else:
    print("\nNo GPT features found. Run 'run_gpt_test.py' first, then generate full features.")
    gpt_cols = []

# Apply feature engineering
print("\nEngineering features...")
train_eng = engineer_features(train)
test_eng = engineer_features(test_features)

# If GPT features were not part of feature engineering, add them now
for col in gpt_cols:
    if col not in train_eng.columns and col in train.columns:
        train_eng[col] = train[col]
    if col not in test_eng.columns and col in test_features.columns:
        test_eng[col] = test_features[col]

# Prepare for modeling
target = 'status_group'
X = train_eng.drop(columns=[target, 'id'], errors='ignore')
y = train_eng[target]

# Encode target
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Prepare test features
X_test = test_eng[X.columns]

# Handle categorical features
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

print(f"\nFeatures: {len(X.columns)} total")
print(f"- Categorical: {len(categorical_features)}")
print(f"- Numerical: {len(numerical_features)}")
if gpt_cols:
    print(f"- GPT features included: {gpt_cols}")

# Encode categoricals
all_data = pd.concat([X[categorical_features], X_test[categorical_features]], axis=0)
for col in categorical_features:
    all_data[col] = all_data[col].fillna('missing')
    X[col] = X[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(all_data[categorical_features])
X[categorical_features] = encoder.transform(X[categorical_features])
X_test[categorical_features] = encoder.transform(X_test[categorical_features])

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

# Model comparison
print("\nTraining models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, 
                                          min_samples_leaf=2, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                    num_leaves=31, subsample=0.8, colsample_bytree=0.8, 
                                    random_state=42, n_jobs=-1)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_model = None
best_model_name = None

print("\nCross-validation scores:")
for name, model in models.items():
    scores = cross_val_score(model, X_imputed, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_score = scores.mean()
    print(f"{name}: {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with {best_score:.4f} accuracy")

# Train final model
print("\nTraining final model...")
best_model.fit(X_imputed, y_encoded)

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 most important features:")
    print(feature_importance.head(20))
    
    # Check GPT feature importance
    if gpt_cols:
        print("\nGPT feature importance:")
        gpt_importance = feature_importance[feature_importance['feature'].isin(gpt_cols)]
        print(gpt_importance)

# Make predictions
y_pred = best_model.predict(X_test_imputed)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Save submission
sample_submission = pd.read_csv('ds19-predictive-modeling-challenge/sample_submission.csv')
submission = sample_submission.copy()
submission['status_group'] = y_pred_labels
submission.to_csv('submission_with_gpt.csv', index=False)
print("\nSubmission saved to 'submission_with_gpt.csv'")