import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
test_features = pd.read_csv('ds19-predictive-modeling-challenge/test_features.csv')
sample_submission = pd.read_csv('ds19-predictive-modeling-challenge/sample_submission.csv')

# Merge training data
train = pd.merge(train_features, train_labels, on='id')
print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test_features.shape}")

# Explore target distribution
print("\nTarget distribution:")
print(train['status_group'].value_counts())
print(train['status_group'].value_counts(normalize=True))

# Feature engineering function
def engineer_features(df):
    """Enhanced feature engineering"""
    df = df.copy()
    
    # Fix latitude anomaly
    df['latitude'] = df['latitude'].replace(-2e-08, 0)
    
    # Handle zeros as missing values
    cols_with_zeros = ['longitude', 'latitude', 'construction_year', 'gps_height', 'population']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    # Time-based features
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['recording_year'] = df['date_recorded'].dt.year
    df['recording_month'] = df['date_recorded'].dt.month
    df['recording_day_of_year'] = df['date_recorded'].dt.dayofyear
    
    # Age of pump (where construction year is known)
    df['pump_age'] = df['recording_year'] - df['construction_year']
    df['pump_age'] = df['pump_age'].apply(lambda x: x if x >= 0 else np.nan)
    
    # Geographic features
    df['distance_from_origin'] = np.sqrt(df['longitude']**2 + df['latitude']**2)
    
    # Population density proxy (population per GPS unit area)
    # Avoid division by zero and infinity
    df['pop_density'] = df['population'] / (df['gps_height'].fillna(1) + 1)
    df['pop_density'] = df['pop_density'].replace([np.inf, -np.inf], np.nan)
    
    # Water quantity and quality interaction
    df['quantity_quality'] = df['quantity'] + '_' + df['quality_group']
    
    # Extraction type and management interaction
    df['extraction_management'] = df['extraction_type_class'] + '_' + df['management_group']
    
    # Payment and water quality
    df['payment_water_quality'] = df['payment_type'] + '_' + df['water_quality']
    
    # Convert boolean-like columns to numeric
    df['has_permit'] = df['permit'].map({True: 1, False: 0}).fillna(0).astype(int)
    df['public_meeting_held'] = df['public_meeting'].map({True: 1, False: 0}).fillna(0).astype(int)
    
    # Aggregate features by region
    region_stats = train.groupby('region').agg({
        'amount_tsh': ['mean', 'std'],
        'gps_height': ['mean', 'std'],
        'population': ['mean', 'std']
    }).reset_index()
    region_stats.columns = ['region'] + ['region_' + '_'.join(col) for col in region_stats.columns[1:]]
    df = df.merge(region_stats, on='region', how='left')
    
    # Installer reliability (from training data only)
    if 'status_group' in df.columns:
        installer_reliability = df.groupby('installer')['status_group'].apply(
            lambda x: (x == 'functional').mean()
        ).reset_index()
        installer_reliability.columns = ['installer', 'installer_reliability']
        df = df.merge(installer_reliability, on='installer', how='left')
    
    # Drop columns with too many unique values or no variance
    drop_cols = ['id', 'wpt_name', 'num_private', 'recorded_by', 'date_recorded', 'permit', 'public_meeting']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return df

# Apply feature engineering
print("\nEngineering features...")
train_eng = engineer_features(train)
test_eng = engineer_features(test_features)

# Handle missing installer reliability for test set
if 'installer_reliability' not in test_eng.columns:
    installer_reliability = train_eng.groupby('installer')['installer_reliability'].first().reset_index()
    test_eng = test_eng.merge(installer_reliability, on='installer', how='left')

# Prepare features and target
target = 'status_group'
features_to_drop = [target, 'status_group']
X = train_eng.drop(columns=[col for col in features_to_drop if col in train_eng.columns])
y = train_eng[target]

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify feature types
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Convert boolean features to int
for col in boolean_features:
    X[col] = X[col].astype(int)
    if col in test_eng.columns:
        test_eng[col] = test_eng[col].astype(int)

print(f"\nCategorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")

# Encode categorical variables using OrdinalEncoder for better handling
from sklearn.preprocessing import OrdinalEncoder

# Create a combined dataset for fitting encoders
all_data = pd.concat([X[categorical_features], test_eng[categorical_features]], axis=0)

# Fill missing values before encoding
for col in categorical_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('missing')
        X[col] = X[col].fillna('missing')
        if col in test_eng.columns:
            test_eng[col] = test_eng[col].fillna('missing')

# Use OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(all_data[categorical_features])

# Transform the data
X[categorical_features] = encoder.transform(X[categorical_features])
test_eng[categorical_features] = encoder.transform(test_eng[categorical_features])

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Prepare test features
X_test = test_eng[X.columns]
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Model comparison
print("\nTraining models...")

# 1. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 2. XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 3. LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-validation scores:")
for name, model in [('Random Forest', rf_model), ('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
    scores = cross_val_score(model, X_imputed, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Train final model (using best performer - typically XGBoost or LightGBM)
print("\nTraining final model...")
final_model = xgb_model
final_model.fit(X_imputed, y_encoded)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20))

# Make predictions
y_pred = final_model.predict(X_test_imputed)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create submission
submission = sample_submission.copy()
submission['status_group'] = y_pred_labels
submission.to_csv('improved_submission.csv', index=False)
print("\nSubmission saved to 'improved_submission.csv'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to 'feature_importance.csv'")