import pandas as pd
import numpy as np

# Load the test results
gpt_features = pd.read_csv('gpt_features_test.csv')

# Load original data to compare
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Merge with GPT features
test_data = train[train['id'].isin(gpt_features['id'])].merge(gpt_features, on='id')

print("=== GPT FEATURE ANALYSIS ===\n")

# 1. Funder Organization Type
print("1. FUNDER ORGANIZATION TYPE Distribution:")
print(test_data['gpt_funder_org_type'].value_counts().sort_index())
print("\nSample mappings:")
funder_samples = test_data[['funder', 'gpt_funder_org_type']].drop_duplicates().head(10)
for _, row in funder_samples.iterrows():
    org_types = {1: 'Government', 2: 'International', 3: 'NGO', 4: 'Religious', 5: 'Private'}
    funder_text = str(row['funder'])[:30] if pd.notna(row['funder']) else 'NaN'
    print(f"  {funder_text:30} -> {org_types[row['gpt_funder_org_type']]}")

# 2. Installer Name Quality
print("\n\n2. INSTALLER NAME QUALITY Distribution:")
print(test_data['gpt_installer_name_quality'].value_counts().sort_index())
print("\nSample mappings:")
installer_samples = test_data[['installer', 'gpt_installer_name_quality']].drop_duplicates().head(10)
for _, row in installer_samples.iterrows():
    quality = {1: 'Very Poor', 2: 'Poor', 3: 'Fair', 4: 'Good', 5: 'Excellent'}
    installer_text = str(row['installer'])[:30] if pd.notna(row['installer']) else 'NaN'
    print(f"  {installer_text:30} -> {quality[row['gpt_installer_name_quality']]}")

# 3. Location Institution Type
print("\n\n3. LOCATION INSTITUTION TYPE Distribution:")
print(test_data['gpt_location_institution_type'].value_counts().sort_index())
print("\nInteresting finds:")
non_village = test_data[test_data['gpt_location_institution_type'] < 5][['ward', 'subvillage', 'gpt_location_institution_type']].head(5)
for _, row in non_village.iterrows():
    inst_types = {1: 'School', 2: 'Religious', 3: 'Health', 4: 'Government', 5: 'Village'}
    subvillage_text = str(row['subvillage'])[:30] if pd.notna(row['subvillage']) else 'NaN'
    print(f"  {subvillage_text:30} -> {inst_types[row['gpt_location_institution_type']]}")

# 4. Scheme Name Pattern
print("\n\n4. SCHEME NAME PATTERN Distribution:")
print(test_data['gpt_scheme_name_pattern'].value_counts().sort_index())
print("\nSample mappings:")
scheme_samples = test_data[['scheme_name', 'gpt_scheme_name_pattern']].dropna().drop_duplicates().head(10)
for _, row in scheme_samples.iterrows():
    patterns = {1: 'Invalid', 2: 'Code Only', 3: 'Location', 4: 'Descriptive', 5: 'Full Name'}
    print(f"  {str(row['scheme_name'])[:30]:30} -> {patterns[row['gpt_scheme_name_pattern']]}")

# 5. Text Language
print("\n\n5. TEXT LANGUAGE Distribution:")
print(test_data['gpt_text_language'].value_counts().sort_index())
print("\nLanguage by region:")
language_by_region = test_data.groupby('region')['gpt_text_language'].mean().sort_values()
print(language_by_region.head(10))

# 6. Name Mismatch Flag (if exists)
if 'gpt_name_mismatch_flag' in test_data.columns:
    print("\n\n6. NAME MISMATCH FLAG Distribution:")
    print(test_data['gpt_name_mismatch_flag'].value_counts().sort_index())
    print("\nSample mappings:")
    mismatch_samples = test_data[['funder', 'installer', 'gpt_name_mismatch_flag']].drop_duplicates().head(10)
    for _, row in mismatch_samples.iterrows():
        mismatch_types = {1: 'Perfect', 2: 'Good', 3: 'Neutral', 4: 'Poor', 5: 'Red Flag'}
        funder_text = str(row['funder'])[:20] if pd.notna(row['funder']) else 'NaN'
        installer_text = str(row['installer'])[:20] if pd.notna(row['installer']) else 'NaN'
        print(f"  {funder_text:20} + {installer_text:20} -> {mismatch_types[row['gpt_name_mismatch_flag']]}")

# Check correlation with pump status
print("\n\n=== CORRELATION WITH PUMP STATUS ===")
status_mapping = {'functional': 2, 'functional needs repair': 1, 'non functional': 0}
test_data['status_numeric'] = test_data['status_group'].map(status_mapping)

gpt_cols = [col for col in test_data.columns if col.startswith('gpt_')]
for col in gpt_cols:
    corr = test_data[[col, 'status_numeric']].corr().iloc[0, 1]
    print(f"{col}: {corr:.3f}")