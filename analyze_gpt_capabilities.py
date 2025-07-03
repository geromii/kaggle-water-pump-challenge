import pandas as pd

# Load data to analyze what GPT can actually help with
train = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')

print("Analyzing text fields that GPT could actually help with...\n")

# 1. Funder field - lots of messy text
print("1. FUNDER field analysis:")
print(f"Unique values: {train['funder'].nunique()}")
print("\nSample values showing inconsistencies:")
funder_sample = train['funder'].value_counts().head(20)
print(funder_sample)
print("\nExamples of potentially similar entries:")
funders = train['funder'].dropna().unique()
gov_variants = [f for f in funders if 'gov' in f.lower() or 'govt' in f.lower()][:10]
print("Government variants:", gov_variants)

# 2. Installer field
print("\n\n2. INSTALLER field analysis:")
print(f"Unique values: {train['installer'].nunique()}")
installer_sample = train['installer'].value_counts().head(20)
print(installer_sample)

# 3. Scheme name
print("\n\n3. SCHEME_NAME field analysis:")
print(f"Unique values: {train['scheme_name'].nunique()}")
scheme_sample = train['scheme_name'].dropna().sample(20)
print("\nRandom samples:")
for s in scheme_sample:
    print(f"  - {s}")

# 4. Ward names
print("\n\n4. WARD field analysis:")
print(f"Unique values: {train['ward'].nunique()}")
ward_sample = train['ward'].value_counts().head(10)
print(ward_sample)

# 5. Subvillage names  
print("\n\n5. SUBVILLAGE field analysis:")
print(f"Unique values: {train['subvillage'].nunique()}")
print("\nSamples showing patterns:")
subvillage_sample = train['subvillage'].dropna().sample(20)
for s in subvillage_sample:
    print(f"  - {s}")

print("\n\nWHAT GPT CAN ACTUALLY DO:")
print("1. Standardize organization names (Gov/Govt/Government)")
print("2. Identify organization type from name patterns")
print("3. Detect language (Swahili vs English names)")
print("4. Extract location type from subvillage names (school/church/center)")
print("5. Parse abbreviations and expand them")
print("6. Identify data quality issues (test entries, placeholders)")
print("7. Group similar text variants together")