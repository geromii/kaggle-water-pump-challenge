import asyncio
from gpt_feature_engineering_json import process_all_features_json, train

# Test with a small sample to verify the fix works
test_sample = train.head(20)

print("Testing JSON fix with 20 rows...")
result = asyncio.run(process_all_features_json(test_sample, batch_size=5, save_progress=False))
print(f"Test successful! Shape: {result.shape}")

gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
print(f"Generated features: {gpt_cols}")

for col in gpt_cols:
    print(f"{col}: min={result[col].min():.1f}, max={result[col].max():.1f}, mean={result[col].mean():.2f}")