import os
import asyncio
import sys

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import our module
from gpt_feature_engineering import test_sample, generate_all_features

print("Running GPT feature generation on 100 test samples...")
print("This should cost approximately $0.0069")
print("-" * 50)

# Run async function
result = asyncio.run(generate_all_features(test_sample))

# Extract only GPT columns and id
gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
output_df = result[['id'] + gpt_cols]

# Save to CSV
output_df.to_csv('gpt_features_test.csv', index=False)

print("\nTest completed! Results saved to 'gpt_features_test.csv'")
print(f"Shape: {output_df.shape}")
print("\nFirst 5 rows:")
print(output_df.head())

print("\nFeature statistics:")
for col in gpt_cols:
    print(f"{col}: min={output_df[col].min()}, max={output_df[col].max()}, mean={output_df[col].mean():.2f}")