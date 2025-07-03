import os
import asyncio
import sys
import time

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import our module
from gpt_feature_engineering import train, generate_all_features

print("Running GPT feature generation on FULL DATASET")
print(f"Dataset size: {len(train):,} rows")
print("Estimated cost: $4.11")
print("Estimated time: 2-4 hours (depending on rate limits)")
print("=" * 60)

# Confirm before running
response = input("Continue with full dataset generation? (y/N): ")
if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

print("\nStarting GPT feature generation...")
print("Features will be saved progressively for resumability")
print("You can stop and restart this script safely")
print("-" * 60)

start_time = time.time()

try:
    # Run async function
    result = asyncio.run(generate_all_features(train, save_progress=True))

    # Extract only GPT columns and id
    gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
    output_df = result[['id'] + gpt_cols]

    # Save final result
    output_df.to_csv('gpt_features_full.csv', index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("FULL DATASET COMPLETED!")
    print(f"Results saved to 'gpt_features_full.csv'")
    print(f"Shape: {output_df.shape}")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Final columns: {gpt_cols}")

    print("\nFeature statistics:")
    for col in gpt_cols:
        print(f"{col}: min={output_df[col].min()}, max={output_df[col].max()}, mean={output_df[col].mean():.2f}")

except KeyboardInterrupt:
    print("\n\nProcess interrupted by user.")
    print("Progress has been saved. You can resume by running this script again.")
except Exception as e:
    print(f"\nError occurred: {e}")
    print("Progress has been saved. You can resume by running this script again.")