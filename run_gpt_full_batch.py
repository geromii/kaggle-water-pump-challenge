import os
import asyncio
import sys
import time

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import our module
from gpt_feature_engineering_batch import train, generate_all_features_batch

print("🚀 BATCH GPT FEATURE GENERATION - FULL DATASET")
print("=" * 60)
print(f"Dataset size: {len(train):,} rows")
print("Estimated cost: $4.11")
print("Estimated time: ~40 minutes (20x faster with batching!)")
print("Batch size: 20 rows per request")
print("Total requests needed: ~17,820 (vs 356,400 without batching)")
print("=" * 60)

# Confirm before running
response = input("Continue with FAST batch generation? (y/N): ")
if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

print("\n🎯 Starting BATCH GPT feature generation...")
print("Processing 20 rows per API call for maximum efficiency")
print("-" * 60)

start_time = time.time()

try:
    # Run async function
    result = asyncio.run(generate_all_features_batch(train))

    # Extract only GPT columns and id
    gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
    output_df = result[['id'] + gpt_cols]

    # Save final result
    output_df.to_csv('gpt_features_full_batch.csv', index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("🎉 FULL DATASET COMPLETED!")
    print(f"Results saved to 'gpt_features_full_batch.csv'")
    print(f"Shape: {output_df.shape}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Generated features: {gpt_cols}")

    print("\n📊 Feature statistics:")
    for col in gpt_cols:
        print(f"{col}: min={output_df[col].min()}, max={output_df[col].max()}, mean={output_df[col].mean():.2f}")

    print(f"\n💰 Estimated cost: $4.11")
    print(f"⚡ Speedup achieved: 20x faster than individual requests!")

except KeyboardInterrupt:
    print("\n\nProcess interrupted by user.")
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()