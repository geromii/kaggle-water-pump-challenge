import pandas as pd

# Load data
train = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')

BATCH_SIZE = 20
total_rows = len(train)
features_count = 6

# Calculate requests needed
batches_per_feature = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
total_requests = batches_per_feature * features_count

print("BATCH PROCESSING SPEED CALCULATION")
print("=" * 50)
print(f"Total rows: {total_rows:,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches per feature: {batches_per_feature:,}")
print(f"Total requests needed: {total_requests:,}")

# Rate limits
requests_per_minute = 500

# Calculate time
time_minutes = total_requests / requests_per_minute
time_hours = time_minutes / 60

print(f"\nAt 500 requests/minute:")
print(f"Time: {time_minutes:.1f} minutes ({time_hours:.1f} hours)")

# With some overhead for processing
realistic_time = time_minutes * 1.1
print(f"With 10% overhead: {realistic_time:.1f} minutes ({realistic_time/60:.1f} hours)")

# Speedup vs original approach
original_requests = total_rows * features_count
speedup = original_requests / total_requests
print(f"\nSpeedup vs individual requests:")
print(f"Original: {original_requests:,} requests → {original_requests/500:.0f} minutes")
print(f"Batched: {total_requests:,} requests → {time_minutes:.0f} minutes")
print(f"Speedup: {speedup:.0f}x faster!")

print(f"\n🚀 FINAL ESTIMATE: ~{realistic_time/60:.1f} hours for full dataset")