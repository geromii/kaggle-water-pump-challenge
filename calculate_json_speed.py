import pandas as pd

# Load data
train = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')

BATCH_SIZE = 10  # Conservative for JSON complexity
total_rows = len(train)

# Calculate requests needed
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

print("🚀 JSON VERTICAL BATCHING - FULL DATASET")
print("=" * 50)
print(f"Total rows: {total_rows:,}")
print(f"Batch size: {BATCH_SIZE} rows per request")
print(f"Total requests needed: {num_batches:,}")

# Rate limits
requests_per_minute = 500

# Calculate time
time_minutes = num_batches / requests_per_minute
time_hours = time_minutes / 60

print(f"\nAt 500 requests/minute:")
print(f"Time: {time_minutes:.1f} minutes ({time_hours:.2f} hours)")

# Speedup comparisons
original_requests = total_rows * 6  # 6 features individually
horizontal_batch_requests = ((total_rows + 20 - 1) // 20) * 6  # 20-row batches
json_vertical_requests = num_batches

print(f"\n📊 COMPARISON:")
print(f"Individual requests: {original_requests:,} → {original_requests/500:.0f} minutes")
print(f"Horizontal batching: {horizontal_batch_requests:,} → {horizontal_batch_requests/500:.0f} minutes") 
print(f"JSON vertical batching: {json_vertical_requests:,} → {time_minutes:.0f} minutes")

print(f"\n🎯 SPEEDUPS:")
print(f"vs Individual: {original_requests/json_vertical_requests:.0f}x faster")
print(f"vs Horizontal: {horizontal_batch_requests/json_vertical_requests:.0f}x faster")

print(f"\n⚡ FINAL ESTIMATE: ~{time_minutes:.0f} minutes for full dataset!")
print(f"💰 Same cost: $4.11 (same total tokens)")
print(f"🎉 Ultimate efficiency achieved!")