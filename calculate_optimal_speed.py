import pandas as pd

# Load data
train = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')

# Rate limits you mentioned
requests_per_minute = 500
tokens_per_minute = 200_000

# Our requirements
total_rows = len(train)
features_count = 6
total_requests = total_rows * features_count

# Token usage from our earlier estimate
avg_tokens_per_request = 145  # input tokens
output_tokens_per_request = 2  # output tokens  
total_tokens_per_request = avg_tokens_per_request + output_tokens_per_request

print("SPEED CALCULATION")
print("=" * 50)
print(f"Total rows: {total_rows:,}")
print(f"Features to generate: {features_count}")
print(f"Total API requests needed: {total_requests:,}")
print(f"Tokens per request: {total_tokens_per_request}")
print(f"Total tokens needed: {total_requests * total_tokens_per_request:,}")

print(f"\nRATE LIMITS:")
print(f"Requests per minute: {requests_per_minute:,}")
print(f"Tokens per minute: {tokens_per_minute:,}")

# Calculate time based on request limit
time_by_requests = total_requests / requests_per_minute  # minutes
print(f"\nTime limited by REQUESTS: {time_by_requests:.1f} minutes ({time_by_requests/60:.1f} hours)")

# Calculate time based on token limit  
total_tokens = total_requests * total_tokens_per_request
time_by_tokens = total_tokens / tokens_per_minute  # minutes
print(f"Time limited by TOKENS: {time_by_tokens:.1f} minutes ({time_by_tokens/60:.1f} hours)")

# The bottleneck is whichever is larger
bottleneck_time = max(time_by_requests, time_by_tokens)
print(f"\nBOTTLENECK: {bottleneck_time:.1f} minutes ({bottleneck_time/60:.1f} hours)")

# Optimal settings
if time_by_requests > time_by_tokens:
    print("\nRequest-limited. Can increase concurrency up to the request limit.")
    optimal_concurrent = min(500, 100)  # Cap at reasonable number
else:
    print("\nToken-limited. Current concurrency is fine.")
    optimal_concurrent = 50

print(f"Recommended concurrent requests: {optimal_concurrent}")
print(f"With {optimal_concurrent} concurrent: ~{bottleneck_time:.0f} minutes total")

# Factor in some overhead for processing, retries, etc.
realistic_time = bottleneck_time * 1.2  # 20% overhead
print(f"Realistic time with overhead: {realistic_time:.0f} minutes ({realistic_time/60:.1f} hours)")