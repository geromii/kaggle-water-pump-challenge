#!/usr/bin/env python3
"""
Test GPT Feature Generation on 1% of Dataset
===========================================
- 594 rows (1% of 59,400)
- Should take ~2-3 minutes
- Cost: ~$0.01
"""

import os
import asyncio
import sys
import time

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import our module
from gpt_feature_engineering_json import train, process_all_features_json

# Take 1% sample
sample_size = int(len(train) * 0.01)  # 594 rows
test_dataset = train.sample(sample_size, random_state=42)

print("🧪 TESTING WITH 1% OF DATASET")
print("=" * 50)
print(f"📊 Original dataset: {len(train):,} rows")
print(f"🎯 Test sample: {len(test_dataset):,} rows (1%)")
print(f"📦 Batches: ~{(len(test_dataset) + 4) // 5} requests (5 rows each)")
print(f"⏰ Expected time: 2-3 minutes")
print(f"💰 Expected cost: ~$0.01")
print("=" * 50)

# Auto-proceed for testing
print("🚀 Auto-proceeding with test...")

async def main():
    print(f"\n🚀 Starting 1% test at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Run with smaller batch size and lower concurrency for testing
        result = await process_all_features_json(
            test_dataset, 
            batch_size=5, 
            save_progress=False  # No need to save progress for small test
        )
        
        elapsed = time.time() - start_time
        
        # Validate results
        gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
        
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"⏰ Time taken: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"📊 Processed: {len(result)} rows")
        print(f"🎯 Features generated: {len(gpt_cols)}")
        print(f"✅ Features: {gpt_cols}")
        
        # Show statistics
        print(f"\n📈 FEATURE STATISTICS:")
        print("-" * 30)
        for col in gpt_cols:
            if col in result.columns:
                min_val = result[col].min()
                max_val = result[col].max()
                mean_val = result[col].mean()
                print(f"{col}: {min_val:.1f}-{max_val:.1f} (avg: {mean_val:.2f})")
        
        # Save small test results
        output_file = 'gpt_features_1percent_test.csv'
        result[['id'] + gpt_cols].to_csv(output_file, index=False)
        print(f"\n💾 Test results saved to: {output_file}")
        
        # Extrapolate to full dataset
        batches_per_minute = ((len(test_dataset) + 4) // 5) / (elapsed / 60)
        full_batches = (len(train) + 4) // 5
        estimated_full_time = full_batches / batches_per_minute
        
        print(f"\n🔮 FULL DATASET EXTRAPOLATION:")
        print("-" * 30)
        print(f"📊 Full dataset batches: {full_batches:,}")
        print(f"⚡ Observed rate: {batches_per_minute:.1f} batches/min")
        print(f"⏰ Estimated full time: {estimated_full_time:.1f} minutes ({estimated_full_time/60:.1f} hours)")
        print(f"💰 Estimated full cost: ${(full_batches * 0.0001):.2f}")
        
        if estimated_full_time > 60:
            print(f"⚠️  Full dataset would take {estimated_full_time/60:.1f} hours")
            print("💡 Consider reducing concurrency or using smaller batches")
        else:
            print(f"✅ Full dataset timing looks reasonable!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print(f"\n🎯 RECOMMENDATION:")
        print("✅ Test successful - safe to proceed with full dataset")
        print("📝 Adjust timing expectations based on observed rate")
    else:
        print(f"\n🔧 RECOMMENDATION:")
        print("❌ Fix issues before running full dataset")
    
    print(f"\n👋 Test complete!")