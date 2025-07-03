#!/usr/bin/env python3
"""
Resume GPT Processing for Missing IDs
====================================
- Process only the missing 19,598 IDs 
- Use existing SQLite infrastructure
- Resume exactly where we left off
"""

import os
import asyncio
import sys
import time
import pandas as pd

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import our module
from gpt_feature_engineering_json import process_all_features_json

print("🔄 RESUME GPT PROCESSING FOR MISSING IDS")
print("=" * 60)

# Load missing IDs
if not os.path.exists('missing_ids_for_gpt.csv'):
    print("❌ Error: missing_ids_for_gpt.csv not found!")
    print("Run check_gpt_data_completeness.py first")
    sys.exit(1)

missing_data = pd.read_csv('missing_ids_for_gpt.csv')
print(f"📊 Missing rows to process: {len(missing_data):,}")

# Estimate work required
num_batches = (len(missing_data) + 4) // 5  # 5 rows per batch
estimated_cost = num_batches * 0.0001
estimated_time_minutes = num_batches / 100  # Conservative estimate

print(f"📦 Batches needed: {num_batches:,}")
print(f"💰 Estimated cost: ${estimated_cost:.2f}")
print(f"⏰ Estimated time: {estimated_time_minutes:.1f} minutes")
print("=" * 60)

# Show sample of missing data
print(f"📋 Sample missing IDs:")
sample_ids = missing_data['id'].head(10).tolist()
print(f"First 10: {sample_ids}")

# Confirm with user
response = input(f"\n🚀 Process {len(missing_data):,} missing rows? (y/N): ").strip().lower()
if response not in ['y', 'yes']:
    print("❌ Operation cancelled by user.")
    sys.exit(0)

async def main():
    print(f"\n🚀 Starting resumed processing at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Run processing on missing data only
        # The SQLite system will automatically append to existing database
        result = await process_all_features_json(
            missing_data, 
            batch_size=5, 
            save_progress=True  # Will use existing SQLite database
        )
        
        elapsed = time.time() - start_time
        
        # Validate results
        gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
        
        print(f"\n🎉 RESUME PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏰ Time taken: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"📊 Processed: {len(result)} missing rows")
        print(f"🎯 Features generated: {len(gpt_cols)}")
        print(f"✅ Features: {gpt_cols}")
        
        # Check final database status
        print(f"\n📊 FINAL DATABASE STATUS:")
        print("-" * 40)
        
        # Quick check of SQLite database
        import sqlite3
        conn = sqlite3.connect('gpt_features_progress.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM gpt_features')
        total_rows = cursor.fetchone()[0]
        conn.close()
        
        print(f"Total rows in database: {total_rows:,}")
        
        # Calculate completion percentage
        original_total = 59400  # Total original dataset
        completion_pct = (total_rows / original_total) * 100
        print(f"Dataset completion: {completion_pct:.1f}%")
        
        if completion_pct >= 99.0:
            print("🎉 Dataset essentially complete!")
        elif completion_pct >= 95.0:
            print("✅ Dataset mostly complete - ready for model training")
        else:
            remaining = original_total - total_rows
            print(f"⚠️  Still missing {remaining:,} rows ({100-completion_pct:.1f}%)")
        
        # Show final statistics
        print(f"\n📈 FINAL FEATURE STATISTICS:")
        print("-" * 40)
        for col in gpt_cols:
            if col in result.columns:
                min_val = result[col].min()
                max_val = result[col].max()
                mean_val = result[col].mean()
                print(f"{col}: {min_val:.1f}-{max_val:.1f} (avg: {mean_val:.2f})")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user.")
        print("📂 Progress has been saved. Run script again to continue.")
        return False
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("📂 Any partial progress has been saved.")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print(f"\n🎯 NEXT STEPS:")
        print("-" * 40)
        print("1. ✅ Check final database completeness")
        print("2. 🤖 Run model with combined GPT + selective geospatial features")
        print("3. 📊 Compare performance against selective geospatial baseline")
        print("4. 🚀 Create final submission if improvement is significant")
    else:
        print(f"\n🔧 TROUBLESHOOTING:")
        print("-" * 40)
        print("1. 🔍 Check API key and rate limits")
        print("2. 📂 Review any error messages above")
        print("3. 🔄 Run this script again to resume from where it stopped")
    
    print(f"\n👋 Resume processing complete!")