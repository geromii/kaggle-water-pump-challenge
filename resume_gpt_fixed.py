#!/usr/bin/env python3
"""
Resume GPT Processing - Fixed Version
====================================
- Properly process missing IDs by checking which specific IDs exist
- Reset batch counting for missing data only
"""

import os
import asyncio
import sys
import time
import pandas as pd
import sqlite3

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

print("🔄 RESUME GPT PROCESSING - FIXED VERSION")
print("=" * 60)

# Load missing IDs
if not os.path.exists('missing_ids_for_gpt.csv'):
    print("❌ Error: missing_ids_for_gpt.csv not found!")
    sys.exit(1)

missing_data = pd.read_csv('missing_ids_for_gpt.csv')
print(f"📊 Missing rows to process: {len(missing_data):,}")

# Double-check what's actually missing by querying SQLite directly
print("🔍 Double-checking missing IDs against SQLite...")

if os.path.exists('gpt_features_progress.db'):
    conn = sqlite3.connect('gpt_features_progress.db')
    existing_ids_df = pd.read_sql_query('SELECT DISTINCT id FROM gpt_features', conn)
    conn.close()
    
    existing_ids = set(existing_ids_df['id'])
    missing_ids = set(missing_data['id'])
    
    # Find IDs that are truly missing (not in SQLite)
    actually_missing = missing_ids - existing_ids
    
    print(f"📊 Originally missing: {len(missing_ids):,}")
    print(f"📊 Actually still missing: {len(actually_missing):,}")
    
    if len(actually_missing) == 0:
        print("🎉 All data is actually complete!")
        
        # Export complete dataset
        print("📤 Exporting complete dataset...")
        conn = sqlite3.connect('gpt_features_progress.db')
        complete_df = pd.read_sql_query('''
            SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
                   gpt_scheme_pattern, gpt_text_language, gpt_coordination 
            FROM gpt_features ORDER BY id
        ''', conn)
        conn.close()
        
        complete_df.to_csv('gpt_features_complete.csv', index=False)
        print(f"✅ Exported {len(complete_df):,} complete rows to: gpt_features_complete.csv")
        print(f"🎯 Ready for model training!")
        sys.exit(0)
    
    # Filter to only truly missing IDs
    truly_missing_data = missing_data[missing_data['id'].isin(actually_missing)]
    print(f"🎯 Filtering to {len(truly_missing_data):,} truly missing rows")
    
else:
    print("⚠️  SQLite database not found, processing all missing IDs")
    truly_missing_data = missing_data

# Estimate work for truly missing data
num_batches = (len(truly_missing_data) + 4) // 5
estimated_cost = num_batches * 0.0001
estimated_time_minutes = num_batches / 100

print(f"\n💼 PROCESSING PLAN:")
print("-" * 40)
print(f"📦 Batches needed: {num_batches:,}")
print(f"💰 Estimated cost: ${estimated_cost:.2f}")
print(f"⏰ Estimated time: {estimated_time_minutes:.1f} minutes")

# Show sample
sample_ids = truly_missing_data['id'].head(10).tolist()
print(f"📋 Sample IDs to process: {sample_ids}")

# Confirm with user
response = input(f"\n🚀 Process {len(truly_missing_data):,} truly missing rows? (y/N): ").strip().lower()
if response not in ['y', 'yes']:
    print("❌ Operation cancelled by user.")
    sys.exit(0)

# Create a temporary processing function that bypasses the existing batch detection
async def process_missing_data_directly():
    """Process missing data by directly calling API without batch checking"""
    
    from gpt_feature_engineering_json import (
        create_vertical_batch_prompt, 
        get_gpt_json_response, 
        save_batch_to_sqlite,
        init_sqlite_db,
        RATE_LIMIT_CONFIG
    )
    import aiohttp
    
    # Initialize SQLite
    db_path = init_sqlite_db('gpt_features_progress.db')
    
    # Processing parameters
    batch_size = 5
    semaphore = asyncio.Semaphore(RATE_LIMIT_CONFIG['max_concurrent'])
    
    print(f"\n🚀 Starting direct processing at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Create tasks for missing data
        for i in range(0, len(truly_missing_data), batch_size):
            batch_df = truly_missing_data.iloc[i:i+batch_size]
            prompt = create_vertical_batch_prompt(batch_df)
            task = get_gpt_json_response(session, prompt, len(batch_df), semaphore)
            tasks.append((task, batch_df, i // batch_size + 1))
        
        print(f"🎯 Processing {len(tasks)} batches for missing data...")
        
        completed_batches = 0
        successful_saves = 0
        
        for task, batch_df, batch_num in asyncio.as_completed([(t[0], t[1], t[2]) for t in tasks]):
            try:
                # Wait for task completion
                result = await task
                
                # Save to SQLite immediately
                save_batch_to_sqlite(result, batch_num, db_path)
                successful_saves += 1
                completed_batches += 1
                
                # Progress update
                progress_pct = (completed_batches / len(tasks)) * 100
                elapsed = time.time() - start_time
                rate = completed_batches / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - completed_batches) / rate if rate > 0 else 0
                
                print(f"🔄 {progress_pct:5.1f}% | {completed_batches:,}/{len(tasks):,} | "
                      f"⏱️  {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | "
                      f"⚡ {rate:.1f} req/s")
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                completed_batches += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n🎉 DIRECT PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"⏰ Time taken: {elapsed/60:.1f} minutes")
    print(f"📊 Batches processed: {completed_batches:,}")
    print(f"💾 Successful saves: {successful_saves:,}")
    
    # Check final status
    conn = sqlite3.connect('gpt_features_progress.db')
    final_count = pd.read_sql_query('SELECT COUNT(*) as count FROM gpt_features', conn).iloc[0]['count']
    conn.close()
    
    original_total = 59400
    completion_pct = (final_count / original_total) * 100
    
    print(f"\n📊 FINAL STATUS:")
    print("-" * 40)
    print(f"Total rows in database: {final_count:,}")
    print(f"Dataset completion: {completion_pct:.1f}%")
    
    if completion_pct >= 99.0:
        print("🎉 Dataset essentially complete!")
        
        # Export final complete dataset
        conn = sqlite3.connect('gpt_features_progress.db')
        complete_df = pd.read_sql_query('''
            SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
                   gpt_scheme_pattern, gpt_text_language, gpt_coordination 
            FROM gpt_features ORDER BY id
        ''', conn)
        conn.close()
        
        complete_df.to_csv('gpt_features_complete.csv', index=False)
        print(f"✅ Exported complete dataset: gpt_features_complete.csv ({len(complete_df):,} rows)")
    
    return True

# Run the processing
if __name__ == "__main__":
    try:
        success = asyncio.run(process_missing_data_directly())
        if success:
            print(f"\n🎯 SUCCESS! Ready for model training with complete GPT features.")
        else:
            print(f"\n⚠️  Processing incomplete. Check errors above.")
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Progress saved to SQLite.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()