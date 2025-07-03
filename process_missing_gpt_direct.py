#!/usr/bin/env python3
"""
Process Missing GPT Data - Direct Approach
==========================================
- Bypass the batch completion logic
- Process missing IDs directly with fresh batch counting
"""

import os
import asyncio
import sys
import time
import pandas as pd
import sqlite3
import aiohttp

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import our functions
from gpt_feature_engineering_json import (
    create_vertical_batch_prompt, 
    get_gpt_json_response, 
    save_batch_to_sqlite,
    init_sqlite_db,
    RATE_LIMIT_CONFIG
)

print("🔄 PROCESS MISSING GPT DATA - DIRECT APPROACH")
print("=" * 60)

# Load original dataset
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Check what's missing from SQLite
db_path = 'gpt_features_progress.db'
if os.path.exists(db_path):
    print(f"🔍 Checking existing SQLite database...")
    conn = sqlite3.connect(db_path)
    existing_ids_df = pd.read_sql_query('SELECT DISTINCT id FROM gpt_features', conn)
    conn.close()
    
    existing_ids = set(existing_ids_df['id'])
    all_ids = set(train['id'])
    missing_ids = all_ids - existing_ids
    
    print(f"📊 Total IDs: {len(all_ids):,}")
    print(f"📊 Existing in SQLite: {len(existing_ids):,}")
    print(f"📊 Missing from SQLite: {len(missing_ids):,}")
    
    if len(missing_ids) == 0:
        print(f"🎉 All IDs already processed!")
        sys.exit(0)
    
    # Filter to missing data
    missing_train = train[train['id'].isin(missing_ids)].copy().reset_index(drop=True)
    print(f"🎯 Processing {len(missing_train):,} missing rows...")
    
else:
    print(f"📂 No existing SQLite database found")
    missing_train = train
    missing_ids = set(train['id'])

# Estimate work
batch_size = 5
num_batches = (len(missing_train) + batch_size - 1) // batch_size
estimated_cost = num_batches * 0.0001
estimated_time = num_batches / 100  # Conservative estimate

print(f"\n💼 PROCESSING PLAN:")
print("-" * 40)
print(f"📦 Batches needed: {num_batches:,}")
print(f"💰 Estimated cost: ${estimated_cost:.2f}")
print(f"⏰ Estimated time: {estimated_time:.1f} minutes")

# Confirm
response = input(f"\n🚀 Process {len(missing_train):,} missing rows? (y/N): ").strip().lower()
if response not in ['y', 'yes']:
    print("❌ Operation cancelled.")
    sys.exit(0)

async def process_missing_data():
    """Process missing data with fresh batch counting"""
    
    # Initialize SQLite (will use existing if present)
    print(f"\n🚀 Starting direct processing at {time.strftime('%H:%M:%S')}")
    print(f"🔧 DEBUG: About to initialize SQLite database...")
    init_sqlite_db(db_path)
    print(f"✅ DEBUG: SQLite database initialized")
    
    print(f"🔧 DEBUG: Creating semaphore with max_concurrent = {RATE_LIMIT_CONFIG['max_concurrent']}")
    semaphore = asyncio.Semaphore(RATE_LIMIT_CONFIG['max_concurrent'])
    start_time = time.time()
    
    print(f"🎯 Creating {num_batches:,} tasks for missing data...")
    print(f"🔧 DEBUG: Missing train data shape: {missing_train.shape}")
    print(f"🔧 DEBUG: Batch size: {batch_size}")
    print(f"🔧 DEBUG: Sample missing IDs: {missing_train['id'].head(5).tolist()}")
    
    async with aiohttp.ClientSession() as session:
        print(f"🔧 DEBUG: Created aiohttp session")
        tasks = []
        
        # Create tasks for all missing data batches
        print(f"🔧 DEBUG: Starting task creation loop...")
        for i in range(0, len(missing_train), batch_size):
            batch_df = missing_train.iloc[i:i+batch_size]
            print(f"🔧 DEBUG: Creating batch {i//batch_size + 1}, rows {i} to {i+len(batch_df)-1}")
            print(f"🔧 DEBUG: Batch IDs: {batch_df['id'].tolist()}")
            
            try:
                prompt = create_vertical_batch_prompt(batch_df)
                print(f"🔧 DEBUG: Created prompt for batch, length: {len(prompt)}")
                
                task = get_gpt_json_response(session, prompt, len(batch_df), semaphore)
                print(f"🔧 DEBUG: Created GPT task for batch")
                
                tasks.append((task, batch_df, i // batch_size + 1))
                print(f"🔧 DEBUG: Added task to list, total tasks: {len(tasks)}")
                
            except Exception as e:
                print(f"❌ DEBUG: Error creating task for batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
            
            # Only create first few tasks for debugging
            if len(tasks) >= 3:
                print(f"🔧 DEBUG: Created {len(tasks)} tasks for testing, breaking...")
                break
        
        print(f"🔄 Processing {len(tasks)} batches...")
        print(f"🔧 DEBUG: About to start processing tasks...")
        
        completed_batches = 0
        successful_saves = 0
        errors = 0
        
        # Process all tasks
        for i, (task, batch_df, batch_num) in enumerate(tasks):
            print(f"🔧 DEBUG: Processing task {i+1}/{len(tasks)}, batch_num: {batch_num}")
            try:
                print(f"🔧 DEBUG: Awaiting GPT response for batch {batch_num}...")
                # Wait for this specific task
                result = await task
                print(f"🔧 DEBUG: Got result for batch {batch_num}, type: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
                
                # Debug the result structure
                if isinstance(result, list) and len(result) > 0:
                    print(f"🔧 DEBUG: First result item: {result[0]}")
                else:
                    print(f"🔧 DEBUG: Unexpected result format: {result}")
                
                print(f"🔧 DEBUG: About to save batch {batch_num} to SQLite...")
                # Save to SQLite immediately
                save_batch_to_sqlite(result, batch_num, db_path)
                print(f"🔧 DEBUG: Successfully saved batch {batch_num} to SQLite")
                
                successful_saves += 1
                completed_batches += 1
                
                print(f"✅ Completed batch {batch_num} ({completed_batches}/{len(tasks)})")
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                print(f"🔧 DEBUG: Full error details:")
                import traceback
                traceback.print_exc()
                errors += 1
                completed_batches += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n🎉 DIRECT PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"⏰ Time taken: {elapsed/60:.1f} minutes")
    print(f"📊 Batches completed: {completed_batches:,}")
    print(f"💾 Successful saves: {successful_saves:,}")
    print(f"❌ Errors: {errors}")
    
    # Check final status
    conn = sqlite3.connect(db_path)
    final_count = pd.read_sql_query('SELECT COUNT(*) as count FROM gpt_features', conn).iloc[0]['count']
    conn.close()
    
    original_total = len(train)
    completion_pct = (final_count / original_total) * 100
    
    print(f"\n📊 FINAL STATUS:")
    print("-" * 40)
    print(f"Total rows in database: {final_count:,}")
    print(f"Dataset completion: {completion_pct:.1f}%")
    
    if completion_pct >= 99.0:
        print("🎉 Dataset essentially complete!")
    elif completion_pct >= 95.0:
        print("✅ Dataset mostly complete - good for modeling")
    else:
        remaining = original_total - final_count
        print(f"⚠️  Still missing {remaining:,} rows")
    
    # Export complete dataset
    print(f"\n📤 EXPORTING COMPLETE DATASET...")
    conn = sqlite3.connect(db_path)
    complete_df = pd.read_sql_query('''
        SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
               gpt_scheme_pattern, gpt_text_language, gpt_coordination 
        FROM gpt_features ORDER BY id
    ''', conn)
    conn.close()
    
    complete_df.to_csv('gpt_features_complete.csv', index=False)
    print(f"✅ Complete dataset exported: gpt_features_complete.csv ({len(complete_df):,} rows)")
    
    return True

# Run the processing
if __name__ == "__main__":
    try:
        success = asyncio.run(process_missing_data())
        if success:
            print(f"\n🎯 SUCCESS! Missing data processed.")
            print(f"📊 Ready for model training with more complete GPT features.")
        else:
            print(f"\n⚠️  Processing incomplete.")
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Progress saved to SQLite.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()