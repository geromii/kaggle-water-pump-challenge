#!/usr/bin/env python3
"""
Final GPT Feature Generation Script
==================================
- GPT-4.1-mini model
- JSON vertical batching (60x speedup)
- Comprehensive progress tracking
- Error handling and recovery
- Cost: ~$0.95, Time: ~12 minutes
"""

import os
import asyncio
import sys
import time
import signal
import pandas as pd

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    print("Please set it with: export OPENAI_API_KEY='your_key_here'")
    sys.exit(1)

# Import our optimized module
try:
    from gpt_feature_engineering_json import train, process_all_features_json
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure gpt_feature_engineering_json.py is in the current directory")
    sys.exit(1)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n⚠️  Received interrupt signal ({signum})")
    print("🛑 Stopping gracefully... Progress has been saved.")
    print("📂 You can resume by running this script again.")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def print_header():
    """Print fancy header"""
    print("🚀" + "=" * 58 + "🚀")
    print("🎯  GPT-4.1-MINI FEATURE GENERATION - FULL DATASET  🎯")
    print("🚀" + "=" * 58 + "🚀")

def print_specs():
    """Print run specifications"""
    print(f"\n📋 RUN SPECIFICATIONS:")
    print("-" * 40)
    print(f"🗃️  Dataset: {len(train):,} water pump records")
    print(f"🤖 Model: GPT-4.1-nano (high quality)")
    print(f"📦 Method: JSON vertical batching")
    print(f"⚡ Speedup: 30x faster than individual requests")
    print(f"🎯 Features: 6 custom features per row")
    print(f"📊 Batches: ~11,880 requests (5 rows each)")
    print(f"⏰ Time: ~24 minutes estimated")
    print(f"💰 Cost: ~$0.95 estimated")
    print(f"🔄 Progress: Real-time updates every 10%")
    print(f"💾 Recovery: Auto-save progress, resumable")

def get_user_confirmation():
    """Get user confirmation with details"""
    print(f"\n⚠️  COST CONFIRMATION:")
    print("-" * 40)
    print(f"This will make ~11,880 API calls to OpenAI")
    print(f"Estimated cost: $0.95 (GPT-4.1-mini)")
    print(f"This is 94% cheaper than individual requests!")
    print(f"\n🔧 SAFETY FEATURES:")
    print("✅ Progress auto-saved every 20%")
    print("✅ Can resume if interrupted")
    print("✅ Error handling with fallbacks")
    print("✅ Rate limiting with exponential backoff")
    
    while True:
        response = input(f"\n🚀 Proceed with full dataset generation? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

async def main():
    """Main execution function"""
    print_header()
    print_specs()
    
    # Check if results already exist
    if os.path.exists('gpt_features_full.csv'):
        print(f"\n📂 Found existing results: gpt_features_full.csv")
        response = input("Results already exist. Regenerate? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("✅ Using existing results.")
            return
    
    if not get_user_confirmation():
        print("❌ Operation cancelled by user.")
        return
    
    print(f"\n🎬 STARTING GENERATION...")
    print("=" * 60)
    print(f"💡 Tip: You can press Ctrl+C to stop safely - progress will be saved!")
    print(f"📂 Watch for progress files: gpt_features_progress.csv")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Check which IDs are missing from SQLite database
        import sqlite3
        
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
                print(f"🎉 All IDs already processed! Exporting final results...")
                conn = sqlite3.connect(db_path)
                complete_df = pd.read_sql_query('''
                    SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
                           gpt_scheme_pattern, gpt_text_language, gpt_coordination 
                    FROM gpt_features ORDER BY id
                ''', conn)
                conn.close()
                
                complete_df.to_csv('gpt_features_full.csv', index=False)
                print(f"✅ Exported {len(complete_df):,} complete rows to: gpt_features_full.csv")
                return
            
            # Filter train data to only missing IDs
            missing_train = train[train['id'].isin(missing_ids)].copy()
            print(f"🎯 Processing only {len(missing_train):,} missing rows...")
            
        else:
            print(f"📂 No existing SQLite database found, processing all data...")
            missing_train = train
        
        # Run the main processing function on missing data only
        result = await process_all_features_json(missing_train, batch_size=5, save_progress=True)
        
        # Validate results
        gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
        
        if len(gpt_cols) != 6:
            print(f"⚠️  Warning: Expected 6 GPT features, got {len(gpt_cols)}")
        
        # Show statistics for newly processed data
        total_time = time.time() - start_time
        print(f"\n🎊 SUCCESS! MISSING DATA PROCESSED!")
        print("=" * 60)
        print(f"📊 Newly processed: {result.shape[0]:,} rows")
        print(f"⏰ Total time: {total_time/60:.1f} minutes")
        print(f"🎯 Generated features: {gpt_cols}")
        
        if len(gpt_cols) > 0 and result.shape[0] > 0:
            print(f"\n📈 NEW DATA STATISTICS:")
            print("-" * 40)
            for col in gpt_cols:
                if col in result.columns:
                    print(f"{col}: min={result[col].min():.1f}, max={result[col].max():.1f}, mean={result[col].mean():.2f}")
        
        # Export complete dataset from SQLite
        print(f"\n📤 EXPORTING COMPLETE DATASET...")
        print("-" * 40)
        conn = sqlite3.connect(db_path)
        complete_df = pd.read_sql_query('''
            SELECT id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
                   gpt_scheme_pattern, gpt_text_language, gpt_coordination 
            FROM gpt_features ORDER BY id
        ''', conn)
        conn.close()
        
        complete_df.to_csv('gpt_features_full.csv', index=False)
        print(f"✅ Complete dataset exported: gpt_features_full.csv ({len(complete_df):,} rows)")
        
        # Check final completion status
        original_total = len(train['id'].unique())
        completion_pct = (len(complete_df) / original_total) * 100
        print(f"📊 Dataset completion: {completion_pct:.1f}%")
        
        print(f"\n🎯 NEXT STEPS:")
        print("-" * 40)
        print("1. 📊 Complete features saved to: gpt_features_full.csv")
        print("2. 🤖 Run combined GPT + geospatial model to test impact")
        print("3. 📈 Compare accuracy with original model")
        print("4. 🚀 Submit improved predictions!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user.")
        print("📂 Progress has been saved. Run script again to resume.")
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        print("📂 Any progress has been saved. Check for partial results.")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\n⏰ Session duration: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required for async features")
        sys.exit(1)
    
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)