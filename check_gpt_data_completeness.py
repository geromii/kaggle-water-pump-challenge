#!/usr/bin/env python3
"""
Check GPT Data Completeness
===========================
- Analyze SQLite database for missing/incomplete rows
- Identify rows with errors or missing values
- Create list of IDs that need to be reprocessed
"""

import pandas as pd
import numpy as np
import sqlite3
import os

print("🔍 CHECKING GPT DATA COMPLETENESS")
print("=" * 60)

# Load original dataset to compare against
print("📊 Loading original dataset...")
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
print(f"✅ Original dataset: {len(train_features):,} rows")

# Check what files we have
print(f"\n📁 CHECKING AVAILABLE FILES:")
print("-" * 40)

files_to_check = [
    'gpt_features_progress.db',
    'gpt_features_progress.csv', 
    'gpt_features_full.csv'
]

for file in files_to_check:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"✅ {file:<30} {size_mb:.1f} MB")
    else:
        print(f"❌ {file:<30} Not found")

# Load data from SQLite
print(f"\n🗄️  ANALYZING SQLITE DATABASE:")
print("-" * 40)

if os.path.exists('gpt_features_progress.db'):
    conn = sqlite3.connect('gpt_features_progress.db')
    
    # Check table structure
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {[t[0] for t in tables]}")
    
    # Load all data from SQLite
    sqlite_df = pd.read_sql_query('SELECT * FROM gpt_features ORDER BY id', conn)
    conn.close()
    
    print(f"📊 SQLite rows: {len(sqlite_df):,}")
    print(f"📊 Unique IDs: {sqlite_df['id'].nunique():,}")
    print(f"📊 Date range: {sqlite_df['timestamp'].min():.0f} - {sqlite_df['timestamp'].max():.0f}")
    
    # Check for completeness
    print(f"\n🔍 DATA QUALITY ANALYSIS:")
    print("-" * 40)
    
    gpt_feature_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                       'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
    
    for col in gpt_feature_cols:
        if col in sqlite_df.columns:
            non_null_count = sqlite_df[col].notna().sum()
            null_count = sqlite_df[col].isna().sum()
            print(f"{col:<25} {non_null_count:6,} complete, {null_count:6,} missing")
        else:
            print(f"{col:<25} Column not found!")
    
    # Check for rows with all features complete
    complete_mask = sqlite_df[gpt_feature_cols].notna().all(axis=1)
    complete_count = complete_mask.sum()
    incomplete_count = len(sqlite_df) - complete_count
    
    print(f"\n📈 COMPLETENESS SUMMARY:")
    print("-" * 40)
    print(f"✅ Complete rows:   {complete_count:,} ({complete_count/len(sqlite_df)*100:.1f}%)")
    print(f"❌ Incomplete rows: {incomplete_count:,} ({incomplete_count/len(sqlite_df)*100:.1f}%)")
    
    # Find rows with any missing values
    if incomplete_count > 0:
        incomplete_rows = sqlite_df[~complete_mask]
        print(f"\n⚠️  INCOMPLETE ROWS ANALYSIS:")
        print("-" * 40)
        print(f"Rows with missing values: {len(incomplete_rows):,}")
        
        # Show which features are missing most often
        for col in gpt_feature_cols:
            missing_in_incomplete = incomplete_rows[col].isna().sum()
            print(f"{col:<25} missing in {missing_in_incomplete:5,} incomplete rows")
    
    # Check which original IDs are missing entirely
    original_ids = set(train_features['id'])
    sqlite_ids = set(sqlite_df['id'])
    missing_ids = original_ids - sqlite_ids
    
    print(f"\n🔍 MISSING IDS ANALYSIS:")
    print("-" * 40)
    print(f"Original dataset IDs: {len(original_ids):,}")
    print(f"SQLite database IDs:  {len(sqlite_ids):,}")
    print(f"Missing IDs:          {len(missing_ids):,}")
    
    if len(missing_ids) > 0:
        missing_ids_list = sorted(list(missing_ids))
        print(f"Sample missing IDs: {missing_ids_list[:10]}")
        
        # Save missing IDs for reprocessing
        missing_df = train_features[train_features['id'].isin(missing_ids)]
        missing_df.to_csv('missing_ids_for_gpt.csv', index=False)
        print(f"💾 Saved {len(missing_df)} missing IDs to: missing_ids_for_gpt.csv")
    
    # Create list of IDs that need reprocessing (missing entirely OR incomplete)
    if incomplete_count > 0:
        incomplete_ids = set(incomplete_rows['id'])
        all_ids_to_reprocess = missing_ids.union(incomplete_ids)
        
        print(f"\n🔄 REPROCESSING SUMMARY:")
        print("-" * 40)
        print(f"IDs missing entirely:     {len(missing_ids):,}")
        print(f"IDs with incomplete data: {len(incomplete_ids):,}")
        print(f"Total IDs to reprocess:   {len(all_ids_to_reprocess):,}")
        
        # Save all IDs that need reprocessing
        reprocess_df = train_features[train_features['id'].isin(all_ids_to_reprocess)]
        reprocess_df.to_csv('ids_to_reprocess_gpt.csv', index=False)
        print(f"💾 Saved {len(reprocess_df)} IDs to reprocess: ids_to_reprocess_gpt.csv")
        
        # Calculate progress percentage
        successful_ids = len(original_ids) - len(all_ids_to_reprocess)
        progress_pct = (successful_ids / len(original_ids)) * 100
        print(f"\n📊 OVERALL PROGRESS:")
        print("-" * 40)
        print(f"Successfully processed: {successful_ids:,} / {len(original_ids):,} ({progress_pct:.1f}%)")
        print(f"Still need processing:  {len(all_ids_to_reprocess):,} ({100-progress_pct:.1f}%)")
        
        # Estimate cost and time for remaining work
        remaining_batches = (len(all_ids_to_reprocess) + 4) // 5  # 5 rows per batch
        estimated_cost = remaining_batches * 0.0001  # Rough estimate
        estimated_time_minutes = remaining_batches / 100  # Rough rate estimate
        
        print(f"\n💰 REPROCESSING ESTIMATES:")
        print("-" * 40)
        print(f"Remaining batches: {remaining_batches:,}")
        print(f"Estimated cost:    ${estimated_cost:.2f}")
        print(f"Estimated time:    {estimated_time_minutes:.1f} minutes")
    
    else:
        print(f"\n🎉 ALL ROWS COMPLETE!")
        print("No reprocessing needed.")

else:
    print("❌ SQLite database not found!")

# Also check CSV files if they exist
print(f"\n📄 CHECKING CSV FILES:")
print("-" * 40)

for csv_file in ['gpt_features_progress.csv', 'gpt_features_full.csv']:
    if os.path.exists(csv_file):
        try:
            csv_df = pd.read_csv(csv_file)
            print(f"{csv_file}:")
            print(f"  Rows: {len(csv_df):,}")
            print(f"  Columns: {list(csv_df.columns)}")
            
            # Check completeness
            if len(csv_df) > 0:
                gpt_cols = [col for col in csv_df.columns if col.startswith('gpt_')]
                if gpt_cols:
                    complete_csv = csv_df[gpt_cols].notna().all(axis=1).sum()
                    print(f"  Complete rows: {complete_csv:,} ({complete_csv/len(csv_df)*100:.1f}%)")
                else:
                    print(f"  No GPT columns found!")
        except Exception as e:
            print(f"{csv_file}: Error reading - {e}")

print(f"\n🎯 RECOMMENDATIONS:")
print("-" * 40)

if os.path.exists('ids_to_reprocess_gpt.csv'):
    print("1. ✅ Use 'ids_to_reprocess_gpt.csv' to resume GPT processing")
    print("2. 🔄 Run modified script on just the missing/incomplete IDs")
    print("3. 📊 This will be much faster than reprocessing everything")
    print("4. 💾 SQLite database will be updated with new results")
else:
    print("1. 🎉 Data appears complete!")
    print("2. 📊 Ready to proceed with model training")

print(f"\n✅ Analysis complete!")