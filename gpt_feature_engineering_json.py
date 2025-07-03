import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import os
from typing import List, Dict
import time
import random
import re
import sqlite3

# Load data
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')
test_sample = train.sample(100, random_state=42)

# IMPROVED VERTICAL BATCHING with enhanced prompts
def create_vertical_batch_prompt(df_batch: pd.DataFrame) -> str:
    """Create an improved prompt using prompt engineering best practices"""
    
    prompt = """Analyze these Tanzanian water pump records and rate each on 6 dimensions using the 1-5 scales below.

CONTEXT: These are rural water pumps in Tanzania. Consider typical patterns:
- Government funders (ministries, districts) vs private/NGO funders
- Data quality varies widely (abbreviations, missing info, inconsistent naming)
- Most locations are villages, but some serve schools/health facilities
- Mixed English/Swahili naming patterns indicate different development eras

RATING SCALES WITH EXAMPLES:

1. FUNDER_ORG_TYPE (Organization type):
   1=Government (Ministry of Water, District Council, Halmashauri)
   2=International (World Bank, UNICEF, Danida, Germany Republic)  
   3=NGO/Charity (World Vision, Oxfam, Foundation, Trust)
   4=Religious (Church, Mission, Islamic, KKKT, TCRS)
   5=Private (Private Individual, Company, Local business)

2. INSTALLER_QUALITY (Data completeness/quality):
   1=Very poor (empty, "0", single letters, clearly invalid)
   2=Poor (heavy abbreviations: "DWE", "RWE", unclear codes)
   3=Fair (partial names: "Commu", "Gov", recognizable but incomplete)
   4=Good (clear abbreviations: "DANIDA", "KKKT", institutional codes)
   5=Excellent (full proper names: "District Council", "World Vision Tanzania")

3. LOCATION_TYPE (Institution served):
   1=School/Education (contains: school, shule, college, education)
   2=Religious (contains: church, mosque, mission, kanisa, dini)
   3=Health (contains: hospital, dispensary, clinic, afya)
   4=Government (contains: office, station, serikali, government center)
   5=Village/General (standard village/community location, no special institution)

4. SCHEME_PATTERN (Project naming quality):
   1=Invalid/Missing (empty, "nan", single characters, clearly invalid)
   2=Code only (cryptic codes: "BL Kikafu", "K", "Chal", numbers only)
   3=Location only (just place names with no project indication)
   4=Descriptive (includes water-related terms: "maji", "water", "pipe", "scheme")
   5=Full project name (complete with type, location, and descriptive elements)

5. TEXT_LANGUAGE (Language pattern):
   1=English only (all fields in English, formal institutional language)
   2=Mostly English (predominantly English with occasional local terms)
   3=Mixed (balanced mix of English and Swahili/local languages)
   4=Mostly local (predominantly Swahili/local with some English)
   5=Local only (primarily Swahili/local language, minimal English)

6. COORDINATION (Funder-installer alignment):
   1=Perfect (same organization: "DANIDA" funder + "DANIDA" installer)
   2=Good (logical partnerships: Government funder + District installer)
   3=Neutral (different but compatible: World Bank + local contractor)
   4=Poor (mismatched: Private funder + Government installer)
   5=Red flag (missing data, inconsistent, or highly suspicious pairings)

DECISION PROCESS:
1. Read each field carefully, noting language, completeness, and patterns
2. Consider the rural Tanzanian context and typical infrastructure patterns
3. Look for consistency between funder/installer relationships
4. Rate based on the specific criteria above, not general impressions
5. When uncertain, use the middle value (3) for that dimension

DATA TO ANALYZE:
"""
    
    # Add the data with improved formatting
    for i, (_, row) in enumerate(df_batch.iterrows()):
        # Clean and format the data presentation
        funder = str(row['funder']) if pd.notna(row['funder']) else 'MISSING'
        installer = str(row['installer']) if pd.notna(row['installer']) else 'MISSING'
        ward = str(row['ward']) if pd.notna(row['ward']) else 'MISSING'
        subvillage = str(row['subvillage']) if pd.notna(row['subvillage']) else 'MISSING'
        scheme = str(row['scheme_name']) if pd.notna(row['scheme_name']) else 'MISSING'
        
        prompt += f"""
{i+1}. ID: {row['id']}
   Funder: {funder}
   Installer: {installer}
   Ward: {ward}
   Subvillage: {subvillage}
   Scheme: {scheme}"""
    
    prompt += f"""

REQUIRED OUTPUT FORMAT:
Return a JSON object with a "results" array containing exactly {len(df_batch)} objects.
Each object must have all 6 numeric ratings (1-5) and the exact field names shown below:

{{
  "results": [
    {{
      "id": {df_batch.iloc[0]['id']},
      "funder_org_type": [1-5],
      "installer_quality": [1-5], 
      "location_type": [1-5],
      "scheme_pattern": [1-5],
      "text_language": [1-5],
      "coordination": [1-5]
    }},
    ...
  ]
}}

IMPORTANT: Provide ratings for ALL {len(df_batch)} records in the same order as listed above."""

    return prompt

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    'max_retries': 6,
    'initial_delay': 1,
    'exponential_base': 2,
    'jitter': True,
    'max_concurrent': 20,  # Much lower to avoid rate limit stampede
    'requests_per_minute': 500,
    'tokens_per_minute': 200000,
}

# Track rate limiting
rate_limit_stats = {
    'requests_made': 0,
    'rate_limit_hits': 0,
    'errors': 0,
    'start_time': None
}

def init_sqlite_db(db_path: str = 'gpt_features_progress.db'):
    """Initialize SQLite database for progress tracking with WAL mode"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrent access
    cursor.execute('PRAGMA journal_mode=WAL')
    
    # Optimize for faster writes
    cursor.execute('PRAGMA synchronous=NORMAL')  # Faster than FULL, still safe
    cursor.execute('PRAGMA cache_size=10000')    # 10MB cache
    cursor.execute('PRAGMA temp_store=MEMORY')   # Use memory for temp tables
    cursor.execute('PRAGMA mmap_size=268435456') # 256MB memory mapping
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gpt_features (
            id INTEGER PRIMARY KEY,
            gpt_funder_org_type REAL,
            gpt_installer_quality REAL,
            gpt_location_type REAL,
            gpt_scheme_pattern REAL,
            gpt_text_language REAL,
            gpt_coordination REAL,
            batch_completed INTEGER,
            timestamp REAL
        )
    ''')
    
    # Create index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_completed ON gpt_features(batch_completed)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_id ON gpt_features(id)')
    
    conn.commit()
    conn.close()
    print(f"✅ SQLite initialized with WAL mode: {db_path}")
    return db_path

def save_batch_to_sqlite(batch_results: List[Dict], batch_num: int, db_path: str = 'gpt_features_progress.db'):
    """Save a single batch of results to SQLite with WAL optimizations"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Use WAL mode optimizations for this connection
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA synchronous=NORMAL')
    
    current_time = time.time()
    
    # Use transaction for batch insert (more efficient)
    cursor.execute('BEGIN TRANSACTION')
    try:
        for result in batch_results:
            cursor.execute('''
                INSERT OR REPLACE INTO gpt_features 
                (id, gpt_funder_org_type, gpt_installer_quality, gpt_location_type, 
                 gpt_scheme_pattern, gpt_text_language, gpt_coordination, batch_completed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['id'],
                result['funder_org_type'],
                result['installer_quality'], 
                result['location_type'],
                result['scheme_pattern'],
                result['text_language'],
                result['coordination'],
                batch_num,
                current_time
            ))
        cursor.execute('COMMIT')
    except Exception as e:
        cursor.execute('ROLLBACK')
        raise e
    finally:
        conn.close()

def load_existing_results(db_path: str = 'gpt_features_progress.db') -> pd.DataFrame:
    """Load existing results from SQLite"""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM gpt_features ORDER BY id', conn)
    conn.close()
    return df

def get_completed_batch_count(db_path: str = 'gpt_features_progress.db') -> int:
    """Get the highest completed batch number"""
    if not os.path.exists(db_path):
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(batch_completed) FROM gpt_features')
    result = cursor.fetchone()[0]
    conn.close()
    return result if result is not None else 0

def try_fix_json(response_text: str) -> str:
    """Try to fix common JSON truncation issues"""
    try:
        # Remove any trailing incomplete text
        response_text = response_text.strip()
        
        # If it ends with a truncated "coordination": value, try to complete it
        if response_text.endswith('"coordination":'):
            response_text += ' 3'
        elif response_text.endswith('"coordination": '):
            response_text += '3'
        elif re.search(r'"coordination":\s*$', response_text):
            response_text += '3'
        
        # Try to close any unclosed braces/brackets
        open_braces = response_text.count('{') - response_text.count('}')
        open_brackets = response_text.count('[') - response_text.count(']')
        
        # Add missing closing braces and brackets
        response_text += '}' * open_braces
        response_text += ']' * open_brackets
        
        # Validate that it's proper JSON now
        json.loads(response_text)
        return response_text
        
    except:
        return None

async def get_gpt_json_response(session: aiohttp.ClientSession, prompt: str, expected_count: int, semaphore: asyncio.Semaphore, retry_count: int = 0) -> List[Dict]:
    """Make async API call expecting JSON response"""
    async with semaphore:
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4.1-nano',
            'messages': [
                {'role': 'system', 'content': 'You are an expert water infrastructure analyst specializing in rural water systems in Tanzania. You have deep knowledge of Tanzanian organizations, water pump installation practices, Swahili language, and local naming conventions. Provide consistent, calibrated ratings based on real Tanzanian infrastructure patterns. Always respond with valid JSON in the exact format requested.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2,  # Slightly higher for more nuanced reasoning
            'max_tokens': 1500,  # More tokens for complex reasoning
            'top_p': 0.9,  # Quality-focused nucleus sampling
            'frequency_penalty': 0.1,  # Reduce repetitive patterns
            'response_format': {'type': 'json_object'}  # Force JSON format
        }
        
        try:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            ) as response:
                rate_limit_stats['requests_made'] += 1
                
                if response.status == 429:
                    rate_limit_stats['rate_limit_hits'] += 1
                    if retry_count >= RATE_LIMIT_CONFIG['max_retries']:
                        return [{"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3}] * expected_count
                    
                    delay = RATE_LIMIT_CONFIG['initial_delay'] * (RATE_LIMIT_CONFIG['exponential_base'] ** retry_count)
                    if RATE_LIMIT_CONFIG['jitter']:
                        delay *= (1 + random.random())
                    
                    print(f"Rate limited. Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    return await get_gpt_json_response(session, prompt, expected_count, semaphore, retry_count + 1)
                
                if response.status != 200:
                    rate_limit_stats['errors'] += 1
                    print(f"API Error {response.status}")
                    return [{"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3}] * expected_count
                
                result = await response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                try:
                    # Parse JSON response
                    json_data = json.loads(response_text)
                    
                    # Extract the results array from the object
                    if isinstance(json_data, dict):
                        if 'results' in json_data:
                            json_data = json_data['results']
                        elif len(json_data) == 1:
                            # If single key, extract the array
                            json_data = list(json_data.values())[0]
                        else:
                            # If it's a single object result, wrap it in an array
                            json_data = [json_data]
                    
                    if not isinstance(json_data, list):
                        print(f"Expected array, got {type(json_data)}")
                        print(f"Content: {json_data}")
                        raise ValueError("Not an array")
                    
                    # Validate and clean each entry
                    cleaned_data = []
                    for item in json_data:
                        if not isinstance(item, dict):
                            continue
                        
                        cleaned_item = {
                            'id': item.get('id', 0),
                            'funder_org_type': max(1, min(5, item.get('funder_org_type', 3))),
                            'installer_quality': max(1, min(5, item.get('installer_quality', 3))),
                            'location_type': max(1, min(5, item.get('location_type', 5))),
                            'scheme_pattern': max(1, min(5, item.get('scheme_pattern', 3))),
                            'text_language': max(1, min(5, item.get('text_language', 3))),
                            'coordination': max(1, min(5, item.get('coordination', 3)))
                        }
                        cleaned_data.append(cleaned_item)
                    
                    # Ensure we have the right number of entries
                    while len(cleaned_data) < expected_count:
                        cleaned_data.append({"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3})
                    
                    return cleaned_data[:expected_count]
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parse error: {e}")
                    print(f"Response length: {len(response_text)} chars")
                    
                    # Try to fix common JSON truncation issues
                    fixed_response = try_fix_json(response_text)
                    if fixed_response:
                        try:
                            json_data = json.loads(fixed_response)
                            if isinstance(json_data, dict) and 'results' in json_data:
                                json_data = json_data['results']
                            if isinstance(json_data, list):
                                print("✅ Successfully recovered truncated JSON")
                                # Continue with normal processing
                                cleaned_data = []
                                for item in json_data:
                                    if isinstance(item, dict):
                                        cleaned_item = {
                                            'id': item.get('id', 0),
                                            'funder_org_type': max(1, min(5, item.get('funder_org_type', 3))),
                                            'installer_quality': max(1, min(5, item.get('installer_quality', 3))),
                                            'location_type': max(1, min(5, item.get('location_type', 5))),
                                            'scheme_pattern': max(1, min(5, item.get('scheme_pattern', 3))),
                                            'text_language': max(1, min(5, item.get('text_language', 3))),
                                            'coordination': max(1, min(5, item.get('coordination', 3)))
                                        }
                                        cleaned_data.append(cleaned_item)
                                
                                while len(cleaned_data) < expected_count:
                                    cleaned_data.append({"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3})
                                
                                return cleaned_data[:expected_count]
                        except:
                            pass
                    
                    print(f"❌ Could not recover JSON, using defaults")
                    rate_limit_stats['errors'] += 1
                    return [{"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3}] * expected_count
                    
        except Exception as e:
            print(f"Error: {e}")
            rate_limit_stats['errors'] += 1
            return [{"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3}] * expected_count

async def process_all_features_json(df: pd.DataFrame, batch_size: int = 10, save_progress: bool = True) -> pd.DataFrame:
    """Process ALL features using JSON vertical batching with comprehensive progress tracking"""
    semaphore = asyncio.Semaphore(RATE_LIMIT_CONFIG['max_concurrent'])
    
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"🚀 JSON VERTICAL BATCHING:")
    print(f"Processing {len(df):,} rows in {num_batches:,} batches of {batch_size}")
    print(f"Total requests: {num_batches:,} (vs {len(df)*6:,} individual requests)")
    print(f"Speedup: {(len(df)*6)/num_batches:.0f}x faster!")
    print(f"Estimated time: {num_batches/500:.1f} minutes")
    print(f"Expected cost: ~$0.95")
    print("=" * 60)
    
    # Initialize SQLite database and check for existing progress
    db_path = 'gpt_features_progress.db'
    completed_batches_so_far = 0
    if save_progress:
        init_sqlite_db(db_path)
        completed_batches_so_far = get_completed_batch_count(db_path)
        if completed_batches_so_far > 0:
            print(f"📂 Found existing progress: {completed_batches_so_far} batches completed")
            existing_df = load_existing_results(db_path)
            print(f"✅ Loaded {len(existing_df)} existing results")
            
            # 🔧 DEBUG: Check which IDs from current dataset are actually missing
            print(f"🔧 DEBUG: Current dataset has {len(df)} rows, needs {num_batches} batches")
            print(f"🔧 DEBUG: SQLite has {completed_batches_so_far} total batches completed")
            
            # Check which specific IDs from current dataset are missing
            current_ids = set(df['id'])
            existing_ids = set(existing_df['id']) if len(existing_df) > 0 else set()
            missing_from_current = current_ids - existing_ids
            
            print(f"🔧 DEBUG: Current dataset IDs: {len(current_ids)}")
            print(f"🔧 DEBUG: Existing IDs in SQLite: {len(existing_ids)}")
            print(f"🔧 DEBUG: Missing from current dataset: {len(missing_from_current)}")
            
            # Only return early if ALL current dataset IDs are already processed
            if len(missing_from_current) == 0:
                print(f"🎉 All current dataset IDs already completed! Returning existing results.")
                result_df = df.merge(existing_df[['id', 'gpt_funder_org_type', 'gpt_installer_quality', 
                                                 'gpt_location_type', 'gpt_scheme_pattern', 
                                                 'gpt_text_language', 'gpt_coordination']], on='id', how='left')
                return result_df
            
            # Filter current dataset to only missing IDs
            df = df[df['id'].isin(missing_from_current)].reset_index(drop=True)
            num_batches = (len(df) + batch_size - 1) // batch_size
            
            print(f"🎯 Filtering to {len(df)} truly missing rows, requiring {num_batches} new batches")
            print(f"🔄 Will process {num_batches} new batches for missing data")
            completed_batches_so_far = 0  # Reset for this subset
        else:
            print(f"📂 Starting fresh - no existing progress found")
    
    rate_limit_stats['start_time'] = time.time()
    rate_limit_stats['requests_made'] = 0
    rate_limit_stats['rate_limit_hits'] = 0
    rate_limit_stats['errors'] = 0
    
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    print("📊 Progress updates every 10% completion")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        completed_batches = completed_batches_so_far  # Start from where we left off
        
        # Create tasks only for remaining batches
        start_batch = completed_batches_so_far
        remaining_batches = num_batches - completed_batches_so_far
        
        print(f"🎯 Creating tasks for {remaining_batches} remaining batches (skipping first {start_batch})...")
        
        for i in range(start_batch * batch_size, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            prompt = create_vertical_batch_prompt(batch_df)
            task = get_gpt_json_response(session, prompt, len(batch_df), semaphore)
            tasks.append(task)
        
        print(f"🎯 Processing {len(tasks)} remaining batches...")
        
        # Process tasks with progress tracking
        batch_results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                batch_results.append(result)
                completed_batches += 1
                
                # Save batch to SQLite immediately
                if save_progress:
                    save_batch_to_sqlite(result, completed_batches, db_path)
                
                # Progress updates
                progress_percent = (completed_batches / num_batches) * 100
                
                # Detailed progress updates every 1%
                if completed_batches % max(1, num_batches // 100) == 0 or completed_batches == num_batches:
                    elapsed = time.time() - rate_limit_stats['start_time']
                    rate = completed_batches / elapsed if elapsed > 0 else 0
                    eta = (num_batches - completed_batches) / rate if rate > 0 else 0
                    
                    print(f"🔄 {progress_percent:5.1f}% | {completed_batches:,}/{num_batches:,} | "
                          f"⏱️  {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | "
                          f"⚡ {rate:.1f} req/s")
                
                # Major progress updates every 10%
                elif completed_batches % max(1, num_batches // 10) == 0 or completed_batches == num_batches:
                    elapsed = time.time() - rate_limit_stats['start_time']
                    rate = completed_batches / elapsed if elapsed > 0 else 0
                    eta = (num_batches - completed_batches) / rate if rate > 0 else 0
                    
                    print(f"📈 {progress_percent:5.1f}% | {completed_batches:,}/{num_batches:,} batches | "
                          f"⏱️  {elapsed/60:.1f}min elapsed | ETA: {eta/60:.1f}min | "
                          f"🚫 Errors: {rate_limit_stats['errors']} | "
                          f"⚡ Rate: {rate:.1f} req/s")
                
                # Optional: Create CSV export every 10% for monitoring
                if save_progress and completed_batches % max(1, num_batches // 10) == 0:
                    sqlite_df = load_existing_results(db_path)
                    if not sqlite_df.empty:
                        gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                                   'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
                        export_df = sqlite_df[['id'] + gpt_cols]
                        export_df.to_csv('gpt_features_progress.csv', index=False)
                        print(f"💾 Progress exported to CSV: {len(export_df)} rows")
                        
            except Exception as e:
                print(f"❌ Error in batch {i}: {e}")
                rate_limit_stats['errors'] += 1
                # Use default values for failed batch
                batch_size_actual = min(batch_size, len(df) - i * batch_size)
                default_result = [{"id": 0, "funder_org_type": 3, "installer_quality": 3, "location_type": 5, "scheme_pattern": 3, "text_language": 3, "coordination": 3}] * batch_size_actual
                batch_results.append(default_result)
                completed_batches += 1
        
        # Sort results to match original order
        sorted_results = []
        for i, batch_result in enumerate(batch_results):
            sorted_results.extend(batch_result)
        
        # Convert to DataFrame
        result_df = df.copy()
        
        # Add GPT features with error handling
        successful_assignments = 0
        for i, result in enumerate(sorted_results[:len(df)]):
            try:
                if i < len(df):
                    result_df.loc[df.index[i], 'gpt_funder_org_type'] = result['funder_org_type']
                    result_df.loc[df.index[i], 'gpt_installer_quality'] = result['installer_quality'] 
                    result_df.loc[df.index[i], 'gpt_location_type'] = result['location_type']
                    result_df.loc[df.index[i], 'gpt_scheme_pattern'] = result['scheme_pattern']
                    result_df.loc[df.index[i], 'gpt_text_language'] = result['text_language']
                    result_df.loc[df.index[i], 'gpt_coordination'] = result['coordination']
                    successful_assignments += 1
            except Exception as e:
                print(f"⚠️  Error assigning result {i}: {e}")
                # Use defaults
                result_df.loc[df.index[i], 'gpt_funder_org_type'] = 3
                result_df.loc[df.index[i], 'gpt_installer_quality'] = 3
                result_df.loc[df.index[i], 'gpt_location_type'] = 5
                result_df.loc[df.index[i], 'gpt_scheme_pattern'] = 3
                result_df.loc[df.index[i], 'gpt_text_language'] = 3
                result_df.loc[df.index[i], 'gpt_coordination'] = 3
        
        elapsed = time.time() - rate_limit_stats['start_time']
        
        # Final results from SQLite
        if save_progress:
            sqlite_df = load_existing_results(db_path)
            if not sqlite_df.empty:
                gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                           'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
                final_df = sqlite_df[['id'] + gpt_cols]
                final_df.to_csv('gpt_features_full.csv', index=False)
                print(f"💾 Final results saved to: gpt_features_full.csv ({len(final_df)} rows)")
                print(f"💾 SQLite database: {db_path} (persistent backup)")
            else:
                print(f"⚠️  No data in SQLite database to export")
                # Fallback to memory results
                gpt_cols = ['gpt_funder_org_type', 'gpt_installer_quality', 'gpt_location_type', 
                           'gpt_scheme_pattern', 'gpt_text_language', 'gpt_coordination']
                final_df = result_df[['id'] + gpt_cols]
                final_df.to_csv('gpt_features_full.csv', index=False)
                print(f"💾 Final results saved to: gpt_features_full.csv (from memory)")
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏰ Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"📊 Processed: {successful_assignments:,}/{len(df):,} rows successfully")
        print(f"🔄 Requests made: {rate_limit_stats['requests_made']:,}")
        print(f"⚡ Rate limits hit: {rate_limit_stats['rate_limit_hits']}")
        print(f"❌ Total errors: {rate_limit_stats['errors']}")
        print(f"💰 Estimated cost: ${(rate_limit_stats['requests_made'] * 0.0002):.2f}")
        
        return result_df

async def save_intermediate_progress(df: pd.DataFrame, batch_results: list, batch_size: int, filename: str):
    """Save intermediate progress to file"""
    try:
        # Flatten current results
        current_results = []
        for batch_result in batch_results:
            current_results.extend(batch_result)
        
        # Create progress DataFrame
        progress_data = []
        for i, result in enumerate(current_results):
            if i < len(df):
                progress_data.append({
                    'id': df.iloc[i]['id'],
                    'gpt_funder_org_type': result['funder_org_type'],
                    'gpt_installer_quality': result['installer_quality'],
                    'gpt_location_type': result['location_type'],
                    'gpt_scheme_pattern': result['scheme_pattern'],
                    'gpt_text_language': result['text_language'],
                    'gpt_coordination': result['coordination']
                })
        
        progress_df = pd.DataFrame(progress_data)
        progress_df.to_csv(filename, index=False)
        print(f"💾 Progress saved: {len(progress_df)} rows")
        
    except Exception as e:
        print(f"⚠️  Could not save progress: {e}")

# Test
if __name__ == "__main__":
    print("Testing JSON vertical batching...")
    result = asyncio.run(process_all_features_json(test_sample, batch_size=5))
    
    gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
    print(f"\nGenerated features: {gpt_cols}")
    
    for col in gpt_cols:
        print(f"{col}: min={result[col].min()}, max={result[col].max()}, mean={result[col].mean():.2f}")
    
    print("\nSample results:")
    print(result[['id'] + gpt_cols].head())