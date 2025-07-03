import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import os
from typing import List, Dict
import tiktoken
import time
import random
from datetime import datetime

# Load data for cost estimation
train_features = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
train_labels = pd.read_csv('ds19-predictive-modeling-challenge/train_labels.csv')
train = pd.merge(train_features, train_labels, on='id')

# Cost calculation
def estimate_tokens(text: str) -> int:
    """Estimate tokens using tiktoken"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(str(text)))

# Sample 100 rows for test
test_sample = train.sample(100, random_state=42)

# Define the 5 custom features we'll create
feature_prompts = {
    'funder_org_type': """Classify this funder organization:
Funder: {funder}

Categories:
1=Government (ministry, district council, government)
2=International (World Bank, UN, foreign country)
3=NGO/Charity (foundation, vision, charity)
4=Religious (church, mission, islamic)
5=Private (company, individual, private)

Look for keywords and patterns. If unclear or "0", output 5.
Output only a number 1-5.""",
    
    'installer_name_quality': """Rate the data quality of this installer name:
Installer: {installer}

1=Very poor (empty, "0", single letter, test data)
2=Poor (heavy abbreviation like "DWE", "RWE")
3=Fair (partial name like "Commu", "Gov")
4=Good (clear abbreviation like "DANIDA", "KKKT")
5=Excellent (full name like "District Council", "World Vision")

Output only a number 1-5.""",
    
    'location_institution_type': """Identify institution from location names:
Ward: {ward}
Subvillage: {subvillage}

1=School/Education (shule, school, college)
2=Religious (church, mosque, mission, kanisa)
3=Health (hospital, dispensary, clinic)
4=Government (office, station, center)
5=None/Village (no institution identified)

Look for Swahili and English keywords.
Output only a number 1-5.""",
    
    'scheme_name_pattern': """Analyze water scheme naming pattern:
Scheme name: {scheme_name}

1=No name/Invalid (empty, single letter, "0")
2=Code only (like "BL Kikafu", "K", "Chal")
3=Location only (village/area name)
4=Descriptive (includes "water", "maji", "pipe", "scheme")
5=Full project name (complete with type and location)

Output only a number 1-5.""",
    
    'text_language': """Identify primary language of these text fields:
Funder: {funder}
Ward: {ward}
Subvillage: {subvillage}

1=English only
2=Mostly English with some local
3=Mixed equally
4=Mostly Swahili/local with some English
5=Swahili/local only

Consider: maji=water, shule=school, kanisa=church
Output only a number 1-5.""",
    
    'name_mismatch_flag': """Check if installer and funder names suggest coordination issues:
Funder: {funder}
Installer: {installer}

1=Perfect match (same organization/clear partnership like "DANIDA" funder with "DANIDA" installer)
2=Good match (related orgs like "Government" funder with "District Council" installer)
3=Neutral (different but compatible like "World Bank" funder with "DWE" installer)
4=Poor match (conflicting like "Private Individual" funder with "Government" installer)
5=Red flag (missing data, "0", or completely incompatible pairing)

Consider abbreviations and partnerships.
Output only a number 1-5."""
}

# Estimate tokens for one row
sample_row = test_sample.iloc[0]
total_input_tokens = 0
total_output_tokens = 0

print("Token estimation for 100 rows:")
print("-" * 50)

for feature_name, prompt_template in feature_prompts.items():
    # Fill in the prompt with actual data
    prompt = prompt_template.format(**sample_row.to_dict())
    input_tokens = estimate_tokens(prompt)
    output_tokens = 2  # We expect single digit responses
    
    total_input_tokens += input_tokens * 100
    total_output_tokens += output_tokens * 100
    
    print(f"{feature_name}: ~{input_tokens} input tokens per row")

print(f"\nTotal for 100 rows:")
print(f"Input tokens: {total_input_tokens:,}")
print(f"Output tokens: {total_output_tokens:,}")
print(f"Estimated cost: ${(total_input_tokens * 0.1 + total_output_tokens * 0.4) / 1_000_000:.4f}")

print(f"\nTotal for full dataset ({len(train):,} rows):")
full_input_tokens = total_input_tokens * len(train) / 100
full_output_tokens = total_output_tokens * len(train) / 100
full_cost = (full_input_tokens * 0.1 + full_output_tokens * 0.4) / 1_000_000
print(f"Input tokens: {full_input_tokens:,.0f}")
print(f"Output tokens: {full_output_tokens:,.0f}")
print(f"Estimated cost: ${full_cost:.2f}")

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    'max_retries': 6,
    'initial_delay': 1,
    'exponential_base': 2,
    'jitter': True,
    'max_concurrent': 100,  # Much higher since we're request-limited
    'requests_per_minute': 500,  # Your actual limit
    'tokens_per_minute': 200000,  # Your actual limit
}

# Track rate limiting
rate_limit_stats = {
    'requests_made': 0,
    'rate_limit_hits': 0,
    'errors': 0,
    'start_time': None
}

# Async implementation with exponential backoff
async def get_gpt_response(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, retry_count: int = 0) -> str:
    """Make async API call to GPT-4.1-nano with exponential backoff"""
    async with semaphore:  # Limit concurrent requests
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4.1-nano',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,  # Low temperature for consistent ratings
            'max_tokens': 5
        }
        
        try:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            ) as response:
                rate_limit_stats['requests_made'] += 1
                
                # Check for rate limit
                if response.status == 429:
                    rate_limit_stats['rate_limit_hits'] += 1
                    retry_after = response.headers.get('Retry-After', None)
                    
                    if retry_count >= RATE_LIMIT_CONFIG['max_retries']:
                        print(f"Max retries ({RATE_LIMIT_CONFIG['max_retries']}) exceeded")
                        return "3"
                    
                    # Calculate delay with exponential backoff
                    delay = RATE_LIMIT_CONFIG['initial_delay'] * (RATE_LIMIT_CONFIG['exponential_base'] ** retry_count)
                    if RATE_LIMIT_CONFIG['jitter']:
                        delay *= (1 + random.random())
                    
                    if retry_after:
                        delay = max(delay, float(retry_after))
                    
                    print(f"Rate limited. Retrying in {delay:.1f}s (attempt {retry_count + 1}/{RATE_LIMIT_CONFIG['max_retries']})")
                    await asyncio.sleep(delay)
                    
                    # Recursive retry
                    return await get_gpt_response(session, prompt, semaphore, retry_count + 1)
                
                # Handle other errors
                if response.status != 200:
                    error_data = await response.json()
                    print(f"API Error {response.status}: {error_data}")
                    rate_limit_stats['errors'] += 1
                    return "3"
                
                result = await response.json()
                return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            print(f"Error: {e}")
            rate_limit_stats['errors'] += 1
            return "3"  # Default middle value on error

async def process_batch(df: pd.DataFrame, feature_name: str, prompt_template: str) -> List[str]:
    """Process a batch of rows for one feature with progress tracking"""
    max_concurrent = RATE_LIMIT_CONFIG['max_concurrent']
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Initialize rate limit tracking
    if rate_limit_stats['start_time'] is None:
        rate_limit_stats['start_time'] = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        total_rows = len(df)
        
        print(f"Processing {total_rows} rows with max {max_concurrent} concurrent requests...")
        
        for idx, (_, row) in enumerate(df.iterrows()):
            prompt = prompt_template.format(**row.to_dict())
            task = get_gpt_response(session, prompt, semaphore)
            tasks.append(task)
            
            # Progress update every 1000 rows
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - rate_limit_stats['start_time']
                rate = rate_limit_stats['requests_made'] / elapsed if elapsed > 0 else 0
                print(f"Progress: {idx + 1}/{total_rows} rows, {rate:.1f} req/s, {rate_limit_stats['rate_limit_hits']} rate limits hit")
        
        results = await asyncio.gather(*tasks)
        
        # Final stats
        elapsed = time.time() - rate_limit_stats['start_time']
        print(f"Completed {feature_name}: {rate_limit_stats['requests_made']} requests in {elapsed:.1f}s")
        print(f"Rate limits hit: {rate_limit_stats['rate_limit_hits']}, Errors: {rate_limit_stats['errors']}")
        
        return results

async def generate_all_features(df: pd.DataFrame, save_progress: bool = True) -> pd.DataFrame:
    """Generate all 6 custom features with progress saving"""
    result_df = df.copy()
    
    for i, (feature_name, prompt_template) in enumerate(feature_prompts.items()):
        print(f"\n=== Generating {feature_name} ({i+1}/{len(feature_prompts)}) ===")
        
        # Check if we already have this feature (resume capability)
        progress_file = f'gpt_progress_{feature_name}.csv'
        if save_progress and os.path.exists(progress_file):
            print(f"Found existing progress for {feature_name}, loading...")
            existing_df = pd.read_csv(progress_file)
            result_df = result_df.merge(existing_df, on='id', how='left')
            continue
        
        # Reset stats for this feature
        rate_limit_stats['requests_made'] = 0
        rate_limit_stats['rate_limit_hits'] = 0
        rate_limit_stats['errors'] = 0
        rate_limit_stats['start_time'] = time.time()
        
        results = await process_batch(df, feature_name, prompt_template)
        
        # Convert to numeric, handling any non-numeric responses
        numeric_results = []
        invalid_count = 0
        for r in results:
            try:
                val = int(r.strip())
                if 1 <= val <= 5:
                    numeric_results.append(val)
                else:
                    numeric_results.append(3)  # Default
                    invalid_count += 1
            except:
                numeric_results.append(3)  # Default
                invalid_count += 1
        
        result_df[f'gpt_{feature_name}'] = numeric_results
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count}/{len(results)} responses were invalid and defaulted to 3")
        
        # Save progress
        if save_progress:
            progress_df = df[['id']].copy()
            progress_df[f'gpt_{feature_name}'] = numeric_results
            progress_df.to_csv(progress_file, index=False)
            print(f"Progress saved to {progress_file}")
        
        # Brief pause between features to be respectful
        if i < len(feature_prompts) - 1:
            print("Waiting 10 seconds before next feature...")
            await asyncio.sleep(10)
    
    return result_df

# Test implementation
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Ready to run GPT feature generation")
    print("="*50)
    
    # Check if we've already generated the features
    if os.path.exists('gpt_features_full.csv'):
        print("\nGPT features already exist at 'gpt_features_full.csv'")
        existing_features = pd.read_csv('gpt_features_full.csv')
        print(f"Shape: {existing_features.shape}")
        print("\nColumns:", existing_features.columns.tolist())
    else:
        print("\nNo existing GPT features found.")
        print("\nTo run on test sample (100 rows) and save:")
        print("""
import asyncio
result = asyncio.run(generate_all_features(test_sample))
gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
result[['id'] + gpt_cols].to_csv('gpt_features_test.csv', index=False)
""")
        print("\nTo run on full dataset and save:")
        print("""
import asyncio
result = asyncio.run(generate_all_features(train))
gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
result[['id'] + gpt_cols].to_csv('gpt_features_full.csv', index=False)
""")