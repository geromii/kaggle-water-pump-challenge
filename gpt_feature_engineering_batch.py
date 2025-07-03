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

# Sample 100 rows for test
test_sample = train.sample(100, random_state=42)

# Define the 6 custom features we'll create with BATCH-FRIENDLY prompts
feature_prompts = {
    'funder_org_type': """Classify these funder organizations (1=Government, 2=International, 3=NGO, 4=Religious, 5=Private):

{batch_data}

Output only numbers 1-5, one per line.""",
    
    'installer_name_quality': """Rate installer name quality (1=Very poor, 2=Poor, 3=Fair, 4=Good, 5=Excellent):

{batch_data}

Output only numbers 1-5, one per line.""",
    
    'location_institution_type': """Identify institution types (1=School, 2=Religious, 3=Health, 4=Government, 5=Village):

{batch_data}

Output only numbers 1-5, one per line.""",
    
    'scheme_name_pattern': """Classify scheme names (1=Invalid, 2=Code, 3=Location, 4=Descriptive, 5=Full):

{batch_data}

Output only numbers 1-5, one per line.""",
    
    'text_language': """Identify language (1=English, 2=Mostly English, 3=Mixed, 4=Mostly Local, 5=Local):

{batch_data}

Output only numbers 1-5, one per line.""",
    
    'name_mismatch_flag': """Rate funder-installer coordination (1=Perfect, 2=Good, 3=Neutral, 4=Poor, 5=Red flag):

{batch_data}

Output only numbers 1-5, one per line."""
}

# Batch size - fit as many as possible in context while staying under token limits
BATCH_SIZE = 20  # Conservative to start

def create_batch_prompt(df_batch: pd.DataFrame, feature_name: str) -> str:
    """Create a batch prompt for multiple rows"""
    
    if feature_name == 'funder_org_type':
        batch_data = "\n".join([f"{i+1}. Funder: {row['funder']}" 
                               for i, (_, row) in enumerate(df_batch.iterrows())])
    
    elif feature_name == 'installer_name_quality':
        batch_data = "\n".join([f"{i+1}. Installer: {row['installer']}" 
                               for i, (_, row) in enumerate(df_batch.iterrows())])
    
    elif feature_name == 'location_institution_type':
        batch_data = "\n".join([f"{i+1}. Ward: {row['ward']}, Subvillage: {row['subvillage']}" 
                               for i, (_, row) in enumerate(df_batch.iterrows())])
    
    elif feature_name == 'scheme_name_pattern':
        batch_data = "\n".join([f"{i+1}. Scheme: {row['scheme_name']}" 
                               for i, (_, row) in enumerate(df_batch.iterrows())])
    
    elif feature_name == 'text_language':
        batch_data = "\n".join([f"{i+1}. Funder: {row['funder']}, Ward: {row['ward']}, Subvillage: {row['subvillage']}" 
                               for i, (_, row) in enumerate(df_batch.iterrows())])
    
    elif feature_name == 'name_mismatch_flag':
        batch_data = "\n".join([f"{i+1}. Funder: {row['funder']}, Installer: {row['installer']}" 
                               for i, (_, row) in enumerate(df_batch.iterrows())])
    
    return feature_prompts[feature_name].format(batch_data=batch_data)

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    'max_retries': 6,
    'initial_delay': 1,
    'exponential_base': 2,
    'jitter': True,
    'max_concurrent': 100,  # High concurrency since we're request-limited
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

async def get_gpt_batch_response(session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore, retry_count: int = 0) -> List[str]:
    """Make async API call for batch processing"""
    async with semaphore:
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4.1-nano',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 100  # Increased for batch responses
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
                        return ["3"] * BATCH_SIZE  # Default values
                    
                    delay = RATE_LIMIT_CONFIG['initial_delay'] * (RATE_LIMIT_CONFIG['exponential_base'] ** retry_count)
                    if RATE_LIMIT_CONFIG['jitter']:
                        delay *= (1 + random.random())
                    
                    print(f"Rate limited. Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    return await get_gpt_batch_response(session, prompt, semaphore, retry_count + 1)
                
                if response.status != 200:
                    rate_limit_stats['errors'] += 1
                    return ["3"] * BATCH_SIZE
                
                result = await response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                # Parse batch response - expect one number per line
                lines = response_text.split('\n')
                values = []
                for line in lines:
                    try:
                        val = int(line.strip())
                        if 1 <= val <= 5:
                            values.append(str(val))
                        else:
                            values.append("3")
                    except:
                        values.append("3")
                
                # Ensure we return exactly BATCH_SIZE values
                while len(values) < BATCH_SIZE:
                    values.append("3")
                
                return values[:BATCH_SIZE]
                
        except Exception as e:
            print(f"Error: {e}")
            rate_limit_stats['errors'] += 1
            return ["3"] * BATCH_SIZE

async def process_feature_in_batches(df: pd.DataFrame, feature_name: str) -> List[str]:
    """Process a feature using batching"""
    semaphore = asyncio.Semaphore(RATE_LIMIT_CONFIG['max_concurrent'])
    
    # Calculate number of requests needed
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    total_requests_for_feature = num_batches
    
    print(f"Processing {len(df)} rows in {num_batches} batches of {BATCH_SIZE}")
    print(f"This feature will use {total_requests_for_feature} API requests (vs {len(df)} without batching)")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i+BATCH_SIZE]
            prompt = create_batch_prompt(batch_df, feature_name)
            task = get_gpt_batch_response(session, prompt, semaphore)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten batch results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        # Trim to exact length needed
        return all_results[:len(df)]

async def generate_all_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all features using efficient batching"""
    result_df = df.copy()
    
    # Calculate total speedup
    total_rows = len(df)
    features_count = len(feature_prompts)
    requests_without_batching = total_rows * features_count
    requests_with_batching = ((total_rows + BATCH_SIZE - 1) // BATCH_SIZE) * features_count
    speedup = requests_without_batching / requests_with_batching
    
    print(f"BATCHING EFFICIENCY:")
    print(f"Without batching: {requests_without_batching:,} requests")
    print(f"With batching: {requests_with_batching:,} requests")
    print(f"Speedup: {speedup:.1f}x faster!")
    print(f"Estimated time: {requests_with_batching / 500:.1f} minutes")
    print("=" * 50)
    
    for i, feature_name in enumerate(feature_prompts.keys()):
        print(f"\n=== Generating {feature_name} ({i+1}/{features_count}) ===")
        
        rate_limit_stats['requests_made'] = 0
        rate_limit_stats['rate_limit_hits'] = 0
        rate_limit_stats['errors'] = 0
        rate_limit_stats['start_time'] = time.time()
        
        results = await process_feature_in_batches(df, feature_name)
        
        # Convert to numeric
        numeric_results = []
        for r in results:
            try:
                val = int(r)
                if 1 <= val <= 5:
                    numeric_results.append(val)
                else:
                    numeric_results.append(3)
            except:
                numeric_results.append(3)
        
        result_df[f'gpt_{feature_name}'] = numeric_results
        
        elapsed = time.time() - rate_limit_stats['start_time']
        print(f"Completed in {elapsed:.1f}s. Rate limits: {rate_limit_stats['rate_limit_hits']}, Errors: {rate_limit_stats['errors']}")
        
        if i < features_count - 1:
            print("Brief pause...")
            await asyncio.sleep(2)
    
    return result_df

# Test with small sample first
if __name__ == "__main__":
    print("Testing batch processing on sample...")
    # Test on small sample first
    result = asyncio.run(generate_all_features_batch(test_sample))
    print(f"\nTest completed! Shape: {result.shape}")
    
    gpt_cols = [col for col in result.columns if col.startswith('gpt_')]
    print(f"Generated features: {gpt_cols}")
    
    for col in gpt_cols:
        print(f"{col}: min={result[col].min()}, max={result[col].max()}, mean={result[col].mean():.2f}")