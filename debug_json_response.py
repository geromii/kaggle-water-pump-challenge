import asyncio
import aiohttp
import json
import os
import pandas as pd

# Load sample data
train = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
sample_row = train.iloc[0]

async def test_json_response():
    """Test what GPT-4.1-mini actually returns"""
    
    prompt = f"""Analyze this water pump and rate it on 6 dimensions (1-5 scale). Return JSON format.

For the pump, provide ratings for:
- funder_org_type: 1=Government, 2=International, 3=NGO/Charity, 4=Religious, 5=Private
- installer_quality: 1=Very poor data, 2=Poor (abbrev), 3=Fair, 4=Good, 5=Excellent (full name)
- location_type: 1=School, 2=Religious, 3=Health, 4=Government, 5=Village/None
- scheme_pattern: 1=Invalid, 2=Code only, 3=Location only, 4=Descriptive, 5=Full name
- text_language: 1=English only, 2=Mostly English, 3=Mixed, 4=Mostly Local, 5=Local only
- coordination: 1=Perfect match, 2=Good match, 3=Neutral, 4=Poor match, 5=Red flag

Data:
ID: {sample_row['id']}
Funder: {sample_row['funder']}
Installer: {sample_row['installer']}
Ward: {sample_row['ward']}
Subvillage: {sample_row['subvillage']}
Scheme: {sample_row['scheme_name']}

Return JSON object:
{{
  "id": {sample_row['id']},
  "funder_org_type": 1,
  "installer_quality": 2,
  "location_type": 5,
  "scheme_pattern": 3,
  "text_language": 2,
  "coordination": 3
}}"""

    headers = {
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-4.1-mini',
        'messages': [
            {'role': 'system', 'content': 'You are a data analyst. Always respond with valid JSON as requested.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.1,
        'max_tokens': 200,
        'response_format': {'type': 'json_object'}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        ) as response:
            result = await response.json()
            response_text = result['choices'][0]['message']['content'].strip()
            
            print("RAW RESPONSE:")
            print("=" * 50)
            print(response_text)
            print("=" * 50)
            
            print(f"\nResponse type: {type(response_text)}")
            print(f"Response length: {len(response_text)}")
            
            try:
                parsed = json.loads(response_text)
                print(f"\nParsed type: {type(parsed)}")
                print(f"Parsed content: {parsed}")
                
                if isinstance(parsed, dict):
                    print(f"Keys: {list(parsed.keys())}")
                    
            except json.JSONDecodeError as e:
                print(f"\nJSON parsing failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_json_response())