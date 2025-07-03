"""
Improved GPT prompts using prompt engineering best practices
"""

# IMPROVED SYSTEM PROMPT
IMPROVED_SYSTEM_PROMPT = """You are an expert water infrastructure analyst specializing in rural water systems in Tanzania. You have deep knowledge of:

- Tanzanian organizations (government, NGOs, international funders)
- Water pump installation practices and data quality patterns
- Swahili language and local naming conventions
- Rural vs urban infrastructure differences

Your task is to analyze water pump data and provide consistent, calibrated ratings. Always respond with valid JSON in the exact format requested. Base your analysis on patterns you would expect in real Tanzanian water infrastructure data."""

# IMPROVED USER PROMPT TEMPLATE
def create_improved_prompt(df_batch) -> str:
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
    
    # Add the data
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

# CONFIGURATION IMPROVEMENTS
IMPROVED_CONFIG = {
    'model': 'gpt-4.1-mini',
    'temperature': 0.2,  # Slightly higher for more nuanced reasoning
    'max_tokens': 1500,  # More tokens for complex reasoning
    'top_p': 0.9,  # Slight nucleus sampling for better quality
    'frequency_penalty': 0.1,  # Reduce repetitive patterns
    'presence_penalty': 0.1,   # Encourage diverse reasoning
}