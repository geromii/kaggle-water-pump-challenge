import tiktoken
import pandas as pd

# Load sample data
train = pd.read_csv('ds19-predictive-modeling-challenge/train_features.csv')
sample_rows = train.head(3)

print("COST COMPARISON: Individual vs JSON Batching")
print("=" * 60)

encoding = tiktoken.encoding_for_model("gpt-4")

# INDIVIDUAL REQUEST (original approach)
print("\n1️⃣ INDIVIDUAL REQUEST APPROACH:")
print("-" * 40)

individual_prompt = """Classify this funder organization:
Funder: Government Of Tanzania

Categories:
1=Government (ministry, district council, government)
2=International (World Bank, UN, foreign country)
3=NGO/Charity (foundation, vision, charity)
4=Religious (church, mission, islamic)
5=Private (company, individual, private)

Look for keywords and patterns. If unclear or "0", output 5.
Output only a number 1-5."""

individual_response = "1"

print("Sample prompt:")
print(individual_prompt)
print(f"\nSample response: {individual_response}")

individual_input_tokens = len(encoding.encode(individual_prompt))
individual_output_tokens = len(encoding.encode(individual_response))

print(f"\nTokens per request:")
print(f"Input: {individual_input_tokens}")
print(f"Output: {individual_output_tokens}")
print(f"Total: {individual_input_tokens + individual_output_tokens}")

# JSON BATCH REQUEST
print("\n\n2️⃣ JSON BATCH REQUEST APPROACH:")
print("-" * 40)

json_prompt = f"""Analyze these water pumps and rate each on 6 dimensions (1-5 scale). Return JSON format.

For each pump, provide ratings for:
- funder_org_type: 1=Government, 2=International, 3=NGO/Charity, 4=Religious, 5=Private
- installer_quality: 1=Very poor data, 2=Poor (abbrev), 3=Fair, 4=Good, 5=Excellent (full name)
- location_type: 1=School, 2=Religious, 3=Health, 4=Government, 5=Village/None
- scheme_pattern: 1=Invalid, 2=Code only, 3=Location only, 4=Descriptive, 5=Full name
- text_language: 1=English only, 2=Mostly English, 3=Mixed, 4=Mostly Local, 5=Local only
- coordination: 1=Perfect match, 2=Good match, 3=Neutral, 4=Poor match, 5=Red flag

Data:

1. ID: {sample_rows.iloc[0]['id']}
   Funder: {sample_rows.iloc[0]['funder']}
   Installer: {sample_rows.iloc[0]['installer']}
   Ward: {sample_rows.iloc[0]['ward']}
   Subvillage: {sample_rows.iloc[0]['subvillage']}
   Scheme: {sample_rows.iloc[0]['scheme_name']}

2. ID: {sample_rows.iloc[1]['id']}
   Funder: {sample_rows.iloc[1]['funder']}
   Installer: {sample_rows.iloc[1]['installer']}
   Ward: {sample_rows.iloc[1]['ward']}
   Subvillage: {sample_rows.iloc[1]['subvillage']}
   Scheme: {sample_rows.iloc[1]['scheme_name']}

3. ID: {sample_rows.iloc[2]['id']}
   Funder: {sample_rows.iloc[2]['funder']}
   Installer: {sample_rows.iloc[2]['installer']}
   Ward: {sample_rows.iloc[2]['ward']}
   Subvillage: {sample_rows.iloc[2]['subvillage']}
   Scheme: {sample_rows.iloc[2]['scheme_name']}

Return JSON array with 3 objects:
[
  {{
    "id": {sample_rows.iloc[0]['id']},
    "funder_org_type": 1,
    "installer_quality": 2,
    "location_type": 5,
    "scheme_pattern": 3,
    "text_language": 2,
    "coordination": 3
  }},
  ...
]"""

json_response = f"""[
  {{
    "id": {sample_rows.iloc[0]['id']},
    "funder_org_type": 1,
    "installer_quality": 2,
    "location_type": 5,
    "scheme_pattern": 3,
    "text_language": 2,
    "coordination": 3
  }},
  {{
    "id": {sample_rows.iloc[1]['id']},
    "funder_org_type": 2,
    "installer_quality": 3,
    "location_type": 5,
    "scheme_pattern": 2,
    "text_language": 1,
    "coordination": 4
  }},
  {{
    "id": {sample_rows.iloc[2]['id']},
    "funder_org_type": 1,
    "installer_quality": 2,
    "location_type": 5,
    "scheme_pattern": 3,
    "text_language": 2,
    "coordination": 2
  }}
]"""

print("Sample prompt:")
print(json_prompt[:500] + "..." if len(json_prompt) > 500 else json_prompt)
print(f"\nSample response:")
print(json_response)

json_input_tokens = len(encoding.encode(json_prompt))
json_output_tokens = len(encoding.encode(json_response))

print(f"\nTokens per batch request (3 rows, 6 features each = 18 individual requests worth):")
print(f"Input: {json_input_tokens}")
print(f"Output: {json_output_tokens}")
print(f"Total: {json_input_tokens + json_output_tokens}")

# COST COMPARISON
print("\n\n💰 COST COMPARISON:")
print("=" * 40)

# Individual approach (18 requests for 3 rows × 6 features)
individual_total_input = individual_input_tokens * 18
individual_total_output = individual_output_tokens * 18
# GPT-4.1-mini pricing: $0.40/1M input, $1.60/1M output
individual_total_cost = (individual_total_input * 0.4 + individual_total_output * 1.6) / 1_000_000

print(f"Individual approach (18 requests):")
print(f"Input tokens: {individual_total_input:,}")
print(f"Output tokens: {individual_total_output:,}")
print(f"Cost: ${individual_total_cost:.6f}")

# JSON batch approach (1 request)
json_total_cost = (json_input_tokens * 0.4 + json_output_tokens * 1.6) / 1_000_000

print(f"\nJSON batch approach (1 request):")
print(f"Input tokens: {json_input_tokens:,}")
print(f"Output tokens: {json_output_tokens:,}")
print(f"Cost: ${json_total_cost:.6f}")

# Savings
cost_savings = individual_total_cost - json_total_cost
percentage_savings = (cost_savings / individual_total_cost) * 100

print(f"\n🎯 SAVINGS:")
print(f"Cost reduction: ${cost_savings:.6f}")
print(f"Percentage savings: {percentage_savings:.1f}%")

# Extrapolate to full dataset
total_rows = len(train)
total_features = 6
total_individual_requests = total_rows * total_features
total_batch_requests = (total_rows + 10 - 1) // 10  # 10 rows per batch

individual_full_cost = (individual_total_input * total_individual_requests / 18) * 0.4 / 1_000_000 + (individual_total_output * total_individual_requests / 18) * 1.6 / 1_000_000
batch_full_cost = (json_input_tokens * total_batch_requests / 3) * 0.4 / 1_000_000 + (json_output_tokens * total_batch_requests / 3) * 1.6 / 1_000_000

print(f"\n🌍 FULL DATASET EXTRAPOLATION:")
print(f"Individual approach: ${individual_full_cost:.2f}")
print(f"JSON batch approach: ${batch_full_cost:.2f}")
print(f"Total savings: ${individual_full_cost - batch_full_cost:.2f}")
print(f"Percentage savings: {((individual_full_cost - batch_full_cost) / individual_full_cost) * 100:.1f}%")