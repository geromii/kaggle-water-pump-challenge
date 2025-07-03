print("BATCHING EXPLANATION")
print("=" * 50)

print("\n🔄 HORIZONTAL BATCHING (what I'm doing):")
print("Multiple ROWS, same FEATURE in one request")
print("\nExample request for 'funder_org_type':")
print("""
Classify these funder organizations (1=Government, 2=International, 3=NGO, 4=Religious, 5=Private):

1. Funder: Government Of Tanzania
2. Funder: World Bank  
3. Funder: UNICEF
4. Funder: Private Individual
5. Funder: Danida
... (up to 20 rows)

Output only numbers 1-5, one per line.
""")

print("Expected response:")
print("1\n2\n3\n5\n2\n...")

print("\n📊 ALTERNATIVE: VERTICAL BATCHING")
print("Multiple FEATURES, same ROW in one request")
print("\nExample request for one row:")
print("""
For this water pump, rate all these aspects (1-5 scale):

Data: Funder=Government, Installer=DWE, Ward=Kilifi, Scheme=Water Project...

1. Funder org type (1=Gov, 2=Intl, 3=NGO, 4=Religious, 5=Private): 
2. Installer quality (1=Very poor, 2=Poor, 3=Fair, 4=Good, 5=Excellent):
3. Location type (1=School, 2=Religious, 3=Health, 4=Gov, 5=Village):
4. Scheme pattern (1=Invalid, 2=Code, 3=Location, 4=Descriptive, 5=Full):
5. Language (1=English, 2=Mostly Eng, 3=Mixed, 4=Mostly Local, 5=Local):
6. Name mismatch (1=Perfect, 2=Good, 3=Neutral, 4=Poor, 5=Red flag):

Output 6 numbers, one per line.
""")

print("\n🔍 COMPARISON:")
print("Current (Horizontal):")
print("- 6 features × 2,970 batches = 17,820 requests")
print("- Each request: 20 rows × 1 feature")

print("\nAlternative (Vertical):")  
print("- 1 feature-set × 2,970 batches = 2,970 requests")
print("- Each request: 1 row × 6 features")
print("- Would be 6x faster! (~6 minutes total)")

print("\n⚠️  TRADE-OFFS:")
print("Horizontal (current):")
print("✅ More reliable - focused prompts")
print("✅ Easier to parse responses") 
print("✅ Can handle failures per feature")
print("❌ More API calls")

print("\nVertical (alternative):")
print("✅ Fewer API calls (6x reduction)")
print("✅ Faster overall")
print("❌ More complex prompts")
print("❌ Harder to parse multi-feature responses")
print("❌ If one feature fails, all fail")

print("\n🤔 RECOMMENDATION:")
print("We could try vertical batching for even more speed!")
print("Risk: More complex parsing, but potential 6x speedup (40min → 6min)")