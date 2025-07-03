# Water Pump Challenge - GPT Feature Engineering Project

## Project Status: PAUSED - Ready for 1% Test

### 🎯 What We're Doing
Creating **6 custom GPT features** for water pump failure prediction using GPT-4.1-mini. We've optimized from 11.9 hours down to ~24 minutes using JSON vertical batching (30x speedup).

### 🚨 Current Issue
Hit rate limits during full dataset run. Need to test with 1% sample (594 rows) to validate approach and get realistic timing estimates.

### 📁 Key Files

#### **Ready to Run:**
- `test_1_percent.py` - **START HERE** - Test with 594 rows (~2-3 min, $0.01)
- `run_gpt_full_final.py` - Full dataset script (use after 1% test succeeds)

#### **Core Implementation:**
- `gpt_feature_engineering_json.py` - Main GPT processing with JSON vertical batching
- `model_with_gpt_features.py` - Integrate GPT features with ML models

#### **Original Work:**
- `Kaggle_Competition_32.ipynb` - Your winning notebook (80.88% accuracy)
- `improved_water_pump_model.py` - My improved baseline (81.77% accuracy)

#### **Analysis & Documentation:**
- `validate_test_results.py` - Quality assessment of GPT features
- `cost_comparison.py` - Cost analysis (individual vs batching)
- `performance_comparison.md` - Detailed comparison document

### 🤖 GPT Features Created (6 total)
1. **funder_org_type**: Government/International/NGO/Religious/Private (1-5)
2. **installer_quality**: Data quality of installer name (1-5)  
3. **location_type**: School/Religious/Health/Government/Village (1-5)
4. **scheme_pattern**: Naming pattern quality (1-5)
5. **text_language**: English/Mixed/Local language patterns (1-5)
6. **coordination**: Funder-installer alignment (1-5)

### 📊 Optimization Journey
- **Original**: 356,400 requests → 11.9 hours → $14.97
- **Horizontal batching**: 17,820 requests → 40 minutes → $0.24
- **JSON vertical batching**: 11,880 requests → 24 minutes → $0.95
- **Rate limited reality**: Much slower due to API limits

### 🎯 Next Steps When You Wake Up

#### **Immediate (Start Here):**
```bash
source .venv/bin/activate
export OPENAI_API_KEY=$OPENAI_API_KEY
python test_1_percent.py
```

#### **If 1% Test Succeeds:**
1. Use observed rate to estimate full dataset time
2. If reasonable (<2 hours), run `python run_gpt_full_final.py`
3. If too slow, reduce concurrency in `gpt_feature_engineering_json.py`

#### **If 1% Test Fails:**
1. Check rate limits in OpenAI dashboard
2. Reduce `max_concurrent` from 50 to 10-20
3. Add delays between requests
4. Consider smaller batch sizes (3 instead of 5)

### 💡 What the GPT Features Should Achieve
- **Correlation with pump status**: Better coordination = better functionality
- **Data quality indicators**: Poor installer names = more failures  
- **Institutional effects**: Schools/hospitals = better maintenance
- **Language patterns**: English vs local may indicate development level

### 🔧 Technical Details
- **Model**: GPT-4.1-mini (better quality than nano)
- **Method**: JSON vertical batching (all 6 features per request)
- **Batch size**: 5 rows per request (prevents JSON truncation)
- **Error handling**: Exponential backoff, JSON recovery, progress saving
- **Concurrency**: 50 (may need to reduce based on your rate limits)

### 📈 Expected Results
If successful, GPT features should improve model accuracy beyond the current 81.77%. The installer reliability and coordination features showed strongest correlation (+0.142) in tests.

### 🐛 Known Issues
- **Rate limiting**: Your OpenAI tier may have lower limits than assumed
- **JSON truncation**: Fixed with increased max_tokens and recovery logic
- **Progress saving**: Works but only saves every 20% completion

### 💰 Costs
- **1% test**: ~$0.01
- **Full dataset**: ~$0.95 (94% cheaper than individual requests)

---

**TL;DR**: Run `python test_1_percent.py` first to validate the approach, then proceed with full dataset if timing looks reasonable.