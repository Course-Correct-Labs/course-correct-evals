# Importer Debug Notes

## Issue: 0/4 Studies Loaded in Colab

### Root Cause Analysis

Tested all four GitHub fallback URLs and found:

1. **Mirror Loop** ✅
   - URL: `https://raw.githubusercontent.com/Course-Correct-Labs/mirror-loop/main/data/mirror_loop_results_all.csv`
   - Status: 200 OK
   - **No changes needed**

2. **Recursive Confabulation** ❌
   - URL: `https://raw.githubusercontent.com/Course-Correct-Labs/recursive-confabulation/main/data/confabulation_results.csv`
   - Status: 404 NOT FOUND
   - **Problem**: The expected file `confabulation_results.csv` does not exist in the repo
   - Actual files in data/: `harm_irr.csv`, `intervention_effects.csv`, `significance_matrix.csv`, etc.
   - **Analysis**: The recursive-confabulation study published aggregated IRR validation data and statistics, not raw conversation data. The importer expects conversation-level data with columns like `conversation_id`, `turn_number`, `content`, `role`, `fabrication_present` which doesn't exist in the public repo.
   - **Solution**: Mark as unavailable for now; may need to add mock data or contact researchers for raw data

3. **Violation State** ❌ → ✅
   - Old URL: `https://raw.githubusercontent.com/Course-Correct-Labs/violation-state/main/data/parsed_turns.csv`
   - Status: 404 NOT FOUND
   - **Problem**: File is in subdirectory `data/processed/` not `data/`
   - Correct URL: `https://raw.githubusercontent.com/Course-Correct-Labs/violation-state/main/data/processed/parsed_turns.csv`
   - Status: 200 OK
   - **Solution**: Update FALLBACK_URL in ViolationStateImporter

4. **Echo Chamber** ✅
   - URL: `https://raw.githubusercontent.com/Course-Correct-Labs/echo-chamber-zero/main/data/simulation_results.csv`
   - Status: 200 OK
   - **No changes needed**

### Expected Outcome After Fixes

- Mirror Loop: Will load from GitHub ✅
- Confabulation: Will show "not available" (no public data in expected format) ⚠️
- Violation State: Will load from GitHub after URL fix ✅
- Echo Chamber: Will load from GitHub ✅

**Expected result**: 3/4 studies loaded (instead of 0/4)

### Next Steps

1. Fix ViolationStateImporter FALLBACK_URL path
2. Simplify all importers to use bulletproof pattern
3. Add clear "Data source not available" message for Confabulation
4. Test end-to-end in simulated Colab environment
