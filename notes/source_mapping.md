# Data Source Mapping for CCL Studies

This document maps the expected data sources from each CCL empirical study repository.

## Study Repositories (READ ONLY)

The following repositories contain the source data for this observatory:

1. **mirror-loop** - Mirror Loop study
2. **recursive-confabulation** - Recursive Confabulation study
3. **violation-state** - Violation State study
4. **echo-chamber-zero** - Echo Chamber study

---

## 1. Mirror Loop Study

### Repository Structure
- **Expected location**: `../mirror-loop/` or configurable path
- **Primary data file**: `mirror_loop_results_all.csv`
- **Alternative locations**:
  - `results/mirror_loop_results_all.csv`
  - `data/mirror_loop_results_all.csv`

### Expected Columns
- `iteration` or `turn` - iteration number in the self-critique sequence
- `model` - model identifier (e.g., "gpt-4", "claude-3-opus")
- `provider` - API provider
- `prompt` or `input_text` - input to the model
- `response` or `output_text` - model output
- `timestamp` - when the run occurred
- `sequence_id` or `run_id` - identifier for the full sequence

### Metrics to Calculate
- **ΔI (Information Change)**:
  - Edit distance between consecutive responses
  - Embedding distance (optional)
  - N-gram novelty
- **Collapse Detection**: Sequences where ΔI → 0
- **Convergence Point**: First iteration where ΔI < threshold

---

## 2. Recursive Confabulation Study

### Repository Structure
- **Expected location**: `../recursive-confabulation/` or configurable path
- **Primary data files**:
  - IRR-validated conversation CSVs
  - `confabulation_results.csv` or similar
  - `annotated_conversations/` directory

### Expected Columns
- `conversation_id` - unique conversation identifier
- `turn_number` - turn within conversation
- `message` or `content` - message text
- `role` - "user" or "assistant"
- `fabrication_present` - boolean or label
- `intervention_arm` - experimental condition (e.g., "baseline", "correction", "clarification")
- `model` - model identifier
- `timestamp`

### Metrics to Calculate
- **Persistence Rate**: % of fabrications that persist after correction
- **Fabrication Propagation**: how fabrications spread across turns
- **Intervention Effectiveness**: comparison across experimental arms

---

## 3. Violation State Study

### Repository Structure
- **Expected location**: `../violation-state/` or configurable path
- **Primary data files**:
  - `parsed_turns.csv`
  - `violation_results.csv`

### Expected Columns
- `conversation_id` - unique identifier
- `turn_number` - turn number
- `message` or `content` - message text
- `violation_type` - type of violation requested
- `model` - model identifier
- `response_type` - classification: "compliance", "refusal", "rate_limit", "error", etc.
- `contamination_detected` - boolean
- `timestamp`

### Metrics to Calculate
- **Contamination Rate**: % of sessions with violation state leakage
- **Refusal Rate**: % of violation requests refused
- **Response Classification**: distribution of response types
- **Session Contamination**: detection of violation state in non-violation turns

---

## 4. Echo Chamber Study

### Repository Structure
- **Expected location**: `../echo-chamber-zero/` or configurable path
- **Primary data file**: `simulation_results.csv`

### Expected Columns
**IMPORTANT**: This study uses **precomputed metrics** in the CSV:

- `simulation_id` - unique simulation identifier
- `step` or `iteration` - step in the simulation
- `agent_id` - which agent in the echo chamber
- `belief_state` or `message` - agent's belief/output
- **Precomputed Metrics** (USE THESE DIRECTLY):
  - `GR` - Group Radicalization metric
  - `SRI` - Self-Reinforcement Index
  - `RE` - Reasoning Entropy
- `model` - model identifier
- `initial_prompt` - starting prompt
- `convergence_reached` - boolean

### Metrics to Calculate
- **Primary**: Use precomputed GR, SRI, RE from CSV
- **Threshold Analysis**: Identify when GR/SRI cross critical thresholds
- **Convergence Detection**: Steps to convergence
- **Optional**: NetworkX reconstruction for visualization (not for metric computation)

### Network Analysis (Optional)
- If needed for future data, can reconstruct:
  - Agent interaction graph
  - Belief propagation network
- But for existing data: **USE PRECOMPUTED METRICS ONLY**

---

## Data Loading Strategy

### Priority Order for File Discovery
For each study, the importer will search in this order:

1. Explicitly provided path (via constructor argument)
2. Environment variable (e.g., `MIRROR_LOOP_DATA_PATH`)
3. Adjacent sibling directories: `../<repo-name>/`
4. Standard subdirectories: `data/`, `results/`, `output/`
5. Current working directory

### Error Handling
- If data file not found: raise clear error with expected locations
- If columns missing: attempt case-insensitive matching and common variants
- If still missing: raise error listing required vs. found columns
- Log warnings for deprecated column names

### Validation Rules
- All importers must return non-empty DataFrames
- Required columns must be present (with fuzzy matching)
- Data types should be validated and coerced when safe
- Duplicates should be checked and reported
- Missing values in critical columns should be flagged

---

## Example Importer Usage

```python
from course_correct_evals import MirrorLoopImporter

# Auto-discover data location
importer = MirrorLoopImporter()
df = importer.load_data()

# Or specify path explicitly
importer = MirrorLoopImporter(data_path="../mirror-loop/results/mirror_loop_results_all.csv")
df = importer.load_data()

# Validate and get info
print(importer.get_data_info())
```

---

## Notes and Assumptions

### Mirror Loop
- **Assumption**: "collapse" is defined as ΔI dropping below 5% of initial ΔI
- **TODO**: Verify threshold with original study paper

### Confabulation
- **Assumption**: "persistence" means fabrication remains in next turn after correction
- **TODO**: Confirm IRR methodology and annotation schema

### Violation State
- **Assumption**: "contamination" is regex-based detection of violation keywords in clean sessions
- **TODO**: Verify contamination detection algorithm

### Echo Chamber
- **Assumption**: Precomputed GR/SRI are ground truth
- **TODO**: Document the computation formulas used in original study

---

## Changelog

- **2025-11-16**: Initial mapping document created
- Expected updates as data sources are validated
