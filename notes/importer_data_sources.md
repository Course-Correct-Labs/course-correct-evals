# Data Importer Sources

Documentation of data file expectations for each CCL study importer.

## Mirror Loop Study

**Importer Class**: `MirrorLoopImporter`
**Expected File**: `mirror_loop_results_all.csv`
**GitHub Repository**: `Course-Correct-Labs/mirror-loop`
**GitHub Raw URL**: `https://raw.githubusercontent.com/Course-Correct-Labs/mirror-loop/main/data/mirror_loop_results_all.csv`

**Required Columns**:
- `iteration` (or: turn, step, turn_number)
- `model` (or: model_name, model_id)
- `response` (or: output, output_text, content, text)
- `sequence_id` (or: run_id, session_id, conversation_id)

**Optional Columns**:
- `prompt`, `provider`, `timestamp`

**Local Search Paths** (in order):
1. `mirror_loop_results_all.csv`
2. `../mirror-loop/mirror_loop_results_all.csv`
3. `../mirror-loop/results/mirror_loop_results_all.csv`
4. `../mirror-loop/data/mirror_loop_results_all.csv`
5. `data/mirror_loop/mirror_loop_results_all.csv`

---

## Recursive Confabulation Study

**Importer Class**: `ConfabulationImporter`
**Expected File**: `confabulation_results.csv`
**GitHub Repository**: `Course-Correct-Labs/recursive-confabulation`
**GitHub Raw URL**: `https://raw.githubusercontent.com/Course-Correct-Labs/recursive-confabulation/main/data/confabulation_results.csv`

**Required Columns**:
- `conversation_id` (or: session_id, run_id)
- `turn_number` (or: turn, iteration)
- `content` (or: message, text, response)
- `role` (or: speaker)
- `fabrication_present` (or: is_fabrication, fabricated)

**Optional Columns**:
- `intervention_arm`, `model`, `timestamp`, `annotator_confidence`

**Local Search Paths** (in order):
1. `confabulation_results.csv`
2. `../recursive-confabulation/confabulation_results.csv`
3. `../recursive-confabulation/results/confabulation_results.csv`
4. `../recursive-confabulation/data/confabulation_results.csv`
5. `data/confabulation/confabulation_results.csv`

---

## Violation State Study

**Importer Class**: `ViolationStateImporter`
**Expected Files**: `parsed_turns.csv` OR `violation_results.csv`
**GitHub Repository**: `Course-Correct-Labs/violation-state`
**GitHub Raw URL**: `https://raw.githubusercontent.com/Course-Correct-Labs/violation-state/main/data/parsed_turns.csv`

**Required Columns**:
- `conversation_id` (or: session_id, run_id)
- `turn_number` (or: turn, iteration)
- `content` (or: message, text, response)

**Optional Columns**:
- `violation_type`, `model`, `response_type`, `contamination_detected`, `timestamp`, `role`

**Local Search Paths** (in order):
1. `parsed_turns.csv`
2. `violation_results.csv`
3. `../violation-state/parsed_turns.csv`
4. `../violation-state/results/parsed_turns.csv`
5. `../violation-state/data/parsed_turns.csv`
6. `data/violation_state/parsed_turns.csv`

---

## Echo Chamber Zero Study

> **⚠ SUPERSEDED.** Echo Chamber Zero has been decoupled from the canonical
> Observatory evaluation set (canonical studies are now Mirror Loop, Recursive
> Confabulation, and Violation State only). The correct metric definitions are
> **GR = Groundedness Ratio, SRI = Synthetic Recurrence Index, RE = Referential
> Entropy** — this study is a percolation simulation on a provenance network,
> not a multi-agent belief-radicalization study. The importer/column
> documentation below is preserved as a historical record; the importer
> itself is retained as noncanonical/opt-in code (see `README.md`).

**Importer Class**: `EchoChamberImporter`
**Expected File**: `simulation_results.csv`
**GitHub Repository**: `Course-Correct-Labs/echo-chamber-zero`
**GitHub Raw URL**: `https://raw.githubusercontent.com/Course-Correct-Labs/echo-chamber-zero/main/data/simulation_results.csv`

**Required Columns**:
- `simulation_id` (or: run_id, session_id)
- `step` (or: iteration, turn)

**Precomputed Metric Columns** (critical):
- `GR` (or: group_radicalization, gr)
- `SRI` (or: self_reinforcement_index, sri)
- `RE` (or: reasoning_entropy, re)

**Optional Columns**:
- `agent_id`, `belief_state`, `model`, `initial_prompt`, `convergence_reached`, `timestamp`

**Local Search Paths** (in order):
1. `simulation_results.csv`
2. `../echo-chamber-zero/simulation_results.csv`
3. `../echo-chamber-zero/results/simulation_results.csv`
4. `../echo-chamber-zero/data/simulation_results.csv`
5. `data/echo_chamber/simulation_results.csv`

---

## Data Loading Strategy

Each importer follows this priority order:

1. **Explicit path**: If `data_path` argument provided and exists
2. **Environment variable**: Check study-specific env var (e.g., `MIRROR_LOOP_DATA_PATH`)
3. **Local discovery**: Check default search paths (repo-relative)
4. **GitHub fallback**: Fetch from GitHub raw URL (added in Phase 2)
5. **Not available**: Return `None` and log message (no crash)

This ensures:
- Local development works with cloned repos
- Colab automatically fetches from GitHub
- Missing data degrades gracefully
- No manual intervention required
