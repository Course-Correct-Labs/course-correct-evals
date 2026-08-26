# Course Correct Labs Reasoning Stability Observatory

**Unified Analysis System for CCL Empirical Studies**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Course-Correct-Labs/course-correct-evals/blob/main/notebooks/CCL_Reasoning_Stability_Observatory.ipynb)

## Overview

The **CCL Reasoning Stability Observatory** is a comprehensive evaluation toolkit that synthesizes findings across three canonical Course Correct Labs empirical studies:

1. **Mirror Loop** - Whether iterative self-critique plateaus (rolling-3-step ΔI decay below threshold)
2. **Recursive Confabulation** - Fabrication persistence across conversational turns
3. **Violation State** - Contamination from refusal states

Echo Chamber Zero, an independent Course Correct Labs theoretical/systemic research
project (synthetic epistemic drift modeled as percolation on a provenance graph), is
**not** part of this canonical evaluation set. Its previously implemented Observatory
integration is retained as noncanonical/opt-in — see
[Noncanonical / Experimental / Provenance Components](#noncanonical--experimental--provenance-components).

This toolkit provides:
- 🔄 **Unified data importers** for all CCL studies (READ ONLY)
- 📊 **Comprehensive metrics** for reasoning stability analysis
- 🎨 **Publication-quality visualizations**
- 🏆 **Cross-study leaderboard** for model comparison
- 📓 **Flagship Jupyter notebook** with end-to-end analysis

**Key Features:**
- ⚡ Runs in <10 minutes using existing data
- 💰 $0 cost (uses precomputed results)
- 🔒 READ ONLY - never modifies source study repos
- 📦 Modular, extensible architecture

---

## Installation

### Requirements

- Python 3.9+
- pip or conda

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/Course-Correct-Labs/course-correct-evals.git
cd course-correct-evals

# Install dependencies
pip install -e .
```

### Development Installation

```bash
# Install with dev dependencies
pip install -e ".[dev]"
```

### Optional: Live Runner (requires API keys)

```bash
# For running optional live demos (costs money!)
pip install -e ".[live-runner]"
```

---

## Quick Start

### 1. Run the Flagship Notebook

**Option A: Google Colab (No Installation Required)**

Click the badge at the top of this README or visit:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Course-Correct-Labs/course-correct-evals/blob/main/notebooks/CCL_Reasoning_Stability_Observatory.ipynb)

**Option B: Local Jupyter**

```bash
jupyter notebook notebooks/CCL_Reasoning_Stability_Observatory.ipynb
```

The notebook provides:
- Complete cross-study analysis (three canonical studies)
- Model leaderboard
- Three-panel comparison figure
- Deep dives into each study
- Optional live demo (disabled by default)
- Echo Chamber Zero appendix (noncanonical, disabled by default)

### 2. Use the Python API

```python
from course_correct_evals import CrossStudyAnalysis

# Initialize Observatory
observatory = CrossStudyAnalysis()

# Load all available studies
observatory.load_all_studies()

# Create cross-study leaderboard
leaderboard = observatory.create_leaderboard()
print(leaderboard)

# Analyze each canonical study
ml_analysis = observatory.analyze_mirror_loop()
conf_analysis = observatory.analyze_confabulation()
vs_analysis = observatory.analyze_violation_state()

# Generate visualizations
from course_correct_evals.analysis.viz import plot_four_panel_comparison
fig = plot_four_panel_comparison(observatory)
```

### 3. Individual Study Importers

```python
from course_correct_evals import MirrorLoopImporter

# Load Mirror Loop data
importer = MirrorLoopImporter(data_path="path/to/data.csv")
df = importer.load_data()

# Get data info
info = importer.get_data_info()
print(info)
```

---

## Project Structure

```
course-correct-evals/
├── pyproject.toml              # Project configuration & dependencies
├── README.md                   # This file
├── notes/
│   └── source_mapping.md       # Data source documentation
├── course_correct_evals/
│   ├── __init__.py
│   ├── importers/              # Data importers (READ ONLY)
│   │   ├── mirror_loop_importer.py
│   │   ├── confab_importer.py
│   │   ├── violation_importer.py
│   │   └── echo_importer.py
│   ├── metrics/                # Metric calculation modules
│   │   ├── information_change.py
│   │   ├── semantic_compression.py
│   │   ├── persistence.py
│   │   ├── session_contamination.py
│   │   └── percolation.py
│   ├── analysis/               # Cross-study analysis
│   │   ├── cross_study.py
│   │   └── viz.py
│   ├── runners/                # Optional live demo runners
│   │   └── mirror_loop_runner.py
│   └── reports/                # Export utilities
│       └── export.py
├── notebooks/
│   └── CCL_Reasoning_Stability_Observatory.ipynb
└── tests/                      # Unit tests
    └── ...
```

---

## How It Works

### Data Importers

The Observatory uses **READ ONLY** importers that:
- Auto-discover data files from study repositories
- Normalize column names across studies
- Validate data integrity
- Provide helpful error messages

**Example:**

```python
from course_correct_evals import MirrorLoopImporter

# Auto-discover data
importer = MirrorLoopImporter()
df = importer.load_data()

# Or specify path explicitly
importer = MirrorLoopImporter(data_path="/path/to/mirror_loop_results.csv")
df = importer.load_data()
```

### Metrics Modules

Each study has associated metric calculation functions:

#### Information Change / Plateau (Mirror Loop, canonical)
```python
observatory = CrossStudyAnalysis()
observatory.load_all_studies()
ml = observatory.analyze_mirror_loop()

# PRIMARY: manuscript-defined plateau rate (tau=0.05, rolling-3-step,
# per-sequence detection then aggregated by model x condition)
ml['plateau']['group_summary']

# SECONDARY sensitivity view (tau=0.02) -- never feeds the leaderboard,
# never alters the primary result above
ml['plateau_sensitivity_tau_0_02']['group_summary']

# DISTINCT finding: grounding rebound (grounded-condition pooled ΔI,
# iteration 2 vs. 4) -- not derived from or combined with plateau
ml['grounding_rebound']
```

The underlying functions (`analyze_mirror_loop_plateau`, `detect_sequence_plateau`,
`compute_grounding_rebound` in `course_correct_evals/metrics/information_change.py`)
are also available directly. See
[Mirror Loop's generic information-change utilities](#mirror-loops-generic-information-change-utilities)
for the noncanonical/legacy `delta_i_edit_distance` / `analyze_sequence` recomputation
utilities and why they are not used canonically.

#### Recursive Confabulation (model x arm, canonical)
```python
observatory = CrossStudyAnalysis()
observatory.load_all_studies()
conf = observatory.analyze_confabulation()

# The released model x arm table (12 rows: 3 models x 4 intervention arms),
# preserved unmodified -- no arm or model collapsing.
conf['model_arm_table']

# Manuscript's N-weighted pooled intervention comparison (persist_rate;
# baseline/fact_table/belief_audit only; grounding_pilot excluded, matching
# the source study)
conf['pooled_intervention_comparison']

# Model-specific grounding_pilot CONFABULATION finding -- NOT pooled, and a
# DIFFERENT outcome variable (confab_rate) than the pooled comparison above
# (persist_rate). Do not substitute one for the other -- see the METRIC
# IDENTITY GUARD in CrossStudyAnalysis.analyze_confabulation().
conf['grounding_confabulation_heterogeneity']
```
Persistence is intervention-dependent, not a single per-model score: there is no
canonical `confab_persistence_rate` scalar anywhere in this Observatory. Recursive
Confabulation involves two distinct outcome variables that must never be substituted
for one another: `persist_rate` (does a fabrication persist after correction — the
pooled intervention comparison) and `confab_rate` (does the model confabulate at all —
the grounding finding). Grounding's effect on persistence specifically is a real
released measurement too (see `model_arm_table` filtered to `arm == 'grounding_pilot'`,
or the `confab_persist_rate_grounding_pilot` leaderboard column below), but it is not
the manuscript's grounding-confabulation finding. See
[Leaderboard Metrics](#leaderboard-metrics) below and the notebook's Recursive
Confabulation Deep Dive.

(`calculate_persistence_rate`/`calculate_intervention_effectiveness` in
`course_correct_evals/metrics/persistence.py` were built for a raw per-turn schema
this study never actually publishes; they remain only in the legacy/unreachable
per-conversation code path — not part of the canonical Recursive Confabulation
analysis.)

#### Violation State (structured, canonical)
```python
from course_correct_evals.metrics import analyze_violation_state_structured

# Restricted to the four canonical benign image prompts; returns
# raw_structured_outcomes (as-released) and published_aggregate
# (the study's historical rate-limit-as-refusal convention), both
# derived from one shared collapsed representation.
result = analyze_violation_state_structured(violation_state_df)
```

(`classify_response_type`/`detect_contamination` — a generic text-pattern
classifier — are retained as noncanonical/legacy code; see
[Noncanonical / Experimental / Provenance Components](#noncanonical--experimental--provenance-components).
They do not implement the Violation State study's actual construct.)

### Cross-Study Analysis

The `CrossStudyAnalysis` class orchestrates the three canonical studies:

```python
observatory = CrossStudyAnalysis()
observatory.load_all_studies()

# Individual analyses
ml = observatory.analyze_mirror_loop()
conf = observatory.analyze_confabulation()
vs = observatory.analyze_violation_state()

# Cross-study comparison
leaderboard = observatory.create_leaderboard()
summary = observatory.get_summary()
```

### Leaderboard Metrics

The leaderboard compares models across Mirror Loop and Recursive Confabulation. Echo
Chamber Zero and Violation State are intentionally excluded — see
[Noncanonical / Experimental / Provenance Components](#noncanonical--experimental--provenance-components)
for Echo Chamber Zero, and the Violation State section above for why that study (a
single production system/interface, not a cross-model comparison) is never presented
as a peer model-comparison metric, even if its data someday contains a `model` column.

| Metric | Description | Better |
|--------|-------------|--------|
| `mirror_plateau_rate_grounded` | Manuscript-defined plateau rate (τ=0.05, rolling-3-step, per-sequence-then-aggregated), grounded condition | See note below |
| `mirror_plateau_rate_ungrounded` | Same statistic, ungrounded condition | See note below |
| `confab_persist_rate_baseline` | Released persistence rate, baseline arm | Lower |
| `confab_persist_rate_fact_table` | Released persistence rate, fact_table arm | Lower |
| `confab_persist_rate_belief_audit` | Released persistence rate, belief_audit arm | Lower |
| `confab_persist_rate_grounding_pilot` | Released persistence rate, grounding_pilot arm | Lower |

`mirror_plateau_rate_*` reports the PRIMARY (τ=0.05) manuscript-defined plateau rate
only — a τ=0.02 sensitivity view exists separately (`plateau_sensitivity_tau_0_02` in
`analyze_mirror_loop()`'s output) and never feeds this leaderboard. There is no single
`mirror_collapse_rate` scalar, and this repository does not assert a default sort
direction ("lower is better") for plateau rate, since that normative interpretation is
the manuscript's to make, not the Observatory's — see the notebook's Mirror Loop Deep
Dive for the full per-group results, including the median/IQR plateau iteration and the
distinct grounding-rebound finding.

The four Recursive Confabulation columns are **measurements under distinct
experimental conditions**, not four interchangeable global model-quality scores and
not an independent ranking metric — each is a direct, unmodified released
`(model, arm).persist_rate` value. **These are persistence measurements, not
grounding-confabulation efficacy** — `confab_persist_rate_grounding_pilot` in
particular must not be read as the manuscript's "grounding reduced confabulation"
finding (that finding uses `confab_rate`; see
[grounding_confabulation_heterogeneity](#recursive-confabulation-model-x-arm-canonical)
above). No averaging across arms is performed anywhere in
the leaderboard; the purpose of including Recursive Confabulation here is to expose
intervention-dependent model behavior, not to manufacture a single ordering of models.

---

## Data Sources

The Observatory expects data from the following CCL study repositories:

1. **mirror-loop** - Mirror Loop study data
2. **recursive-confabulation** - Confabulation study data
3. **violation-state** - Violation State study data

(Echo Chamber Zero data is supported on an opt-in basis only — see
[Noncanonical / Experimental / Provenance Components](#noncanonical--experimental--provenance-components).)

### Data Discovery

Importers search for data in this order:

1. Explicitly provided path (constructor argument)
2. Environment variable (e.g., `MIRROR_LOOP_DATA_PATH`)
3. Adjacent sibling directories (`../mirror-loop/`, etc.)
4. Standard subdirectories (`data/`, `results/`, etc.)

### Environment Variables

Set these to specify data locations:

```bash
export MIRROR_LOOP_DATA_PATH="/path/to/mirror_loop_results.csv"
export CONFABULATION_DATA_PATH="/path/to/confabulation_results.csv"
export VIOLATION_STATE_DATA_PATH="/path/to/parsed_turns.csv"
```

(`ECHO_CHAMBER_DATA_PATH` is also supported, for the noncanonical/opt-in Echo Chamber
Zero importer — see [Noncanonical / Experimental / Provenance Components](#noncanonical--experimental--provenance-components).)

See `notes/source_mapping.md` for detailed data source documentation.

---

## Optional: Live Demo Runner

The repository includes an **optional** Mirror Loop runner for generating new data.

**⚠️ WARNING: This costs money and requires API keys!**

### Setup

```bash
pip install -e ".[live-runner]"
export OPENAI_API_KEY="your-key-here"
```

### Usage

```python
from course_correct_evals.runners.mirror_loop_runner import (
    run_mirror_loop_demo,
    analyze_live_demo
)

# Enable the runner (disabled by default)
import course_correct_evals.runners.mirror_loop_runner as runner
runner.RUN_LIVE_DEMO = True

# Run demo (costs ~$0.01-0.05)
results = run_mirror_loop_demo(
    prompt="Explain recursion in programming.",
    model="gpt-3.5-turbo",
    max_iterations=10
)

# Analyze results
analysis = analyze_live_demo(results)
print(f"Collapse detected: {analysis['collapse_detected']}")
```

**Note:** The live demo is disabled by default in the notebook. Set `RUN_LIVE_DEMO = True` to enable.

---

## Exporting Results

### CSV Export

```python
from course_correct_evals.reports import export_csv_results

# Export all results to CSV
files = export_csv_results(observatory, output_dir='results')
```

Exports (canonical, default):
- `leaderboard.csv` - Model comparison
- `mirror_loop_analysis.csv` - Sequence-level results
- `confabulation_model_arm.csv` - Released model x arm table (source measurements, one row per (model, arm))
- `confabulation_pooled_intervention_comparison.csv` - Manuscript-defined N-weighted pooled comparison (baseline/fact_table/belief_audit; provenance-labeled, kept separate from released measurements)
- `violation_state_analysis.csv` - Refusal/contamination rates
- `summary.json` - Overall summary

### Report Generation

```python
from course_correct_evals.reports import export_pdf_report

# Generate markdown report
report_path = export_pdf_report(observatory, output_path='results/ccl_observatory_report.pdf')
```

Convert to PDF:
```bash
pandoc ccl_observatory_report.md -o ccl_observatory_report.pdf
```

---

## Visualization

### Three-Panel Comparison

```python
from course_correct_evals.analysis.viz import plot_four_panel_comparison

fig = plot_four_panel_comparison(
    observatory,
    figsize=(18, 6),
    save_path='comparison.png'
)
```

(Function name retained as `plot_four_panel_comparison` for backward compatibility; it draws the three canonical panels below.)

Generates:
- Panel 1: Mirror Loop manuscript-defined plateau rate (τ=0.05, primary) by model × condition — grouped bars, not pooled ΔI trajectories with a single threshold line; per-sequence detection is aggregated after the fact, never a crossing detected on a pooled/averaged curve
- Panel 2: Recursive Confabulation's manuscript pooled intervention comparison (baseline/fact_table/belief_audit, N-weighted across models); grounding_pilot's model-specific heterogeneity is annotated as text, not rendered as a fourth pooled bar — see the notebook deep dive for the full model × arm breakdown
- Panel 3: Violation State raw structured outcomes, contaminated vs control (published/historical aggregate shown as an annotation, not the bar heights)

### Leaderboard Heatmap

```python
from course_correct_evals.analysis.viz import plot_leaderboard

leaderboard = observatory.create_leaderboard()
fig = plot_leaderboard(leaderboard, save_path='leaderboard.png')
```

### Detailed Plots

```python
from course_correct_evals.analysis.viz import plot_mirror_loop_detail

fig = plot_mirror_loop_detail(
    observatory,
    sequence_id='seq_001',
    save_path='detail.png'
)
```

---

## Noncanonical / Experimental / Provenance Components

Echo Chamber Zero is an independent Course Correct Labs theoretical/systemic research
project (synthetic epistemic drift modeled as percolation on a provenance graph). Its
previously implemented Observatory integration is **retained** in this repository, but
it is:

- **noncanonical** — not one of the three studies the Observatory canonically evaluates
- **opt-in** — never loaded, analyzed, or exported by default
- **not a cross-model comparison metric** — never appears in `create_leaderboard()`, regardless of whether it is opted into

What's retained and how to use it:

```python
from course_correct_evals import CrossStudyAnalysis

observatory = CrossStudyAnalysis()

# Explicit opt-in required — Echo Chamber Zero is never loaded by default
observatory.load_all_studies(include_echo_chamber=True)

if observatory._data_loaded['echo_chamber']:
    echo_analysis = observatory.analyze_echo_chamber()  # noncanonical
```

- `EchoChamberImporter` (`course_correct_evals/importers/echo_importer.py`) — standalone data importer; reads `ECHO_CHAMBER_DATA_PATH` or an explicit path
- `analyze_echo_metrics`, `detect_threshold_crossing`, `calculate_convergence_statistics`, `analyze_metric_trajectories` (`course_correct_evals/metrics/percolation.py`) — standalone metric functions
- `_plot_echo_chamber_panel` (`course_correct_evals/analysis/viz.py`) — standalone plotting helper, not called by the canonical `plot_four_panel_comparison()`
- The flagship notebook's **Appendix — Echo Chamber Zero (Decoupled / Non-Canonical)** section — retained for provenance, does not execute during a normal top-to-bottom run (requires explicitly setting `RUN_ECHO_CHAMBER_APPENDIX = True` in that section)

**Correct terminology** (the original integration used incorrect names, since corrected in code and docs): GR = **Groundedness Ratio**, SRI = **Synthetic Recurrence Index**, RE = **Referential Entropy** — not "Group Radicalization," "Self-Reinforcement Index," or "Reasoning Entropy." This is a percolation simulation on a provenance network, not a multi-agent belief-radicalization study.

Echo Chamber Zero's simulation reproducibility is a separate, currently open scientific
question, tracked independently of this Observatory architecture. Nothing in this
repository's retained code validates or resolves it.

See `notes/source_mapping.md` for the historical record of the original integration and its correction.

### Violation State's generic text classifier

Unlike Echo Chamber Zero, **Violation State remains a canonical Observatory study** —
only its *generic text-pattern classifier* is noncanonical, not the study itself. The
canonical analysis (`analyze_violation_state()` / `analyze_violation_state_structured()`)
uses the study's structured experimental fields directly.

`classify_response_type`, `detect_contamination`, `classify_responses_dataframe`,
`detect_contamination_dataframe`, `calculate_refusal_rates`, `analyze_contamination_spread`
(`course_correct_evals/metrics/session_contamination.py`) are retained, unmodified,
noncanonical/legacy code:

- They are a generic keyword/regex classifier, not an implementation of the Violation
  State study's construct — their phrase library does not match this study's actual
  refusal phrasing, and they are not called from any canonical Observatory path.
- They are not presented as implementing the canonical Violation State metric.
- A known, unfixed defect exists in this legacy code (case-sensitivity in
  `REFUSAL_PATTERNS` matching against lowercased text) — left as-is; a dedicated fix is
  a separate, later decision.

### Mirror Loop's generic information-change utilities

Like Violation State, **Mirror Loop remains a canonical Observatory study** — only its
*generic, text-recomputation utilities* are noncanonical, not the study itself. The
canonical analysis (`analyze_mirror_loop()`) uses the study's **released** `edit_change`
column directly for the manuscript-defined rolling-3-step plateau statistic — never a
recomputation from response text.

The released `ngram_novelty` column is a separate, complementary measurement. The
manuscript's novelty finding is trajectory-based/per-iteration (a pooled curve decaying
toward near-zero by iterations 6–7), not a single dataset-wide scalar; the Observatory
does not currently expose any canonical ngram_novelty-based aggregate (an earlier
`mean_ngram_novelty_overall` dataset-wide mean was removed as an unsupported,
Observatory-invented aggregate with no manuscript-defined referent — not a reproduction
of a reported statistic). Implementing the manuscript's actual trajectory-based novelty
finding is a separate, later scope decision. The released `ngram_novelty` column itself
is untouched and available on the raw data (`observatory.mirror_loop_data`).

`delta_i_edit_distance`, `delta_i_embedding`, `ngram_novelty`, `analyze_sequence`,
`analyze_dataframe_sequences` (`course_correct_evals/metrics/information_change.py`)
are retained, unmodified, noncanonical/legacy generic utilities:

- Recomputing ΔI from response text via `delta_i_edit_distance()` does not reproduce
  the study's released `edit_change` column (a direct numerical check found systematic
  ~50% inflation); recomputing novelty via `ngram_novelty()` similarly diverges from the
  released `ngram_novelty` column (up to ~10x in places). Canonical plateau analysis
  therefore never recomputes ΔI from response text — it reads the released `edit_change`
  column. Wherever the Observatory does legitimately consume `ngram_novelty` in the
  future, it must likewise read the released column, not recompute it.
- `analyze_sequence(..., collapse_threshold=...)`'s single-crossing "collapse" detector
  is a different, simpler algorithm than the manuscript's rolling-3-step plateau
  statistic (`detect_sequence_plateau()`) and is not used by canonical analysis. It
  remains available for the optional live-demo path (`runners/mirror_loop_runner.py`,
  disabled by default), where there is no released measurement to read.
- These functions are not presented as implementing the canonical Mirror Loop plateau
  construct.

---

## Testing

Run tests:

```bash
pytest tests/
```

Test individual components:

```bash
pytest tests/test_importers.py
pytest tests/test_metrics.py
pytest tests/test_analysis.py
```

---

## Design Principles

### 1. READ ONLY Operation

- **Never** modifies source study repositories
- **Never** reformats original CSVs
- **Never** commits to other repos

### 2. Fail Gracefully

- Missing studies don't break the system
- Clear error messages for missing data
- Warnings for optional features

### 3. Conservative Budget

- Primary mode: $0 (uses existing data)
- Optional live demo: <$0.05
- No bulk API experiments

### 4. Modularity

- Importers independent of metrics
- Metrics independent of analysis
- Each study can run standalone

### 5. Documentation

- Clear docstrings for all functions
- Type hints throughout
- Comprehensive README and examples

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Citation

If you use this observatory in your research, please cite:

```bibtex
@software{ccl_observatory,
  title = {Course Correct Labs Reasoning Stability Observatory},
  author = {Course Correct Labs},
  year = {2025},
  url = {https://github.com/Course-Correct-Labs/course-correct-evals}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Support

- **Documentation**: See `notes/source_mapping.md`
- **Issues**: [GitHub Issues](https://github.com/Course-Correct-Labs/course-correct-evals/issues)
- **Contact**: labs@coursecorrect.ai

---

## Acknowledgments

This observatory synthesizes data from three canonical CCL empirical studies:
- Mirror Loop study
- Recursive Confabulation study
- Violation State study

Echo Chamber Zero, an independent Course Correct Labs theoretical/systemic research
project, is acknowledged separately — see [Noncanonical / Experimental / Provenance Components](#noncanonical--experimental--provenance-components).

Special thanks to all contributors and researchers involved in these projects.

---

**Course Correct Labs**
*Building tools for safer, more reliable AI systems*
