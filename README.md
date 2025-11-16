# Course Correct Labs Reasoning Stability Observatory

**Unified Analysis System for CCL Empirical Studies**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Course-Correct-Labs/course-correct-evals/blob/main/notebooks/CCL_Reasoning_Stability_Observatory.ipynb)

## Overview

The **CCL Reasoning Stability Observatory** is a comprehensive evaluation toolkit that synthesizes findings across four Course Correct Labs empirical studies:

1. **Mirror Loop** - Information collapse in iterative self-critique
2. **Recursive Confabulation** - Fabrication persistence across conversational turns
3. **Violation State** - Contamination from refusal states
4. **Echo Chamber** - Belief percolation in multi-agent systems

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
- Complete cross-study analysis
- Model leaderboard
- Four-panel comparison figure
- Deep dives into each study
- Optional live demo (disabled by default)

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

# Analyze each study
ml_analysis = observatory.analyze_mirror_loop()
conf_analysis = observatory.analyze_confabulation()
vs_analysis = observatory.analyze_violation_state()
echo_analysis = observatory.analyze_echo_chamber()

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

#### Information Change (Mirror Loop)
```python
from course_correct_evals.metrics import delta_i_edit_distance, analyze_sequence

# Calculate ΔI between two texts
di = delta_i_edit_distance(text1, text2)

# Analyze full sequence
analysis = analyze_sequence(texts, collapse_threshold=0.05)
```

#### Fabrication Persistence (Confabulation)
```python
from course_correct_evals.metrics import calculate_persistence_rate

# Calculate persistence rate
stats = calculate_persistence_rate(df)
print(f"Persistence: {stats['overall']['persistence_rate']:.1%}")
```

#### Session Contamination (Violation State)
```python
from course_correct_evals.metrics import classify_response_type, detect_contamination

# Classify response
response_type = classify_response_type(text)

# Detect contamination
result = detect_contamination(text)
```

#### Echo Chamber Metrics
```python
from course_correct_evals.metrics import analyze_echo_metrics

# Analyze precomputed GR, SRI, RE metrics
stats = analyze_echo_metrics(df)
```

### Cross-Study Analysis

The `CrossStudyAnalysis` class orchestrates all studies:

```python
observatory = CrossStudyAnalysis()
observatory.load_all_studies()

# Individual analyses
ml = observatory.analyze_mirror_loop()
conf = observatory.analyze_confabulation()
vs = observatory.analyze_violation_state()
echo = observatory.analyze_echo_chamber()

# Cross-study comparison
leaderboard = observatory.create_leaderboard()
summary = observatory.get_summary()
```

### Leaderboard Metrics

The leaderboard compares models across studies:

| Metric | Description | Better |
|--------|-------------|--------|
| `mirror_collapse_rate` | % of sequences with information collapse | Lower |
| `confab_persistence_rate` | % of fabrications that persist | Lower |
| `violation_contamination_rate` | % of turns contaminated by violation state | Lower |
| `echo_mean_GR` | Mean Group Radicalization score | Lower |
| `echo_mean_SRI` | Mean Self-Reinforcement Index | Lower |

---

## Data Sources

The Observatory expects data from the following CCL study repositories:

1. **mirror-loop** - Mirror Loop study data
2. **recursive-confabulation** - Confabulation study data
3. **violation-state** - Violation State study data
4. **echo-chamber-zero** - Echo Chamber study data

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
export ECHO_CHAMBER_DATA_PATH="/path/to/simulation_results.csv"
```

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

Exports:
- `leaderboard.csv` - Model comparison
- `mirror_loop_analysis.csv` - Sequence-level results
- `confabulation_analysis.json` - Persistence statistics
- `violation_state_analysis.csv` - Refusal/contamination rates
- `echo_chamber_trajectories.csv` - Simulation trajectories
- `summary.json` - Overall summary

### Report Generation

```python
from course_correct_evals.reports import export_pdf_report

# Generate markdown report
report_path = export_pdf_report(observatory)
```

Convert to PDF:
```bash
pandoc ccl_observatory_report.md -o ccl_observatory_report.pdf
```

---

## Visualization

### Four-Panel Comparison

```python
from course_correct_evals.analysis.viz import plot_four_panel_comparison

fig = plot_four_panel_comparison(
    observatory,
    figsize=(16, 12),
    save_path='comparison.png'
)
```

Generates:
- Panel 1: Mirror Loop ΔI trajectories
- Panel 2: Confabulation persistence by intervention
- Panel 3: Violation State response distribution
- Panel 4: Echo Chamber metric evolution

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

This observatory synthesizes data from four CCL empirical studies:
- Mirror Loop study
- Recursive Confabulation study
- Violation State study
- Echo Chamber study

Special thanks to all contributors and researchers involved in these projects.

---

**Course Correct Labs**
*Building tools for safer, more reliable AI systems*
