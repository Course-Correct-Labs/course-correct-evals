# Phase 8 Release-Engineering Checklist

This is a **Phase 8 release-engineering checklist**, not a scientific result and not
part of the Phase 2–6 Observatory integrity-repair/integration work. It tracks
operational items to resolve before final public release, deferred from earlier
phases.

## Items

1. **Notebook execution side effect — RESOLVED.**

   *Found:* the flagship notebook's export cells wrote relative artifacts directly
   into `notebooks/` (`results/`, `three_panel_comparison.png`,
   `observatory_leaderboard.csv`, `ccl_observatory_report.md`). Phase 6 validation
   caught and manually cleaned these after a top-to-bottom execution. A Phase 8
   mechanism trace additionally found: (a) three of the four artifact types were
   already covered by `.gitignore` (`results/`, `*.png`, `*.csv`), but the report's
   `.md` output was not, which is why only that one file showed as untracked; (b)
   in Google Colab specifically, the notebook's setup cell clones the repo but never
   `os.chdir()`s into it, so Colab's default `/content` CWD means these writes do
   **not** land inside the cloned repo there — the repo-pollution symptom was
   local/`nbconvert`-execution-specific, confirmed empirically (CWD always resolves
   to the notebook's own containing directory, regardless of invocation directory);
   (c) `README.md`'s own `export_pdf_report(observatory)` usage example
   independently taught the same unsafe bare-default pattern.

   *Resolved:* the notebook's setup cell now defines a single
   `RESULTS_DIR = "results"` and creates it once (`os.makedirs(RESULTS_DIR,
   exist_ok=True)`); the four write cells (leaderboard CSV, comparison figure,
   `export_csv_results`, `export_pdf_report`) all target paths under `RESULTS_DIR`.
   The README's `export_pdf_report(observatory)` example was updated to explicitly
   pass `output_path='results/ccl_observatory_report.pdf'`. No production
   `course_correct_evals` module or default parameter value was changed — this was
   a notebook- and documentation-only fix. `results/` is already fully gitignored,
   so no `.gitignore` change was required either.

2. **README Colab badge branch target — RESOLVED.**

   *Found:* `README.md` contained two Colab badge URLs hardcoded to the preserved
   long branch (`claude/build-ccl-observatory-01Tm8d1ASgVx3NTHTqaXPEpt`). Phase 7
   proved `main` and the long branch point to identical content (same notebook blob
   SHA) and made `main` the canonical/default branch.

   *Resolved:* both badges now point to
   `blob/main/notebooks/CCL_Reasoning_Stability_Observatory.ipynb`. The preserved
   long branch itself was not touched and its old badge-style URL continues to
   resolve to the same content, since the branch is never deleted.
