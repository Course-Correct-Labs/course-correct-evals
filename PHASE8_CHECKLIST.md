# Phase 8 Release-Engineering Checklist

This is a **Phase 8 release-engineering checklist**, not a scientific result and not
part of the Phase 2–6 Observatory integrity-repair/integration work. It tracks
operational items to resolve before final public release, deferred from earlier
phases.

## Items

1. **Notebook execution side effect.** The flagship notebook's export cells write
   relative artifacts into `notebooks/` (`results/`, `three_panel_comparison.png`,
   `observatory_leaderboard.csv`, `ccl_observatory_report.md`). Phase 6 validation
   caught and cleaned these. Phase 8 must determine whether this behavior should be
   changed before final public release, or explicitly documented if retained.
