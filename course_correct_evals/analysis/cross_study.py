"""
Cross-Study Analysis

Unified analysis system for all CCL empirical studies.
This is the main analytical engine of the Observatory.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import warnings

from ..importers import (
    MirrorLoopImporter,
    ConfabulationImporter,
    ViolationStateImporter,
    EchoChamberImporter,
)

from ..metrics import (
    analyze_mirror_loop_plateau,
    compute_grounding_rebound,
    detect_compression,
    calculate_persistence_rate,
    calculate_intervention_effectiveness,
    analyze_violation_state_structured,
    analyze_echo_metrics,
    detect_threshold_crossing,
    calculate_convergence_statistics,
    analyze_metric_trajectories,
)


class CrossStudyAnalysis:
    """
    Main analysis class for the CCL Reasoning Stability Observatory.

    This class coordinates data loading, metric calculation, and
    cross-study comparisons across the three canonical CCL empirical
    studies: Mirror Loop, Recursive Confabulation, and Violation State.

    Echo Chamber Zero is retained here as noncanonical/opt-in provenance
    support (see load_all_studies(include_echo_chamber=...) and
    analyze_echo_chamber()). It is an independent theoretical/systemic
    research project, not a peer behavioral model evaluation, and it is
    never included in the canonical study count, leaderboard, dashboard,
    or default exports.
    """

    def __init__(self):
        """Initialize the observatory."""
        self.mirror_loop_data: Optional[pd.DataFrame] = None
        self.confabulation_data: Optional[pd.DataFrame] = None
        self.violation_state_data: Optional[pd.DataFrame] = None
        self.echo_chamber_data: Optional[pd.DataFrame] = None

        self.mirror_loop_analysis: Optional[Dict[str, Any]] = None
        self.confabulation_analysis: Optional[Dict[str, Any]] = None
        self.violation_state_analysis: Optional[Dict[str, Any]] = None
        self.echo_chamber_analysis: Optional[Dict[str, Any]] = None

        self._data_loaded = {
            'mirror_loop': False,
            'confabulation': False,
            'violation_state': False,
            'echo_chamber': False,
        }

        # Track where data was loaded from
        self._data_sources = {
            'mirror_loop': None,
            'confabulation': None,
            'violation_state': None,
            'echo_chamber': None,
        }

    def load_all_studies(
        self,
        mirror_loop_path: Optional[str] = None,
        confabulation_path: Optional[str] = None,
        violation_state_path: Optional[str] = None,
        echo_chamber_path: Optional[str] = None,
        fail_on_missing: bool = False,
        include_echo_chamber: bool = False
    ) -> Dict[str, bool]:
        """
        Load data for the three canonical Observatory studies.

        Args:
            mirror_loop_path: Optional path to mirror loop data
            confabulation_path: Optional path to confabulation data
            violation_state_path: Optional path to violation state data
            echo_chamber_path: Optional path to echo chamber data (only
                used if include_echo_chamber=True)
            fail_on_missing: If True, raise error if any canonical data missing
            include_echo_chamber: If True, also attempts to load Echo Chamber
                Zero data as an OPTIONAL, NONCANONICAL addition. Echo Chamber
                is never counted toward the canonical study total, never
                affects canonical success/failure, and never appears in
                model-comparison output regardless of this flag.

        Returns:
            Dictionary indicating which studies loaded successfully. Always
            contains 'echo_chamber' for backward compatibility, defaulting
            to False unless include_echo_chamber=True and loading succeeds.
        """
        print("=" * 60)
        print("LOADING CCL STUDIES (canonical: Mirror Loop, Recursive Confabulation, Violation State)")
        print("=" * 60)

        # Try to load each canonical study
        # Mirror Loop
        print("\n[1/3] Mirror Loop Study...")
        importer = MirrorLoopImporter(data_path=mirror_loop_path)
        try:
            self.mirror_loop_data = importer.load_data()
            if self.mirror_loop_data is not None:
                self._data_loaded['mirror_loop'] = True
                self._data_sources['mirror_loop'] = importer.data_source
                print("✓ Mirror Loop loaded successfully")
            else:
                if fail_on_missing:
                    raise ValueError("Mirror Loop data not available")
                print("✗ Mirror Loop not available")
        except Exception as e:
            if fail_on_missing:
                raise
            warnings.warn(f"Error loading Mirror Loop: {e}")
            print(f"✗ Mirror Loop error: {e}")

        # Confabulation
        print("\n[2/3] Recursive Confabulation Study...")
        importer = ConfabulationImporter(data_path=confabulation_path)
        try:
            self.confabulation_data = importer.load_data()
            if self.confabulation_data is not None:
                self._data_loaded['confabulation'] = True
                self._data_sources['confabulation'] = importer.data_source
                print("✓ Confabulation loaded successfully")
            else:
                if fail_on_missing:
                    raise ValueError("Confabulation data not available")
                print("✗ Confabulation not available")
        except Exception as e:
            if fail_on_missing:
                raise
            warnings.warn(f"Error loading Confabulation: {e}")
            print(f"✗ Confabulation error: {e}")

        # Violation State
        print("\n[3/3] Violation State Study...")
        importer = ViolationStateImporter(data_path=violation_state_path)
        try:
            self.violation_state_data = importer.load_data()
            if self.violation_state_data is not None:
                self._data_loaded['violation_state'] = True
                self._data_sources['violation_state'] = importer.data_source
                print("✓ Violation State loaded successfully")
            else:
                if fail_on_missing:
                    raise ValueError("Violation State data not available")
                print("✗ Violation State not available")
        except Exception as e:
            if fail_on_missing:
                raise
            warnings.warn(f"Error loading Violation State: {e}")
            print(f"✗ Violation State error: {e}")

        # Echo Chamber Zero — OPTIONAL, NONCANONICAL. Not part of the
        # Observatory's canonical study set; only attempted if the caller
        # explicitly opts in. Never counted in the canonical study total.
        if include_echo_chamber:
            print("\n[optional/noncanonical] Echo Chamber Zero...")
            importer = EchoChamberImporter(data_path=echo_chamber_path)
            try:
                self.echo_chamber_data = importer.load_data()
                if self.echo_chamber_data is not None:
                    self._data_loaded['echo_chamber'] = True
                    self._data_sources['echo_chamber'] = importer.data_source
                    print("✓ Echo Chamber Zero loaded (noncanonical, opt-in)")
                else:
                    if fail_on_missing:
                        raise ValueError("Echo Chamber data not available")
                    print("✗ Echo Chamber Zero not available")
            except FileNotFoundError as e:
                if fail_on_missing:
                    raise
                warnings.warn(f"Echo Chamber data not found: {e}")
                print("✗ Echo Chamber Zero not available")
            except Exception as e:
                if fail_on_missing:
                    raise
                warnings.warn(f"Error loading Echo Chamber: {e}")
                print(f"✗ Echo Chamber Zero error: {e}")

        print("\n" + "=" * 60)
        canonical_studies = ('mirror_loop', 'confabulation', 'violation_state')
        loaded_count = sum(self._data_loaded[s] for s in canonical_studies)
        print(f"LOADED {loaded_count}/3 CANONICAL STUDIES")
        if include_echo_chamber:
            echo_status = "loaded" if self._data_loaded['echo_chamber'] else "not loaded"
            print(f"Echo Chamber Zero (noncanonical, opt-in): {echo_status}")
        print("=" * 60)

        return self._data_loaded.copy()

    def analyze_mirror_loop(self) -> Dict[str, Any]:
        """
        Analyze Mirror Loop study data using the manuscript-defined
        rolling-three-step plateau statistic, computed PER SEQUENCE first
        and aggregated afterward -- never a rolling average on a
        pooled/averaged group trajectory.

        Canonical measurement source: the released `edit_change` column
        (NOT a recomputation from response text via delta_i_edit_distance
        -- a direct numerical check established that recomputation does
        not reproduce the released measurement).

        PROVENANCE: the rolling-three-step plateau definition, the primary
        tau=0.05 threshold, the tau=0.02 sensitivity threshold, the
        per-sequence-then-aggregate interpretation, and the
        grounding-rebound definition all come from the Mirror Loop
        manuscript, supplied outside the cloned mirror-loop GitHub
        repository (that repository explicitly does not include the
        manuscript). Only the released numerical measurements themselves
        are verified directly from data/mirror_loop_results_all.csv.

        NOTE on novelty: the manuscript's n-gram novelty finding is
        trajectory-based/per-iteration (a pooled curve decaying toward
        near-zero by iterations 6-7), not a single dataset-wide mean. An
        overall mean of the released `ngram_novelty` column was previously
        exposed here as 'mean_ngram_novelty_overall'; it has been removed
        because it was an Observatory-invented aggregate with no
        manuscript-defined referent, not a reproduction of a reported
        statistic. The released `ngram_novelty` column itself is untouched
        and remains available (e.g. via self.mirror_loop_data or the
        generic, noncanonical ngram_novelty() utility's own analysis) --
        only the unsupported canonical scalar was removed. Implementing the
        manuscript's actual trajectory-based novelty finding is a separate,
        later scope decision.

        Returns a dict with:
          - 'plateau': the PRIMARY canonical result at tau=0.05 (drives the
            leaderboard, canonical panel, and default notebook/report view).
          - 'plateau_sensitivity_tau_0_02': an explicitly SECONDARY
            sensitivity view. Never feeds the leaderboard and never
            replaces/mutates the primary result.
          - 'grounding_rebound': a DISTINCT manuscript finding (pooled
            iteration-2-vs-4 comparison within the grounded condition) --
            not derived from, and not folded into, the plateau structure.

        There is no `mirror_collapse_rate` or "collapse" construct here --
        that was Observatory terminology attached to a drifted single-value
        first-crossing detector that did not reproduce the manuscript's
        statistic; it has been removed, not aliased.
        """
        if not self._data_loaded['mirror_loop']:
            return {"error": "Mirror Loop data not loaded"}

        print("\nAnalyzing Mirror Loop study...")

        df = self.mirror_loop_data

        if 'edit_change' not in df.columns:
            raise ValueError(
                "Mirror Loop canonical analysis requires the released 'edit_change' "
                "column; it is not present in this data. Canonical analysis does not "
                "fall back to recomputing ΔI from response text."
            )

        total_sequences = int(df['sequence_id'].nunique())
        group_cols = ['model', 'condition'] if 'condition' in df.columns else ['model']

        # PRIMARY canonical plateau result (tau=0.05).
        plateau_primary = analyze_mirror_loop_plateau(df, tau=0.05, window=3, group_cols=group_cols)

        # Explicitly SECONDARY sensitivity view (tau=0.02).
        plateau_sensitivity = analyze_mirror_loop_plateau(df, tau=0.02, window=3, group_cols=group_cols)

        # DISTINCT finding: grounding rebound (pooled, iteration 2 vs 4,
        # grounded condition only). Only computed if a 'condition' column
        # with a 'grounded' value actually exists in this data.
        if 'condition' in df.columns and (df['condition'] == 'grounded').any():
            grounding_rebound = compute_grounding_rebound(df, condition='grounded')
        else:
            grounding_rebound = None

        mean_delta_i_overall = float(df['edit_change'].mean())

        self.mirror_loop_analysis = {
            'total_sequences': total_sequences,
            'mean_delta_i_overall': mean_delta_i_overall,
            'plateau': plateau_primary,
            'plateau_sensitivity_tau_0_02': plateau_sensitivity,
            'grounding_rebound': grounding_rebound,
        }

        print(f"✓ Analyzed {total_sequences} sequences (released edit_change; "
              f"manuscript rolling-3-step plateau, tau=0.05 primary)")
        for group_key, stats in plateau_primary['group_summary'].items():
            label = ' / '.join(str(g) for g in group_key)
            iqr_txt = ""
            if stats['median_plateau_iteration'] is not None:
                iqr_txt = (f", median iter {stats['median_plateau_iteration']:.0f} "
                           f"(IQR {stats['plateau_iteration_iqr'][0]:.0f}-"
                           f"{stats['plateau_iteration_iqr'][1]:.0f})")
            print(f"  {label}: {stats['n_plateaued']}/{stats['n_sequences']} plateaued "
                  f"({stats['plateau_rate']:.1%}){iqr_txt}")
        if grounding_rebound is not None:
            print(f"  Grounding rebound ({grounding_rebound['condition']}, "
                  f"iter {grounding_rebound['iteration_from']}->{grounding_rebound['iteration_to']}): "
                  f"{grounding_rebound['pct_increase']:.1f}% [manuscript-defined, distinct from plateau]")

        return self.mirror_loop_analysis

    def analyze_confabulation(self) -> Dict[str, Any]:
        """Analyze Recursive Confabulation study data.

        Canonical path: the released model x arm table (preserved
        unmodified by ConfabulationImporter -- one row per (model, arm),
        3 models x 4 arms). No arm or model collapsing happens anywhere in
        this method. Two distinct, separately-labeled views are derived
        from that SAME table:

          - pooled_intervention_comparison: the manuscript's N-weighted
            pooled comparison across baseline/fact_table/belief_audit
            (grounding_pilot deliberately excluded -- the source study's
            own pooled comparison excludes it too).
          - grounding_confabulation_heterogeneity: the model-specific
            grounding_pilot finding, kept separate because the source
            study reports this arm's effect as model-heterogeneous, not
            poolable.

        No per-model, across-arm scalar (an "average persistence rate for
        this model") is computed or exposed anywhere in this method.

        ----------------------------------------------------------------
        METRIC IDENTITY GUARD -- do not conflate these two outcome
        variables (source-verified against data/summary_by_model_arm.csv
        in the recursive-confabulation repository):

          - persist_rate: whether a fabrication PERSISTS after
            correction/turns. Used for pooled_intervention_comparison
            (baseline/fact_table/belief_audit -- the intervention
            "backfire" finding).
          - confab_rate: whether the model CONFABULATES at all,
            initially. Used for grounding_confabulation_heterogeneity --
            this is the field the source study's "grounding reduced
            confabulation" finding (README.md, RC_publication_pack.md)
            actually reports.

        Grounding's effect on PERSISTENCE specifically is a real released
        measurement too (visible via model_arm_table filtered to
        arm == 'grounding_pilot', and via the
        confab_persist_rate_grounding_pilot leaderboard column) -- but it
        is NOT the manuscript's "grounding reduced confabulation" finding
        and must never be presented as such. Using persist_rate for that
        narrative would misstate it: GPT-4o-mini's persist_rate under
        grounding (0.5) is HIGHER than its own baseline (0.3) --
        persistence got worse, not better, under grounding for that
        model. Do not substitute one field for the other in either
        direction.
        ----------------------------------------------------------------
        """
        if not self._data_loaded['confabulation']:
            return {"error": "Confabulation data not loaded"}

        print("\nAnalyzing Recursive Confabulation study...")

        # Canonical model x arm table (the released summary_by_model_arm.csv
        # schema, as returned unmodified by ConfabulationImporter).
        if 'arm' in self.confabulation_data.columns and 'persist_rate' in self.confabulation_data.columns:
            table = self.confabulation_data

            models = sorted(table['model'].unique().tolist())
            arms = sorted(table['arm'].unique().tolist())

            # Manuscript-defined pooled intervention comparison
            # (baseline, fact_table, belief_audit only), N-weighted
            # across models: sum(persist_rate * n) / sum(n), per arm.
            pooled_arms = ['baseline', 'fact_table', 'belief_audit']
            pooled_intervention_comparison = {}
            for arm in pooled_arms:
                arm_rows = table[table['arm'] == arm]
                if len(arm_rows) == 0:
                    continue
                total_n = int(arm_rows['n'].sum())
                weighted_persisting = (arm_rows['persist_rate'] * arm_rows['n']).sum()
                pooled_intervention_comparison[arm] = {
                    'n': total_n,
                    'persist_rate': float(weighted_persisting / total_n) if total_n > 0 else 0.0,
                }

            # Model-specific grounding_pilot CONFABULATION finding -- NOT
            # pooled. Uses confab_rate (see METRIC IDENTITY GUARD above),
            # which is the field the source study's "Grounding reduced
            # confabulation for GPT-4o mini only" finding (README.md,
            # RC_publication_pack.md) actually reports. This is
            # deliberately NOT persist_rate.
            grounding_rows = table[table['arm'] == 'grounding_pilot']
            grounding_confabulation_heterogeneity = {
                row['model']: {'n': int(row['n']), 'confab_rate': float(row['confab_rate'])}
                for _, row in grounding_rows.iterrows()
            }

            self.confabulation_analysis = {
                'data_type': 'model_arm_table',
                'model_arm_table': table,
                'models': models,
                'arms': arms,
                'total_conversations': int(table['n'].sum()),
                'pooled_intervention_comparison': pooled_intervention_comparison,
                'grounding_confabulation_heterogeneity': grounding_confabulation_heterogeneity,
            }

            print(f"✓ Model x arm table: {len(table)} rows ({len(models)} models x {len(arms)} arms)")
            for arm, stats in pooled_intervention_comparison.items():
                print(f"  Pooled {arm}: {stats['persist_rate']:.2%} (N={stats['n']}, N-weighted across models, persist_rate)")
            print("  Grounding confabulation (model-specific, not pooled, confab_rate):")
            for model, stats in grounding_confabulation_heterogeneity.items():
                print(f"    {model}: {stats['confab_rate']:.2%} (N={stats['n']})")
            print("  (Grounding's effect on persistence specifically is also a released")
            print("   measurement -- see model_arm_table filtered to arm=='grounding_pilot' --")
            print("   but it is NOT the manuscript's confabulation finding above.)")

        else:
            # Legacy mode: per-conversation data (if someone provides it locally).
            # Unreachable via ConfabulationImporter -> load_all_studies(), since
            # that importer now always returns the model x arm table; retained
            # unmodified for defensive compatibility with manually-injected data.
            persistence_stats = calculate_persistence_rate(self.confabulation_data)

            # Intervention effectiveness
            if 'intervention_arm' in self.confabulation_data.columns:
                intervention_stats = calculate_intervention_effectiveness(
                    self.confabulation_data,
                    baseline_name='baseline'
                )
            else:
                intervention_stats = None

            self.confabulation_analysis = {
                'data_type': 'per_conversation',
                'persistence_statistics': persistence_stats,
                'intervention_effectiveness': intervention_stats,
                'total_conversations': self.confabulation_data['conversation_id'].nunique(),
                'total_turns': len(self.confabulation_data),
            }

            overall_persistence = persistence_stats['overall']['persistence_rate']
            print(f"✓ Overall persistence rate: {overall_persistence:.1%}")

        return self.confabulation_analysis

    def analyze_violation_state(self) -> Dict[str, Any]:
        """
        Analyze Violation State study data using its structured
        experimental fields (canonical) — see
        analyze_violation_state_structured() for the Rule C collapsing
        methodology and the raw/published provenance-layer distinction.

        Does NOT canonically use the generic text-pattern classifier
        (classify_response_type / detect_contamination and friends,
        in metrics/session_contamination.py) — that classifier's phrase
        library doesn't match this study's actual refusal phrasing and
        remains retained only as noncanonical/legacy functionality.
        """
        if not self._data_loaded['violation_state']:
            return {"error": "Violation State data not loaded"}

        print("\nAnalyzing Violation State study...")

        structured = analyze_violation_state_structured(self.violation_state_data)

        self.violation_state_analysis = {
            'structured': structured,
            'total_conversations': self.violation_state_data['conversation_id'].nunique(),
            'total_turns': len(self.violation_state_data),
        }

        raw = structured['raw_structured_outcomes']
        published = structured['published_aggregate']

        for cond in published:
            r = raw.get(cond, {})
            p = published[cond]
            print(f"  [{cond}] RAW structured outcomes (N={r.get('n', 0)}): {r.get('counts', {})}")
            print(f"  [{cond}] PUBLISHED/HISTORICAL aggregate: {p['refused']}/{p['n']} refused "
                  f"({p['refusal_rate']:.2%}) — historical rate-limit-as-refusal convention")

        return self.violation_state_analysis

    def analyze_echo_chamber(self) -> Dict[str, Any]:
        """Analyze Echo Chamber study data."""
        if not self._data_loaded['echo_chamber']:
            return {"error": "Echo Chamber data not loaded"}

        print("\nAnalyzing Echo Chamber study...")

        # Analyze precomputed metrics
        echo_stats = analyze_echo_metrics(self.echo_chamber_data)

        # Convergence statistics
        convergence_stats = calculate_convergence_statistics(self.echo_chamber_data)

        # Trajectory analysis
        trajectories = analyze_metric_trajectories(self.echo_chamber_data)

        # Threshold crossings (example thresholds)
        threshold_crossings = {}
        for metric, threshold in [('GR', 0.7), ('SRI', 0.8), ('RE', 0.5)]:
            if metric in self.echo_chamber_data.columns:
                crossings = detect_threshold_crossing(
                    self.echo_chamber_data,
                    metric_col=metric,
                    threshold=threshold,
                    direction='above'
                )
                threshold_crossings[metric] = crossings

        self.echo_chamber_analysis = {
            'echo_statistics': echo_stats,
            'convergence_statistics': convergence_stats,
            'trajectories': trajectories,
            'threshold_crossings': threshold_crossings,
            'total_simulations': self.echo_chamber_data['simulation_id'].nunique(),
            'total_steps': len(self.echo_chamber_data),
        }

        print(f"✓ Analyzed {echo_stats['total_simulations']} simulations")

        return self.echo_chamber_analysis

    def create_leaderboard(self, models: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create cross-study leaderboard comparing models.

        Args:
            models: Optional list of models to include. If None, includes all.

        Returns:
            DataFrame with model performance across studies
        """
        print("\nCreating cross-study leaderboard...")

        leaderboard_data = []

        # Collect all models
        all_models = set()

        if self._data_loaded['mirror_loop'] and 'model' in self.mirror_loop_data.columns:
            all_models.update(self.mirror_loop_data['model'].unique())

        if self._data_loaded['confabulation'] and 'model' in self.confabulation_data.columns:
            # RC data is the released model x arm table (one row per
            # (model, arm)); each model may appear in multiple rows.
            all_models.update(self.confabulation_data['model'].unique())

        # Violation State is intentionally excluded from the leaderboard:
        # it is a single production system/interface study (not a
        # cross-model comparison), and its canonical result is a
        # contaminated-vs-control structured breakdown, not a per-model
        # score. This exclusion is unconditional, even if Violation State
        # data someday contains a 'model' column.

        # Echo Chamber Zero is intentionally excluded from the leaderboard:
        # it is a systemic/theoretical percolation framework, not a peer
        # behavioral model evaluation, and has no model-comparison meaning.
        # This exclusion is unconditional, independent of whether Echo
        # Chamber data was separately opted into via load_all_studies().

        # Filter to requested models
        if models:
            all_models = {m for m in all_models if m in models}

        # Build leaderboard row for each model
        for model in sorted(all_models):
            row = {'model': model}

            # Mirror Loop: manuscript-defined plateau rate (tau=0.05,
            # PRIMARY canonical threshold only -- the tau=0.02 sensitivity
            # view never feeds the leaderboard), condition-explicit. No
            # legacy 'mirror_collapse_rate' -- removed, not aliased.
            row['mirror_plateau_rate_grounded'] = None
            row['mirror_plateau_rate_ungrounded'] = None
            if (self._data_loaded['mirror_loop']
                    and 'model' in self.mirror_loop_data.columns
                    and 'edit_change' in self.mirror_loop_data.columns
                    and 'condition' in self.mirror_loop_data.columns):
                model_data = self.mirror_loop_data[self.mirror_loop_data['model'] == model]
                if len(model_data) > 0:
                    model_plateau = analyze_mirror_loop_plateau(
                        model_data, tau=0.05, window=3, group_cols=['condition']
                    )
                    for (condition,), stats in model_plateau['group_summary'].items():
                        if condition in ('grounded', 'ungrounded'):
                            row[f'mirror_plateau_rate_{condition}'] = stats['plateau_rate']

            # Confabulation: four arm-explicit columns, each a direct
            # released (model, arm).persist_rate measurement -- no
            # averaging across arms, no Observatory-derived model score.
            # These are measurements under distinct experimental
            # conditions, not four interchangeable global model-quality
            # scores or independent ranking metrics.
            #
            # NOTE (metric identity): these four columns are PERSISTENCE
            # measurements (persist_rate). They are not the manuscript's
            # "grounding reduced confabulation" finding, which is based on
            # confab_rate -- see grounding_confabulation_heterogeneity in
            # analyze_confabulation(). confab_persist_rate_grounding_pilot
            # is a real released measurement in its own right; it must not
            # be read as evidence of grounding's confabulation-reduction
            # effect.
            confab_arm_columns = {
                'confab_persist_rate_baseline': 'baseline',
                'confab_persist_rate_fact_table': 'fact_table',
                'confab_persist_rate_belief_audit': 'belief_audit',
                'confab_persist_rate_grounding_pilot': 'grounding_pilot',
            }
            if (self._data_loaded['confabulation']
                    and 'model' in self.confabulation_data.columns
                    and 'arm' in self.confabulation_data.columns
                    and 'persist_rate' in self.confabulation_data.columns):
                model_rows = self.confabulation_data[self.confabulation_data['model'] == model]
                for col_name, arm_name in confab_arm_columns.items():
                    arm_row = model_rows[model_rows['arm'] == arm_name]
                    if len(arm_row) > 0:
                        row[col_name] = float(arm_row['persist_rate'].iloc[0])
                    else:
                        row[col_name] = None
            else:
                for col_name in confab_arm_columns:
                    row[col_name] = None

            leaderboard_data.append(row)

        leaderboard_df = pd.DataFrame(leaderboard_data)

        print(f"✓ Leaderboard created with {len(leaderboard_df)} models")

        return leaderboard_df

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the three canonical studies and their analyses.

        Note: 'studies_loaded' retains the dormant 'echo_chamber' key for
        backward compatibility, but 'total_studies_loaded' counts only the
        three canonical studies (Mirror Loop, Recursive Confabulation,
        Violation State) regardless of whether Echo Chamber was separately
        opted into.
        """
        canonical_studies = ('mirror_loop', 'confabulation', 'violation_state')
        summary = {
            'studies_loaded': self._data_loaded,
            'total_studies_loaded': sum(self._data_loaded[s] for s in canonical_studies),
        }

        # Add analysis summaries if available
        if self.mirror_loop_analysis:
            # Canonical: no 'collapse_rate' scalar. 'plateau' is the
            # PRIMARY (tau=0.05) manuscript-defined per-sequence-then-
            # aggregated result; 'plateau_sensitivity_tau_0_02' is
            # explicitly secondary; 'grounding_rebound' is a distinct
            # finding, not derived from the plateau structure.
            def _stringify_group_keys(group_summary):
                # group_summary keys are tuples (e.g. (model, condition)),
                # which json.dump cannot serialize as object keys.
                return {
                    ' / '.join(str(g) for g in group_key): stats
                    for group_key, stats in group_summary.items()
                }

            summary['mirror_loop'] = {
                'total_sequences': self.mirror_loop_analysis['total_sequences'],
                'mean_delta_i_overall': self.mirror_loop_analysis['mean_delta_i_overall'],
                'plateau_group_summary': _stringify_group_keys(
                    self.mirror_loop_analysis['plateau']['group_summary']),
                'plateau_sensitivity_tau_0_02_group_summary': _stringify_group_keys(
                    self.mirror_loop_analysis['plateau_sensitivity_tau_0_02']['group_summary']),
                'grounding_rebound': self.mirror_loop_analysis['grounding_rebound'],
            }

        if self.confabulation_analysis:
            if self.confabulation_analysis.get('data_type') == 'model_arm_table':
                # Canonical: no across-arm persistence scalar is exposed.
                # 'pooled_intervention_comparison' (persist_rate) is the
                # manuscript's N-weighted baseline/fact_table/belief_audit
                # comparison; 'grounding_confabulation_heterogeneity'
                # (confab_rate) is the model-specific grounding_pilot
                # finding, kept separate. See the METRIC IDENTITY GUARD in
                # analyze_confabulation() -- these two use different
                # outcome variables and must not be substituted.
                summary['confabulation'] = {
                    'data_type': 'model_arm_table',
                    'models': self.confabulation_analysis['models'],
                    'arms': self.confabulation_analysis['arms'],
                    'total_conversations': self.confabulation_analysis.get('total_conversations'),
                    'pooled_intervention_comparison': self.confabulation_analysis['pooled_intervention_comparison'],
                    'grounding_confabulation_heterogeneity': self.confabulation_analysis['grounding_confabulation_heterogeneity'],
                }
            else:
                # Legacy per-conversation mode; unreachable via the normal
                # importer path (see analyze_confabulation()).
                summary['confabulation'] = {
                    'data_type': 'per_conversation',
                    'total_conversations': self.confabulation_analysis.get('total_conversations', 0),
                    'persistence_rate': self.confabulation_analysis.get('persistence_statistics', {}).get('overall', {}).get('persistence_rate', 0),
                }

        if self.violation_state_analysis:
            structured = self.violation_state_analysis['structured']
            summary['violation_state'] = {
                'total_conversations': self.violation_state_analysis['total_conversations'],
                'raw_structured_outcomes': structured['raw_structured_outcomes'],
                'published_aggregate': structured['published_aggregate'],
            }

        if self.echo_chamber_analysis:
            summary['echo_chamber'] = {
                'total_simulations': self.echo_chamber_analysis['total_simulations'],
            }

        return summary

    def get_data_source_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get data source status for the three canonical studies.

        Useful for debugging and transparency about where data was loaded from.

        Echo Chamber Zero is intentionally excluded from this canonical
        summary (it is not a canonical study). Its status, when opted into
        via load_all_studies(include_echo_chamber=True), can be inspected
        directly via self._data_loaded['echo_chamber'] / self._data_sources['echo_chamber'].

        Returns:
            Dictionary mapping study names to their load status and source.
            Each study has:
                - 'loaded': bool indicating if data is available
                - 'source': string describing where data came from
                  (e.g., 'explicit_path:...', 'local:...', 'remote:...', 'not_loaded')
        """
        summary = {}

        for study_name in ['mirror_loop', 'confabulation', 'violation_state']:
            summary[study_name] = {
                'loaded': self._data_loaded[study_name],
                'source': self._data_sources[study_name] if self._data_sources[study_name] else 'not_loaded'
            }

        return summary
