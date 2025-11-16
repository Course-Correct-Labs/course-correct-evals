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
    analyze_sequence,
    analyze_dataframe_sequences,
    detect_compression,
    calculate_persistence_rate,
    calculate_intervention_effectiveness,
    classify_responses_dataframe,
    detect_contamination_dataframe,
    analyze_contamination_spread,
    calculate_refusal_rates,
    analyze_echo_metrics,
    detect_threshold_crossing,
    calculate_convergence_statistics,
    analyze_metric_trajectories,
)


class CrossStudyAnalysis:
    """
    Main analysis class for the CCL Reasoning Stability Observatory.

    This class coordinates data loading, metric calculation, and
    cross-study comparisons across all CCL empirical studies.
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
        fail_on_missing: bool = False
    ) -> Dict[str, bool]:
        """
        Load data from all available studies.

        Args:
            mirror_loop_path: Optional path to mirror loop data
            confabulation_path: Optional path to confabulation data
            violation_state_path: Optional path to violation state data
            echo_chamber_path: Optional path to echo chamber data
            fail_on_missing: If True, raise error if any data missing

        Returns:
            Dictionary indicating which studies loaded successfully
        """
        print("=" * 60)
        print("LOADING CCL STUDIES")
        print("=" * 60)

        # Try to load each study
        # Mirror Loop
        print("\n[1/4] Mirror Loop Study...")
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
        print("\n[2/4] Recursive Confabulation Study...")
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
        print("\n[3/4] Violation State Study...")
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

        # Echo Chamber
        print("\n[4/4] Echo Chamber Study...")
        importer = EchoChamberImporter(data_path=echo_chamber_path)
        try:
            self.echo_chamber_data = importer.load_data()
            if self.echo_chamber_data is not None:
                self._data_loaded['echo_chamber'] = True
                self._data_sources['echo_chamber'] = importer.data_source
                print("✓ Echo Chamber loaded successfully")
            else:
                if fail_on_missing:
                    raise ValueError("Echo Chamber data not available")
                print("✗ Echo Chamber not available")
        except FileNotFoundError as e:
            if fail_on_missing:
                raise
            warnings.warn(f"Echo Chamber data not found: {e}")
            print("✗ Echo Chamber not available")
        except Exception as e:
            if fail_on_missing:
                raise
            warnings.warn(f"Error loading Echo Chamber: {e}")
            print(f"✗ Echo Chamber error: {e}")

        print("\n" + "=" * 60)
        loaded_count = sum(self._data_loaded.values())
        print(f"LOADED {loaded_count}/4 STUDIES")
        print("=" * 60)

        return self._data_loaded.copy()

    def analyze_mirror_loop(self) -> Dict[str, Any]:
        """Analyze Mirror Loop study data."""
        if not self._data_loaded['mirror_loop']:
            return {"error": "Mirror Loop data not loaded"}

        print("\nAnalyzing Mirror Loop study...")

        # Calculate ΔI metrics for all sequences
        sequence_analysis = analyze_dataframe_sequences(
            self.mirror_loop_data,
            use_embeddings=False,  # Optional, can be slow
            collapse_threshold=0.05
        )

        # Overall statistics
        total_sequences = len(sequence_analysis)
        collapsed_sequences = sequence_analysis['collapse_detected'].sum()
        collapse_rate = collapsed_sequences / total_sequences if total_sequences > 0 else 0.0

        # Model-level statistics
        if 'model' in self.mirror_loop_data.columns:
            model_stats = []
            for model in self.mirror_loop_data['model'].unique():
                model_df = self.mirror_loop_data[self.mirror_loop_data['model'] == model]
                model_analysis = analyze_dataframe_sequences(model_df)

                model_total = len(model_analysis)
                model_collapsed = model_analysis['collapse_detected'].sum()

                model_stats.append({
                    'model': model,
                    'total_sequences': model_total,
                    'collapsed_sequences': int(model_collapsed),
                    'collapse_rate': float(model_collapsed / model_total) if model_total > 0 else 0.0,
                    'mean_delta_i': float(model_analysis['delta_i_edit_mean'].mean()),
                })

            model_stats_df = pd.DataFrame(model_stats)
        else:
            model_stats_df = None

        self.mirror_loop_analysis = {
            'total_sequences': total_sequences,
            'collapsed_sequences': int(collapsed_sequences),
            'collapse_rate': float(collapse_rate),
            'mean_delta_i_overall': float(sequence_analysis['delta_i_edit_mean'].mean()),
            'sequence_analysis': sequence_analysis,
            'model_statistics': model_stats_df,
        }

        print(f"✓ Analyzed {total_sequences} sequences, {collapsed_sequences} collapsed ({collapse_rate:.1%})")

        return self.mirror_loop_analysis

    def analyze_confabulation(self) -> Dict[str, Any]:
        """Analyze Recursive Confabulation study data.

        Note: RC data is now aggregate (per-model summary), not per-conversation.
        This method returns summary statistics from the published aggregate data.
        """
        if not self._data_loaded['confabulation']:
            return {"error": "Confabulation data not loaded"}

        print("\nAnalyzing Recursive Confabulation study...")

        # Check if this is aggregate data
        if 'confab_persistence_rate' in self.confabulation_data.columns:
            # Aggregate mode: summarize from pre-computed metrics
            mean_persistence = self.confabulation_data['confab_persistence_rate'].mean()
            mean_confab = self.confabulation_data.get('confab_rate', pd.Series([None])).mean()

            self.confabulation_analysis = {
                'data_type': 'aggregate',
                'num_models': len(self.confabulation_data),
                'models': sorted(self.confabulation_data['model'].tolist()),
                'mean_persistence_rate': float(mean_persistence),
                'mean_confab_rate': float(mean_confab) if pd.notna(mean_confab) else None,
                'persistence_by_model': self.confabulation_data[['model', 'confab_persistence_rate']].to_dict('records'),
            }

            print(f"✓ Aggregate data: {len(self.confabulation_data)} models")
            print(f"  Mean persistence rate: {mean_persistence:.1%}")

        else:
            # Legacy mode: per-conversation data (if someone provides it locally)
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
        """Analyze Violation State study data."""
        if not self._data_loaded['violation_state']:
            return {"error": "Violation State data not loaded"}

        print("\nAnalyzing Violation State study...")

        # Classify responses if not already done
        if 'response_type' not in self.violation_state_data.columns:
            self.violation_state_data = classify_responses_dataframe(
                self.violation_state_data,
                content_col='content'
            )

        # Detect contamination if not already done
        if 'contamination_detected' not in self.violation_state_data.columns:
            self.violation_state_data = detect_contamination_dataframe(
                self.violation_state_data,
                content_col='content'
            )

        # Calculate refusal rates
        refusal_stats = calculate_refusal_rates(
            self.violation_state_data,
            group_by_col='model' if 'model' in self.violation_state_data.columns else None
        )

        # Contamination spread analysis
        contamination_stats = analyze_contamination_spread(self.violation_state_data)

        self.violation_state_analysis = {
            'refusal_statistics': refusal_stats,
            'contamination_statistics': contamination_stats,
            'total_conversations': self.violation_state_data['conversation_id'].nunique(),
            'total_turns': len(self.violation_state_data),
        }

        print(f"✓ Contamination rate: {contamination_stats['contamination_rate']:.1%}")

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
            # RC data is aggregate (per-model summary), not per-conversation
            all_models.update(self.confabulation_data['model'].unique())

        if self._data_loaded['violation_state'] and 'model' in self.violation_state_data.columns:
            all_models.update(self.violation_state_data['model'].unique())

        if self._data_loaded['echo_chamber'] and 'model' in self.echo_chamber_data.columns:
            all_models.update(self.echo_chamber_data['model'].unique())

        # Filter to requested models
        if models:
            all_models = {m for m in all_models if m in models}

        # Build leaderboard row for each model
        for model in sorted(all_models):
            row = {'model': model}

            # Mirror Loop: collapse rate
            if self._data_loaded['mirror_loop'] and 'model' in self.mirror_loop_data.columns:
                model_data = self.mirror_loop_data[self.mirror_loop_data['model'] == model]
                if len(model_data) > 0:
                    model_seq_analysis = analyze_dataframe_sequences(model_data)
                    collapse_rate = model_seq_analysis['collapse_detected'].mean()
                    row['mirror_collapse_rate'] = float(collapse_rate)
                else:
                    row['mirror_collapse_rate'] = None
            else:
                row['mirror_collapse_rate'] = None

            # Confabulation: persistence rate
            # Note: RC data is aggregate (per-model summary from summary_by_model_arm.csv)
            if self._data_loaded['confabulation'] and 'model' in self.confabulation_data.columns:
                # Check if this is aggregate data (has confab_persistence_rate column)
                if 'confab_persistence_rate' in self.confabulation_data.columns:
                    # Aggregate mode: look up metric directly
                    model_row = self.confabulation_data[self.confabulation_data['model'] == model]
                    if len(model_row) > 0:
                        row['confab_persistence_rate'] = float(model_row['confab_persistence_rate'].iloc[0])
                    else:
                        row['confab_persistence_rate'] = None
                else:
                    # Legacy mode: per-conversation data (if someone provides it locally)
                    model_data = self.confabulation_data[self.confabulation_data['model'] == model]
                    if len(model_data) > 0:
                        pers_stats = calculate_persistence_rate(model_data)
                        row['confab_persistence_rate'] = float(pers_stats['overall']['persistence_rate'])
                    else:
                        row['confab_persistence_rate'] = None
            else:
                row['confab_persistence_rate'] = None

            # Violation State: contamination rate
            if self._data_loaded['violation_state'] and 'model' in self.violation_state_data.columns:
                model_data = self.violation_state_data[self.violation_state_data['model'] == model]
                if len(model_data) > 0:
                    if 'contamination_detected' not in model_data.columns:
                        model_data = detect_contamination_dataframe(model_data)
                    contam_rate = model_data['contamination_detected'].mean()
                    row['violation_contamination_rate'] = float(contam_rate)
                else:
                    row['violation_contamination_rate'] = None
            else:
                row['violation_contamination_rate'] = None

            # Echo Chamber: mean GR/SRI
            if self._data_loaded['echo_chamber'] and 'model' in self.echo_chamber_data.columns:
                model_data = self.echo_chamber_data[self.echo_chamber_data['model'] == model]
                if len(model_data) > 0:
                    if 'GR' in model_data.columns:
                        row['echo_mean_GR'] = float(model_data['GR'].mean())
                    if 'SRI' in model_data.columns:
                        row['echo_mean_SRI'] = float(model_data['SRI'].mean())
                else:
                    row['echo_mean_GR'] = None
                    row['echo_mean_SRI'] = None
            else:
                row['echo_mean_GR'] = None
                row['echo_mean_SRI'] = None

            leaderboard_data.append(row)

        leaderboard_df = pd.DataFrame(leaderboard_data)

        print(f"✓ Leaderboard created with {len(leaderboard_df)} models")

        return leaderboard_df

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded studies and analyses."""
        summary = {
            'studies_loaded': self._data_loaded,
            'total_studies_loaded': sum(self._data_loaded.values()),
        }

        # Add analysis summaries if available
        if self.mirror_loop_analysis:
            summary['mirror_loop'] = {
                'total_sequences': self.mirror_loop_analysis['total_sequences'],
                'collapse_rate': self.mirror_loop_analysis['collapse_rate'],
            }

        if self.confabulation_analysis:
            # Handle both aggregate and per-conversation data
            if self.confabulation_analysis.get('data_type') == 'aggregate':
                summary['confabulation'] = {
                    'data_type': 'aggregate',
                    'num_models': self.confabulation_analysis['num_models'],
                    'mean_persistence_rate': self.confabulation_analysis['mean_persistence_rate'],
                }
            else:
                summary['confabulation'] = {
                    'data_type': 'per_conversation',
                    'total_conversations': self.confabulation_analysis.get('total_conversations', 0),
                    'persistence_rate': self.confabulation_analysis.get('persistence_statistics', {}).get('overall', {}).get('persistence_rate', 0),
                }

        if self.violation_state_analysis:
            summary['violation_state'] = {
                'total_conversations': self.violation_state_analysis['total_conversations'],
                'contamination_rate': self.violation_state_analysis['contamination_statistics']['contamination_rate'],
            }

        if self.echo_chamber_analysis:
            summary['echo_chamber'] = {
                'total_simulations': self.echo_chamber_analysis['total_simulations'],
            }

        return summary

    def get_data_source_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get data source status for all studies.

        Useful for debugging and transparency about where data was loaded from.

        Returns:
            Dictionary mapping study names to their load status and source.
            Each study has:
                - 'loaded': bool indicating if data is available
                - 'source': string describing where data came from
                  (e.g., 'explicit_path:...', 'local:...', 'remote:...', 'not_loaded')
        """
        summary = {}

        for study_name in ['mirror_loop', 'confabulation', 'violation_state', 'echo_chamber']:
            summary[study_name] = {
                'loaded': self._data_loaded[study_name],
                'source': self._data_sources[study_name] if self._data_sources[study_name] else 'not_loaded'
            }

        return summary
