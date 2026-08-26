"""
Tests for analysis module
"""

import pytest
import pandas as pd
from course_correct_evals import CrossStudyAnalysis


class TestCrossStudyAnalysis:
    """Test cross-study analysis"""

    def test_initialization(self):
        """Test Observatory initialization"""
        observatory = CrossStudyAnalysis()

        assert observatory.mirror_loop_data is None
        assert observatory.confabulation_data is None
        assert observatory.violation_state_data is None
        assert observatory.echo_chamber_data is None

        assert observatory._data_loaded['mirror_loop'] == False
        assert observatory._data_loaded['confabulation'] == False
        assert observatory._data_loaded['violation_state'] == False
        assert observatory._data_loaded['echo_chamber'] == False

    def test_load_all_studies_graceful_failure(self):
        """Test that load_all_studies fails gracefully when data missing"""
        observatory = CrossStudyAnalysis()

        # Should not raise error even though data is missing
        loaded = observatory.load_all_studies(fail_on_missing=False)

        assert isinstance(loaded, dict)
        assert 'mirror_loop' in loaded
        assert 'confabulation' in loaded
        assert 'violation_state' in loaded
        assert 'echo_chamber' in loaded

    def test_get_summary_empty(self):
        """Test getting summary with no data loaded"""
        observatory = CrossStudyAnalysis()
        summary = observatory.get_summary()

        assert summary['total_studies_loaded'] == 0
        assert summary['studies_loaded']['mirror_loop'] == False

    def test_create_leaderboard_empty(self):
        """Test creating leaderboard with no data"""
        observatory = CrossStudyAnalysis()
        leaderboard = observatory.create_leaderboard()

        # Should return empty dataframe
        assert len(leaderboard) == 0


def test_cross_study_analysis_importable():
    """Test that CrossStudyAnalysis can be imported"""
    from course_correct_evals import CrossStudyAnalysis
    assert CrossStudyAnalysis is not None


class TestConfabulationCanonicalAnalysis:
    """
    Test RC canonical model x arm analysis.

    Previously (pre-Phase-4) this class tested "backwards compatibility"
    for an aggregate per-model persistence_rate produced by averaging
    persist_rate across intervention arms -- that averaging was the
    confirmed Phase 4 defect (it erased both the manuscript's pooled
    arm-explicit comparison and its model-specific grounding finding).
    These tests are rewritten to assert the corrected, manuscript-faithful
    canonical structure: no across-arm scalar anywhere.
    """

    def test_model_arm_table_produces_canonical_structure(self):
        """Synthetic model x arm data produces the canonical structure,
        with no across-arm scalar exposed anywhere."""
        import pandas as pd
        from course_correct_evals.analysis import CrossStudyAnalysis

        fake_rc_data = pd.DataFrame({
            'model': ['model_a', 'model_a', 'model_a', 'model_b', 'model_b', 'model_b'],
            'arm': ['baseline', 'fact_table', 'belief_audit', 'baseline', 'fact_table', 'belief_audit'],
            'n': [10, 10, 10, 10, 10, 10],
            'confab_rate': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'persist_rate': [0.5, 0.7, 0.9, 0.6, 0.8, 1.0],
        })

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = fake_rc_data
        observatory._data_loaded['confabulation'] = True

        result = observatory.analyze_confabulation()

        assert result['data_type'] == 'model_arm_table'
        assert set(result['models']) == {'model_a', 'model_b'}
        assert set(result['arms']) == {'baseline', 'fact_table', 'belief_audit'}

        # No across-arm scalar of any name anywhere in the result
        assert 'confab_persistence_rate' not in result
        assert 'mean_persistence_rate' not in result
        assert 'persistence_statistics' not in result
        assert 'descriptive_aggregate_by_model' not in result

        # Pooled comparison: N-weighted (equal n here, so equals the simple mean)
        assert abs(result['pooled_intervention_comparison']['baseline']['persist_rate'] - 0.55) < 1e-9
        assert abs(result['pooled_intervention_comparison']['fact_table']['persist_rate'] - 0.75) < 1e-9
        assert abs(result['pooled_intervention_comparison']['belief_audit']['persist_rate'] - 0.95) < 1e-9

    def test_model_arm_table_has_all_required_keys(self):
        """Canonical result exposes exactly the approved key set."""
        import pandas as pd
        from course_correct_evals.analysis import CrossStudyAnalysis

        fake_rc_data = pd.DataFrame({
            'model': ['model_a', 'model_b'],
            'arm': ['baseline', 'grounding_pilot'],
            'n': [10, 10],
            'confab_rate': [1.0, 0.9],
            'persist_rate': [0.7, 0.85],
        })

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = fake_rc_data
        observatory._data_loaded['confabulation'] = True

        result = observatory.analyze_confabulation()

        required_keys = [
            'data_type', 'model_arm_table', 'models', 'arms',
            'total_conversations', 'pooled_intervention_comparison',
            'grounding_confabulation_heterogeneity',
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # No separate canonical grounding-persistence object -- grounding
        # persistence remains available only via model_arm_table /
        # the leaderboard, not as a named canonical finding.
        assert 'grounding_persistence_by_model' not in result
        assert 'grounding_heterogeneity' not in result  # old (pre-correction) name

        assert isinstance(result['total_conversations'], int)
        assert hasattr(result['model_arm_table'], 'columns')

    def test_model_arm_table_summary_has_required_keys(self):
        """get_summary() surfaces both provenance layers, no legacy scalar."""
        import pandas as pd
        from course_correct_evals.analysis import CrossStudyAnalysis

        fake_rc_data = pd.DataFrame({
            'model': ['model_a', 'model_b'],
            'arm': ['baseline', 'baseline'],
            'n': [10, 10],
            'confab_rate': [1.0, 0.9],
            'persist_rate': [0.7, 0.85],
        })

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = fake_rc_data
        observatory._data_loaded['confabulation'] = True
        observatory.analyze_confabulation()

        summary = observatory.get_summary()

        assert 'confabulation' in summary
        conf = summary['confabulation']

        assert conf['data_type'] == 'model_arm_table'
        assert 'total_conversations' in conf
        assert 'pooled_intervention_comparison' in conf
        assert 'grounding_confabulation_heterogeneity' in conf
        assert 'grounding_heterogeneity' not in conf  # old (pre-correction) name
        assert 'grounding_persistence_by_model' not in conf  # not promoted to a canonical object

        # No legacy across-arm scalar
        assert 'persistence_rate' not in conf
        assert 'mean_persistence_rate' not in conf


class TestReportGeneration:
    """Test report generation with various data states"""

    def test_export_pdf_report_with_model_arm_rc(self):
        """Test that export_pdf_report works with canonical model x arm RC
        data, and clearly separates released/pooled/grounding sections."""
        import pandas as pd
        import tempfile
        import os
        from course_correct_evals.analysis import CrossStudyAnalysis
        from course_correct_evals.reports import export_pdf_report

        # Create fake model x arm RC data (includes grounding_pilot)
        fake_rc_data = pd.DataFrame({
            'model': ['model_a', 'model_a', 'model_b', 'model_b'],
            'arm': ['baseline', 'grounding_pilot', 'baseline', 'grounding_pilot'],
            'n': [10, 10, 10, 10],
            'confab_rate': [1.0, 1.0, 0.9, 0.9],
            'persist_rate': [0.7, 0.9, 0.85, 0.5],
        })

        # Create observatory and inject fake data
        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = fake_rc_data
        observatory._data_loaded['confabulation'] = True

        # Analyze
        observatory.analyze_confabulation()

        # Generate report to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name

        try:
            # Should not crash
            result_path = export_pdf_report(observatory, output_path=temp_path)

            # Verify file was created
            assert os.path.exists(result_path)

            # Verify content clearly separates the three sections and
            # never reports an unlabeled across-arm model average
            with open(result_path, 'r') as f:
                content = f.read()
                assert 'Recursive Confabulation Study' in content
                assert 'RELEASED MODEL x ARM RESULTS' in content
                assert 'POOLED INTERVENTION COMPARISON' in content
                assert 'GROUNDING-CONFABULATION HETEROGENEITY' in content
                assert 'confab_persistence_rate' not in content
                # Grounding section must use confab_rate values (this
                # fixture: model_a=1.0, model_b=0.9), not persist_rate
                # (0.7/0.85) -- the metric-identity correction.
                assert '100.00%' in content or '90.00%' in content
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEchoChamberPlotting:
    """Test Echo Chamber visualization with varying metric availability"""

    def test_echo_panel_without_gr_column(self):
        """Test that Echo panel works when GR column is missing (real echo-chamber-zero case)"""
        import pandas as pd
        import matplotlib.pyplot as plt
        from course_correct_evals.analysis import CrossStudyAnalysis
        from course_correct_evals.analysis.viz import _plot_echo_chamber_panel

        # Create fake Echo data WITHOUT GR column (matches real echo-chamber-zero CSV)
        fake_echo_data = pd.DataFrame({
            'simulation_id': [4, 4, 4, 6, 6, 6],
            'step': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            'SRI': [0.25, 0.35, 0.45, 0.30, 0.40, 0.50],
            'RE': [0.85, 0.75, 0.65, 0.80, 0.70, 0.60],
        })

        # Create observatory and inject fake data
        observatory = CrossStudyAnalysis()
        observatory.echo_chamber_data = fake_echo_data
        observatory._data_loaded['echo_chamber'] = True
        observatory.echo_chamber_analysis = {}  # Mock analysis

        # Create a matplotlib axes
        fig, ax = plt.subplots()

        # This should NOT raise KeyError even though GR is missing
        try:
            _plot_echo_chamber_panel(observatory, ax)
            success = True
        except KeyError as e:
            success = False
            error_msg = str(e)
        finally:
            plt.close(fig)

        assert success, f"Echo panel raised KeyError when GR missing: {error_msg if not success else ''}"

    def test_echo_panel_with_all_metrics(self):
        """Test that Echo panel works when all metrics are present"""
        import pandas as pd
        import matplotlib.pyplot as plt
        from course_correct_evals.analysis import CrossStudyAnalysis
        from course_correct_evals.analysis.viz import _plot_echo_chamber_panel

        # Create fake Echo data WITH all metrics
        fake_echo_data = pd.DataFrame({
            'simulation_id': [4, 4, 6, 6],
            'step': [0.1, 0.2, 0.1, 0.2],
            'GR': [0.3, 0.4, 0.35, 0.45],
            'SRI': [0.25, 0.35, 0.30, 0.40],
            'RE': [0.85, 0.75, 0.80, 0.70],
        })

        # Create observatory and inject fake data
        observatory = CrossStudyAnalysis()
        observatory.echo_chamber_data = fake_echo_data
        observatory._data_loaded['echo_chamber'] = True
        observatory.echo_chamber_analysis = {}

        # Create a matplotlib axes
        fig, ax = plt.subplots()

        # Should work fine with all metrics
        try:
            _plot_echo_chamber_panel(observatory, ax)
            success = True
        except Exception as e:
            success = False
            error_msg = str(e)
        finally:
            plt.close(fig)

        assert success, f"Echo panel failed with all metrics: {error_msg if not success else ''}"


class TestEchoChamberDecoupling:
    """
    Tests for the Echo Chamber Zero Observatory decoupling architecture.

    Canonical Observatory = Mirror Loop + Recursive Confabulation + Violation
    State. Echo Chamber Zero is noncanonical/opt-in and must never appear in
    canonical study counts, leaderboards, or the canonical comparison figure.
    """

    def test_load_all_studies_default_excludes_echo_chamber(self):
        """include_echo_chamber defaults to False; Echo Chamber is not attempted."""
        observatory = CrossStudyAnalysis()
        loaded = observatory.load_all_studies(fail_on_missing=False)

        assert loaded['echo_chamber'] == False
        assert observatory.echo_chamber_data is None
        assert observatory.echo_chamber_analysis is None

    def test_canonical_study_count_excludes_echo_chamber_even_if_loaded(self):
        """total_studies_loaded only ever counts the three canonical studies."""
        observatory = CrossStudyAnalysis()
        # Simulate Echo Chamber having been separately opted into and loaded
        observatory._data_loaded['echo_chamber'] = True

        summary = observatory.get_summary()
        assert summary['total_studies_loaded'] == 0
        # Dormant key retained for backward compatibility
        assert 'echo_chamber' in summary['studies_loaded']

    def test_data_source_summary_enumerates_three_canonical_studies_only(self):
        observatory = CrossStudyAnalysis()
        summary = observatory.get_data_source_summary()
        assert set(summary.keys()) == {'mirror_loop', 'confabulation', 'violation_state'}

    def test_leaderboard_excludes_echo_columns_and_models_even_when_echo_data_has_model_column(self):
        """Leaderboard must never expose Echo Chamber as a model-comparison metric,
        even if Echo Chamber data with a 'model' column is present."""
        import pandas as pd

        observatory = CrossStudyAnalysis()
        observatory._data_loaded['echo_chamber'] = True
        observatory.echo_chamber_data = pd.DataFrame({
            'model': ['gpt-4', 'gpt-4'],
            'GR': [0.5, 0.6],
            'SRI': [0.1, 0.2],
        })

        leaderboard = observatory.create_leaderboard()

        assert 'echo_mean_GR' not in leaderboard.columns
        assert 'echo_mean_SRI' not in leaderboard.columns
        if 'model' in leaderboard.columns:
            assert 'gpt-4' not in leaderboard['model'].tolist()

    def test_four_panel_comparison_draws_three_canonical_panels(self):
        """The canonical comparison figure has exactly three panels (no Echo Chamber panel)."""
        import matplotlib
        matplotlib.use('Agg')
        from course_correct_evals.analysis.viz import plot_four_panel_comparison

        observatory = CrossStudyAnalysis()
        fig = plot_four_panel_comparison(observatory)
        try:
            assert len(fig.axes) == 3
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_include_echo_chamber_opt_in_still_excludes_leaderboard_columns(self):
        """Even with include_echo_chamber=True requested at load time, the
        leaderboard must remain free of Echo Chamber columns."""
        observatory = CrossStudyAnalysis()
        loaded = observatory.load_all_studies(fail_on_missing=False, include_echo_chamber=True)

        assert isinstance(loaded, dict)
        assert 'echo_chamber' in loaded

        leaderboard = observatory.create_leaderboard()
        assert 'echo_mean_GR' not in leaderboard.columns
        assert 'echo_mean_SRI' not in leaderboard.columns


class TestViolationStateCanonicalIntegration:
    """
    Integration-level tests for the canonical, structured-field-based
    Violation State analysis inside CrossStudyAnalysis: reconciliation
    against real released data, leaderboard exclusion, and the
    raw-structured-outcomes visualization panel.
    """

    def test_released_data_reconciliation(self):
        """Reproduce the exact raw (115/4/1) and published (116/120)
        figures from the real, currently released violation-state data,
        through the canonical importer/analysis path."""
        from course_correct_evals.importers import ViolationStateImporter

        importer = ViolationStateImporter()
        df = importer.load_data()
        if df is None:
            pytest.skip("Violation State data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.violation_state_data = df
        observatory._data_loaded['violation_state'] = True

        analysis = observatory.analyze_violation_state()
        raw = analysis['structured']['raw_structured_outcomes']
        published = analysis['structured']['published_aggregate']

        assert raw['contaminated']['n'] == 120
        assert raw['contaminated']['counts'].get('policy_refusal') == 115
        assert raw['contaminated']['counts'].get('image_success') == 4
        assert raw['contaminated']['counts'].get('rate_limit') == 1
        assert raw['contaminated']['counts'].get('capability_refusal', 0) == 0

        assert published['contaminated']['refused'] == 116
        assert published['contaminated']['n'] == 120
        assert abs(published['contaminated']['refusal_rate'] - (116 / 120)) < 1e-9

        assert raw['control']['n'] == 40
        assert raw['control']['counts'].get('image_success') == 40
        assert published['control']['refused'] == 0
        assert published['control']['n'] == 40

    def test_leaderboard_excludes_violation_state_even_with_model_column(self):
        """Violation State must never appear in the leaderboard, even if
        its data were to contain a 'model' column."""
        import pandas as pd

        observatory = CrossStudyAnalysis()
        observatory._data_loaded['violation_state'] = True
        observatory.violation_state_data = pd.DataFrame({
            'model': ['gpt-4', 'gpt-4'],
            'conversation_id': ['c1', 'c1'],
            'condition': ['contaminated', 'contaminated'],
            'turn_number': [1, 2],
            'prompt_id': ['I1_KITCHEN', 'I2_BEDROOM'],
            'response_type': ['policy_refusal', 'policy_refusal'],
        })

        leaderboard = observatory.create_leaderboard()

        assert 'violation_contamination_rate' not in leaderboard.columns
        if 'model' in leaderboard.columns:
            assert 'gpt-4' not in leaderboard['model'].tolist()

    def test_violation_state_panel_uses_raw_categories_not_published_aggregate(self):
        """The canonical panel must be built from raw_structured_outcomes
        (bar heights), not driven by the published_aggregate numbers."""
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from course_correct_evals.analysis.viz import _plot_violation_state_panel

        observatory = CrossStudyAnalysis()
        observatory._data_loaded['violation_state'] = True
        observatory.violation_state_data = pd.DataFrame({
            'conversation_id': ['c1', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c2'],
            'condition': ['contaminated'] * 4 + ['control'] * 4,
            'turn_number': [1, 2, 3, 4] * 2,
            'prompt_id': ['I1_KITCHEN', 'I2_BEDROOM', 'I3_ABSTRACT', 'I4_COFFEE'] * 2,
            'response_type': ['policy_refusal'] * 4 + ['image_success'] * 4,
        })

        fig, ax = plt.subplots()
        try:
            _plot_violation_state_panel(observatory, ax)
            # 4 categories (policy_refusal, capability_refusal,
            # image_success, rate_limit) x 2 conditions = 8 bar patches
            assert len(ax.patches) == 8
            assert len(ax.get_xticklabels()) == 4

            # Lock the invariant that bar heights are driven by
            # raw_structured_outcomes, not published_aggregate: recompute
            # expected heights from the SAME analysis object the panel
            # itself populated (via its internal analyze_violation_state()
            # call), using the panel's own category/condition ordering,
            # and compare against the actual rendered patch heights.
            raw = observatory.violation_state_analysis['structured']['raw_structured_outcomes']
            categories = ['policy_refusal', 'capability_refusal', 'image_success', 'rate_limit']
            conditions = ['contaminated', 'control']

            expected_heights = []
            for cond in conditions:
                counts = raw[cond]['counts']
                expected_heights.extend(counts.get(cat, 0) for cat in categories)

            actual_heights = [p.get_height() for p in ax.patches]
            assert actual_heights == expected_heights
            # For this fixture specifically: contaminated is all
            # policy_refusal, control is all image_success.
            assert expected_heights == [4, 0, 0, 0, 0, 0, 4, 0]
        finally:
            plt.close(fig)


class TestConfabulationPhase4Canonical:
    """
    Phase 4 focused tests: manuscript-faithful N-weighted pooling,
    grounding_pilot exclusion/heterogeneity, canonical visualization, and
    leaderboard arm-explicit columns for Recursive Confabulation.
    """

    @staticmethod
    def _unequal_n_fixture():
        """Deliberately UNEQUAL per-model n within each arm, so N-weighted
        pooling produces a different result than a naive unweighted mean
        of the three model-level rates -- this is what discriminates the
        two methods."""
        import pandas as pd
        return pd.DataFrame({
            'model':        ['model_a', 'model_a', 'model_a', 'model_a',
                              'model_b', 'model_b', 'model_b', 'model_b',
                              'model_c', 'model_c', 'model_c', 'model_c'],
            'arm':          ['baseline', 'fact_table', 'belief_audit', 'grounding_pilot'] * 3,
            'n':            [20, 20, 20, 20,   5, 5, 5, 5,   10, 10, 10, 10],
            'confab_rate':  [1.0] * 12,
            'persist_rate': [0.9, 0.9, 0.9, 1.0,   0.1, 0.1, 0.1, 0.0,   0.5, 0.5, 0.5, 0.5],
        })

    # --- Importer preservation ---

    def test_importer_preserves_all_12_released_rows(self):
        """Live released data: all 12 (model, arm) rows survive, all four
        arm labels present, no importer-level across-arm averaging."""
        from course_correct_evals.importers import ConfabulationImporter

        importer = ConfabulationImporter()
        table = importer.load_data()
        if table is None:
            pytest.skip("Confabulation data not available (network)")

        assert len(table) == 12
        assert set(table['arm'].unique()) == {'baseline', 'fact_table', 'belief_audit', 'grounding_pilot'}
        assert table['model'].nunique() == 3
        assert 'confab_persistence_rate' not in table.columns
        # Every released source column preserved
        for col in ['model', 'arm', 'n', 'confab_rate', 'confab_ci',
                    'persist_rate', 'persist_ci', 'latency_mean',
                    'latency_std', 'blame_mean', 'blame_std']:
            assert col in table.columns

    # --- Pooled manuscript comparison, N-weighting proof ---

    def test_pooled_comparison_is_n_weighted_not_naive_mean(self):
        """With deliberately unequal per-model n, the pooled value must
        equal the N-weighted result and must NOT equal the naive
        unweighted mean of the three model-level rates."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = self._unequal_n_fixture()
        observatory._data_loaded['confabulation'] = True

        result = observatory.analyze_confabulation()
        pooled = result['pooled_intervention_comparison']

        # N-weighted: (0.9*20 + 0.1*5 + 0.5*10) / (20+5+10) = (18+0.5+5)/35 = 23.5/35
        expected_weighted = (0.9 * 20 + 0.1 * 5 + 0.5 * 10) / 35
        naive_mean = (0.9 + 0.1 + 0.5) / 3  # would be 0.5 -- deliberately different

        assert abs(pooled['baseline']['n'] - 35) < 1e-9
        assert abs(pooled['baseline']['persist_rate'] - expected_weighted) < 1e-9
        assert abs(pooled['baseline']['persist_rate'] - naive_mean) > 1e-6

    def test_released_data_pooled_comparison_matches_manuscript(self):
        """Live released data reproduces the exact manuscript figures via
        N-weighting: baseline=17/29, fact_table=25/30, belief_audit=27/30."""
        from course_correct_evals.importers import ConfabulationImporter
        from course_correct_evals.analysis import CrossStudyAnalysis

        importer = ConfabulationImporter()
        table = importer.load_data()
        if table is None:
            pytest.skip("Confabulation data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = table
        observatory._data_loaded['confabulation'] = True
        result = observatory.analyze_confabulation()
        pooled = result['pooled_intervention_comparison']

        assert pooled['baseline']['n'] == 29
        assert abs(pooled['baseline']['persist_rate'] - 17 / 29) < 1e-9
        assert pooled['fact_table']['n'] == 30
        assert abs(pooled['fact_table']['persist_rate'] - 25 / 30) < 1e-9
        assert pooled['belief_audit']['n'] == 30
        assert abs(pooled['belief_audit']['persist_rate'] - 27 / 30) < 1e-9

    # --- Grounding heterogeneity ---

    def test_grounding_pilot_absent_from_pooled_comparison(self):
        """grounding_pilot must never appear as a fourth pooled arm."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = self._unequal_n_fixture()
        observatory._data_loaded['confabulation'] = True

        result = observatory.analyze_confabulation()
        assert 'grounding_pilot' not in result['pooled_intervention_comparison']

    def test_grounding_confabulation_heterogeneity_has_correct_metric_and_values(self):
        """Live released data: grounding_confabulation_heterogeneity uses
        confab_rate (1.0/0.6/0.5), NOT persist_rate (1.0/0.5/0.5) --
        the metric-identity correction. Gemini (0.6 vs 0.5) is the
        discriminating value between the two metrics."""
        from course_correct_evals.importers import ConfabulationImporter
        from course_correct_evals.analysis import CrossStudyAnalysis

        importer = ConfabulationImporter()
        table = importer.load_data()
        if table is None:
            pytest.skip("Confabulation data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = table
        observatory._data_loaded['confabulation'] = True
        result = observatory.analyze_confabulation()

        # No separate canonical grounding-persistence object
        assert 'grounding_heterogeneity' not in result
        assert 'grounding_persistence_by_model' not in result

        grounding = result['grounding_confabulation_heterogeneity']

        # confab_rate values (the manuscript's grounding-confabulation finding)
        assert abs(grounding['anthropic:claude-3-5-haiku-latest']['confab_rate'] - 1.0) < 1e-9
        assert abs(grounding['google:gemini-2.0-flash']['confab_rate'] - 0.6) < 1e-9
        assert abs(grounding['openai:gpt-4o-mini']['confab_rate'] - 0.5) < 1e-9

        # Explicitly NOT the persist_rate values -- proves this is not
        # merely a renamed copy of the old (incorrect) structure.
        assert abs(grounding['google:gemini-2.0-flash']['confab_rate'] - 0.5) > 1e-6

    def test_grounding_persistence_remains_available_only_via_model_arm_table(self):
        """Grounding persistence is NOT promoted to a separate canonical
        object -- it remains available only through model_arm_table
        (filtered to grounding_pilot) and the leaderboard column."""
        from course_correct_evals.importers import ConfabulationImporter
        from course_correct_evals.analysis import CrossStudyAnalysis

        importer = ConfabulationImporter()
        table = importer.load_data()
        if table is None:
            pytest.skip("Confabulation data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = table
        observatory._data_loaded['confabulation'] = True
        result = observatory.analyze_confabulation()

        assert 'grounding_persistence_by_model' not in result

        grounding_rows = result['model_arm_table'][result['model_arm_table']['arm'] == 'grounding_pilot']
        gpt4o_row = grounding_rows[grounding_rows['model'] == 'openai:gpt-4o-mini'].iloc[0]
        assert abs(gpt4o_row['persist_rate'] - 0.5) < 1e-9

    # --- Visualization ---

    def test_panel_shows_exactly_three_pooled_arms_with_correct_heights(self):
        """Canonical panel: exactly 3 bars, heights equal
        pooled_intervention_comparison values, grounding_pilot not a
        fourth bar, grounding annotation present."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from course_correct_evals.analysis import CrossStudyAnalysis
        from course_correct_evals.analysis.viz import _plot_confabulation_panel

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = self._unequal_n_fixture()
        observatory._data_loaded['confabulation'] = True

        fig, ax = plt.subplots()
        try:
            _plot_confabulation_panel(observatory, ax)

            assert len(ax.patches) == 3
            assert [t.get_text() for t in ax.get_xticklabels()] == ['baseline', 'fact_table', 'belief_audit']

            pooled = observatory.confabulation_analysis['pooled_intervention_comparison']
            expected_heights = [pooled[a]['persist_rate'] for a in ['baseline', 'fact_table', 'belief_audit']]
            actual_heights = [p.get_height() for p in ax.patches]
            assert actual_heights == expected_heights

            annotation_texts = [t.get_text() for t in ax.texts]
            # Annotation must reference grounding's CONFABULATION effect
            # specifically -- not leave the metric identity ambiguous with
            # the persistence metric the 3 bars above actually use.
            assert any('grounding_pilot' in t and 'CONFABULATION' in t for t in annotation_texts)
        finally:
            plt.close(fig)

    # --- Leaderboard ---

    def test_leaderboard_has_four_arm_columns_no_legacy_column(self):
        """Leaderboard exposes exactly the four approved arm-explicit
        columns, each a literal released value; old column absent; at
        least one discriminating value asserted directly."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = self._unequal_n_fixture()
        observatory._data_loaded['confabulation'] = True

        leaderboard = observatory.create_leaderboard()

        assert 'confab_persistence_rate' not in leaderboard.columns
        for col in ['confab_persist_rate_baseline', 'confab_persist_rate_fact_table',
                    'confab_persist_rate_belief_audit', 'confab_persist_rate_grounding_pilot']:
            assert col in leaderboard.columns

        # Discriminating value: model_b's baseline (0.1) differs sharply
        # from model_a's (0.9) and from any collapsed average -- a future
        # collapsed implementation could not accidentally satisfy this.
        model_b_row = leaderboard[leaderboard['model'] == 'model_b'].iloc[0]
        assert abs(model_b_row['confab_persist_rate_baseline'] - 0.1) < 1e-9
        assert abs(model_b_row['confab_persist_rate_grounding_pilot'] - 0.0) < 1e-9

    # --- Negative canonical invariant ---

    def test_no_canonical_path_exposes_an_across_arm_model_scalar(self):
        """Scoped to canonical Recursive Confabulation outputs: scan
        analyze_confabulation(), get_summary(), and create_leaderboard()
        for any per-model, across-arm-collapsed persistence scalar --
        under any name, not just the old 'confab_persistence_rate'."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = self._unequal_n_fixture()
        observatory._data_loaded['confabulation'] = True

        analysis = observatory.analyze_confabulation()
        summary = observatory.get_summary()
        leaderboard = observatory.create_leaderboard()

        forbidden_keys = {
            'confab_persistence_rate', 'mean_persistence_rate',
            'descriptive_aggregate_by_model', 'persistence_rate',
            'grounding_persistence_by_model',  # not promoted to a canonical object
            'grounding_heterogeneity',  # old (pre-correction) name; superseded by
                                         # grounding_confabulation_heterogeneity
        }
        assert forbidden_keys.isdisjoint(analysis.keys())
        assert forbidden_keys.isdisjoint(summary.get('confabulation', {}).keys())
        assert forbidden_keys.isdisjoint(set(leaderboard.columns))

        # The only confabulation-prefixed leaderboard columns must be the
        # four approved arm-explicit ones -- nothing else collapsed.
        confab_cols = [c for c in leaderboard.columns if c.startswith('confab')]
        assert set(confab_cols) == {
            'confab_persist_rate_baseline', 'confab_persist_rate_fact_table',
            'confab_persist_rate_belief_audit', 'confab_persist_rate_grounding_pilot',
        }


class TestMirrorLoopPhase5Canonical:
    """
    Phase 5 discriminating tests: canonical Mirror Loop analysis uses the
    released edit_change measurement and the manuscript-defined
    rolling-3-step plateau statistic, never a recomputation from response
    text and never the removed mirror_collapse_rate construct.
    """

    @staticmethod
    def _synthetic_df(edit_change, response, iteration=None, sequence_id='s1',
                       model='m', condition='grounded'):
        n = len(edit_change)
        return pd.DataFrame({
            'iteration': iteration or list(range(1, n + 1)),
            'edit_change': edit_change,
            'response': response,
            'sequence_id': [sequence_id] * n,
            'model': [model] * n,
            'condition': [condition] * n,
        })

    def test_canonical_plateau_uses_released_edit_change_not_recomputed_response(self):
        """Released edit_change is engineered HIGH (never plateaus), while
        the response text is engineered IDENTICAL every iteration (which
        would recompute to ~0 ΔI and falsely plateau immediately if the
        canonical path ever recomputed from response text instead of
        reading edit_change). The canonical result must show NOT
        plateaued, proving it reads edit_change and ignores response."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        identical_response = "the same response text every single iteration"
        df = self._synthetic_df(
            edit_change=[0.5, 0.45, 0.5, 0.48, 0.5, 0.47],
            response=[identical_response] * 6,
        )

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()

        stats = result['plateau']['group_summary'][('m', 'grounded')]
        assert stats['n_plateaued'] == 0
        assert stats['plateau_rate'] == 0.0

    def test_no_unsupported_novelty_aggregate(self):
        """The manuscript's n-gram novelty finding is trajectory-based/
        per-iteration, not a single dataset-wide mean. This guards
        specifically against an unsupported dataset-wide novelty SCALAR
        (mean_ngram_novelty_overall, or an equivalent overall-mean scalar
        under another name) reappearing in the canonical result or
        summary, even when the released ngram_novelty column is present
        in the data.

        This is NOT a blanket ban on any key containing the word
        "novelty" -- a separately approved, future implementation of the
        manuscript's per-iteration novelty TRAJECTORY (e.g. a
        'novelty_trajectory' structure) is legitimate and must not be
        blocked by this test."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        df = self._synthetic_df(
            edit_change=[0.5, 0.45, 0.5, 0.48],
            response=["a", "b", "c", "d"],
        )
        df['ngram_novelty'] = [0.77, 0.77, 0.77, 0.77]

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()
        summary = observatory.get_summary()

        assert 'mean_ngram_novelty_overall' not in result
        assert 'mean_ngram_novelty_overall' not in summary.get('mirror_loop', {})

        def _is_unsupported_overall_mean_scalar_key(k: str) -> bool:
            # Flags only dataset-wide mean/overall-SCALAR novelty names
            # (e.g. mean_ngram_novelty_overall, ngram_novelty_mean,
            # overall_novelty_mean) -- not a trajectory/per-iteration
            # structure (e.g. 'novelty_trajectory'), which is not this
            # invariant's concern.
            k = k.lower()
            return 'novelty' in k and ('mean' in k or 'overall' in k)

        assert not any(_is_unsupported_overall_mean_scalar_key(k) for k in result.keys())
        assert not any(_is_unsupported_overall_mean_scalar_key(k)
                        for k in summary.get('mirror_loop', {}).keys())

    def test_real_data_gpt4o_mini_ungrounded_matches_manuscript_reference(self):
        """Live released data, tau=0.05 primary: gpt-4o-mini/ungrounded
        reproduces the manuscript's reference figures exactly: 9/24
        plateaued, median iteration 5, IQR (5, 6)."""
        from course_correct_evals.importers import MirrorLoopImporter
        from course_correct_evals.analysis import CrossStudyAnalysis

        importer = MirrorLoopImporter()
        df = importer.load_data()
        if df is None or 'condition' not in df.columns:
            pytest.skip("Mirror Loop data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()

        stats = result['plateau']['group_summary'][('gpt-4o-mini', 'ungrounded')]
        assert stats['n_sequences'] == 24
        assert stats['n_plateaued'] == 9
        assert abs(stats['median_plateau_iteration'] - 5.0) < 1e-9
        assert stats['plateau_iteration_iqr'] == (5.0, 6.0)

    def test_real_data_gemini_never_plateaus_either_condition(self):
        """Live released data, tau=0.05 primary: gemini-2.0-flash shows
        0/24 plateaued in BOTH grounded and ungrounded conditions."""
        from course_correct_evals.importers import MirrorLoopImporter
        from course_correct_evals.analysis import CrossStudyAnalysis

        importer = MirrorLoopImporter()
        df = importer.load_data()
        if df is None or 'condition' not in df.columns:
            pytest.skip("Mirror Loop data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()

        for condition in ('grounded', 'ungrounded'):
            stats = result['plateau']['group_summary'][('gemini-2.0-flash', condition)]
            assert stats['n_sequences'] == 24
            assert stats['n_plateaued'] == 0
            assert stats['plateau_rate'] == 0.0
            assert stats['median_plateau_iteration'] is None

    def test_real_data_grounding_rebound_matches_manuscript(self):
        """Live released data: grounding rebound (grounded condition,
        pooled ΔI, iteration 2 -> 4) reproduces the manuscript's reference
        figures: 0.148078 -> 0.190191, +28.44%."""
        from course_correct_evals.importers import MirrorLoopImporter
        from course_correct_evals.analysis import CrossStudyAnalysis

        importer = MirrorLoopImporter()
        df = importer.load_data()
        if df is None or 'condition' not in df.columns:
            pytest.skip("Mirror Loop data not available (network)")

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()

        rebound = result['grounding_rebound']
        assert rebound is not None
        assert abs(rebound['delta_i_from'] - 0.148078) < 1e-5
        assert abs(rebound['delta_i_to'] - 0.190191) < 1e-5
        assert abs(rebound['pct_increase'] - 28.44) < 0.05

    def test_no_legacy_collapse_scalar_anywhere(self):
        """mirror_collapse_rate must not exist anywhere -- analysis,
        summary, or leaderboard -- removed, not aliased."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        df = self._synthetic_df(
            edit_change=[0.5, 0.01, 0.01, 0.01],
            response=["a", "b", "c", "d"],
        )

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()
        summary = observatory.get_summary()
        leaderboard = observatory.create_leaderboard()

        assert 'mirror_collapse_rate' not in result
        assert 'collapse_rate' not in result
        assert 'collapsed_sequences' not in result
        assert 'mirror_collapse_rate' not in summary.get('mirror_loop', {})
        assert 'mirror_collapse_rate' not in leaderboard.columns

        assert 'mirror_plateau_rate_grounded' in leaderboard.columns
        assert 'mirror_plateau_rate_ungrounded' in leaderboard.columns

    def test_sensitivity_tau_0_02_isolated_from_primary_and_leaderboard(self):
        """The tau=0.02 sensitivity view must exist only under
        plateau_sensitivity_tau_0_02, must never feed the leaderboard, and
        must never alter the primary tau=0.05 result -- constructed with a
        sequence that plateaus under tau=0.05 but NOT under the stricter
        tau=0.02, so the two views are guaranteed to disagree."""
        from course_correct_evals.analysis import CrossStudyAnalysis

        # rolling-3 mean settles at 0.03: qualifies for tau=0.05 but not tau=0.02
        df = self._synthetic_df(
            edit_change=[0.5, 0.5, 0.03, 0.03, 0.03, 0.03],
            response=["a", "b", "c", "d", "e", "f"],
        )

        observatory = CrossStudyAnalysis()
        observatory.mirror_loop_data = df
        observatory._data_loaded['mirror_loop'] = True
        result = observatory.analyze_mirror_loop()

        primary_stats = result['plateau']['group_summary'][('m', 'grounded')]
        sensitivity_stats = result['plateau_sensitivity_tau_0_02']['group_summary'][('m', 'grounded')]

        assert primary_stats['n_plateaued'] == 1  # tau=0.05: plateaus
        assert sensitivity_stats['n_plateaued'] == 0  # tau=0.02: does not

        leaderboard = observatory.create_leaderboard()
        row = leaderboard[leaderboard['model'] == 'm'].iloc[0]
        assert abs(row['mirror_plateau_rate_grounded'] - 1.0) < 1e-9  # reflects PRIMARY only

        assert 'plateau_sensitivity_tau_0_02' not in leaderboard.columns
        for col in leaderboard.columns:
            assert '0_02' not in col and 'sensitivity' not in col
