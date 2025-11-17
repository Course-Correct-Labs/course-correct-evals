"""
Tests for analysis module
"""

import pytest
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


class TestConfabulationBackwardsCompatibility:
    """Test RC analysis backwards compatibility"""

    def test_aggregate_rc_has_persistence_statistics(self):
        """Test that aggregate RC data returns persistence_statistics structure"""
        import pandas as pd
        from course_correct_evals.analysis import CrossStudyAnalysis

        # Create fake aggregate RC data
        fake_rc_data = pd.DataFrame({
            'model': ['model_a', 'model_b', 'model_c'],
            'confab_persistence_rate': [0.7, 0.85, 0.6],
            'confab_rate': [1.0, 0.9, 1.0],
            'n': [10, 10, 10],
        })

        # Create observatory and inject fake data
        observatory = CrossStudyAnalysis()
        observatory.confabulation_data = fake_rc_data
        observatory._data_loaded['confabulation'] = True

        # Analyze
        result = observatory.analyze_confabulation()

        # Check backwards-compatible structure
        assert 'persistence_statistics' in result, "Missing persistence_statistics key"
        assert 'overall' in result['persistence_statistics'], "Missing persistence_statistics.overall"
        assert 'persistence_rate' in result['persistence_statistics']['overall'], "Missing overall persistence_rate"

        # Verify it's a float
        assert isinstance(result['persistence_statistics']['overall']['persistence_rate'], float)

        # Check per-model structure
        assert 'by_model' in result['persistence_statistics'], "Missing by_model key"
        assert 'model_a' in result['persistence_statistics']['by_model']
        assert 'model_b' in result['persistence_statistics']['by_model']
        assert 'model_c' in result['persistence_statistics']['by_model']

        # Verify model-specific rates
        assert abs(result['persistence_statistics']['by_model']['model_a']['persistence_rate'] - 0.7) < 0.01
        assert abs(result['persistence_statistics']['by_model']['model_b']['persistence_rate'] - 0.85) < 0.01
        assert abs(result['persistence_statistics']['by_model']['model_c']['persistence_rate'] - 0.6) < 0.01

        # Verify overall is the mean
        expected_mean = (0.7 + 0.85 + 0.6) / 3
        assert abs(result['persistence_statistics']['overall']['persistence_rate'] - expected_mean) < 0.01


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
