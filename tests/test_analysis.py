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
