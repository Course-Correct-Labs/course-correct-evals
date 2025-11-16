"""
Tests for data importers
"""

import pytest
import pandas as pd
import tempfile
import os

from course_correct_evals.importers import (
    MirrorLoopImporter,
    ConfabulationImporter,
    ViolationStateImporter,
    EchoChamberImporter,
)


@pytest.fixture
def sample_mirror_loop_csv():
    """Create a sample Mirror Loop CSV file"""
    data = pd.DataFrame({
        'iteration': [0, 1, 2, 0, 1, 2],
        'model': ['gpt-4', 'gpt-4', 'gpt-4', 'claude-3', 'claude-3', 'claude-3'],
        'response': ['First response', 'Second response', 'Third response',
                     'Another first', 'Another second', 'Another third'],
        'sequence_id': ['seq1', 'seq1', 'seq1', 'seq2', 'seq2', 'seq2'],
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def sample_confabulation_csv():
    """Create a sample Confabulation CSV file"""
    data = pd.DataFrame({
        'conversation_id': ['conv1', 'conv1', 'conv1', 'conv2', 'conv2'],
        'turn_number': [1, 2, 3, 1, 2],
        'content': ['Message 1', 'Message 2', 'Message 3', 'Msg A', 'Msg B'],
        'role': ['user', 'assistant', 'user', 'user', 'assistant'],
        'fabrication_present': [False, True, True, False, False],
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def sample_violation_state_csv():
    """Create a sample Violation State CSV file"""
    data = pd.DataFrame({
        'conversation_id': ['sess1', 'sess1', 'sess2', 'sess2'],
        'turn_number': [1, 2, 1, 2],
        'content': ['Request 1', 'Response 1', 'Request 2', 'Response 2'],
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def sample_echo_chamber_csv():
    """Create a sample Echo Chamber CSV file"""
    data = pd.DataFrame({
        'simulation_id': ['sim1', 'sim1', 'sim1', 'sim2', 'sim2', 'sim2'],
        'step': [0, 1, 2, 0, 1, 2],
        'GR': [0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
        'SRI': [0.2, 0.4, 0.6, 0.3, 0.5, 0.7],
        'RE': [0.9, 0.7, 0.5, 0.8, 0.6, 0.4],
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    os.unlink(f.name)


class TestMirrorLoopImporter:
    """Test Mirror Loop importer"""

    def test_load_data(self, sample_mirror_loop_csv):
        """Test loading Mirror Loop data"""
        importer = MirrorLoopImporter(data_path=sample_mirror_loop_csv)
        df = importer.load_data()

        assert df is not None
        assert len(df) == 6
        assert 'iteration' in df.columns
        assert 'model' in df.columns
        assert 'response' in df.columns
        assert 'sequence_id' in df.columns

    def test_get_data_info(self, sample_mirror_loop_csv):
        """Test getting data info"""
        importer = MirrorLoopImporter(data_path=sample_mirror_loop_csv)
        importer.load_data()
        info = importer.get_data_info()

        assert info['status'] == 'loaded'
        assert info['total_rows'] == 6
        assert info['num_sequences'] == 2

    def test_get_sequence(self, sample_mirror_loop_csv):
        """Test getting a specific sequence"""
        importer = MirrorLoopImporter(data_path=sample_mirror_loop_csv)
        importer.load_data()
        seq = importer.get_sequence('seq1')

        assert len(seq) == 3
        assert all(seq['sequence_id'] == 'seq1')


class TestConfabulationImporter:
    """Test Confabulation importer"""

    def test_load_data(self, sample_confabulation_csv):
        """Test loading Confabulation data"""
        importer = ConfabulationImporter(data_path=sample_confabulation_csv)
        df = importer.load_data()

        assert df is not None
        assert len(df) == 5
        assert 'conversation_id' in df.columns
        assert 'turn_number' in df.columns
        assert 'content' in df.columns
        assert 'fabrication_present' in df.columns

    def test_get_fabricated_turns(self, sample_confabulation_csv):
        """Test getting fabricated turns"""
        importer = ConfabulationImporter(data_path=sample_confabulation_csv)
        importer.load_data()
        fab_turns = importer.get_fabricated_turns()

        assert len(fab_turns) == 2  # Two fabrications in sample data


class TestViolationStateImporter:
    """Test Violation State importer"""

    def test_load_data(self, sample_violation_state_csv):
        """Test loading Violation State data"""
        importer = ViolationStateImporter(data_path=sample_violation_state_csv)
        df = importer.load_data()

        assert df is not None
        assert len(df) == 4
        assert 'conversation_id' in df.columns
        assert 'turn_number' in df.columns
        assert 'content' in df.columns


class TestEchoChamberImporter:
    """Test Echo Chamber importer"""

    def test_load_data(self, sample_echo_chamber_csv):
        """Test loading Echo Chamber data"""
        importer = EchoChamberImporter(data_path=sample_echo_chamber_csv)
        df = importer.load_data()

        assert df is not None
        assert len(df) == 6
        assert 'simulation_id' in df.columns
        assert 'step' in df.columns
        assert 'GR' in df.columns
        assert 'SRI' in df.columns
        assert 'RE' in df.columns

    def test_get_metrics_over_time(self, sample_echo_chamber_csv):
        """Test getting metrics over time"""
        importer = EchoChamberImporter(data_path=sample_echo_chamber_csv)
        importer.load_data()
        metrics = importer.get_metrics_over_time()

        assert 'step' in metrics.columns
        assert 'GR' in metrics.columns or 'SRI' in metrics.columns


def test_all_importers_importable():
    """Test that all importers can be imported"""
    from course_correct_evals import (
        MirrorLoopImporter,
        ConfabulationImporter,
        ViolationStateImporter,
        EchoChamberImporter,
    )
    assert True
