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


class TestRealWorldSchemas:
    """Test importers with real-world GitHub CSV schemas"""

    def test_mirror_loop_with_seq_id(self):
        """Test MirrorLoopImporter with real GitHub schema using seq_id"""
        # Real schema from mirror-loop repo
        data = pd.DataFrame({
            'provider': ['openai', 'openai', 'openai'],
            'model': ['gpt-4', 'gpt-4', 'gpt-4'],
            'temperature': [0.7, 0.7, 0.7],
            'task_type': ['critique', 'critique', 'critique'],
            'task_id': ['task1', 'task1', 'task1'],
            'condition': ['control', 'control', 'control'],
            'iteration': [0, 1, 2],
            'output_text': ['First output', 'Second output', 'Third output'],
            'output_length_chars': [100, 95, 90],
            'edit_change': [0.0, 0.1, 0.15],
            'char_entropy': [4.5, 4.3, 4.2],
            'ngram_novelty': [0.8, 0.6, 0.5],
            'emb_cosine': [1.0, 0.95, 0.92],
            'no_new_info_claimed': [False, False, True],
            'possible_grounding_violation': [False, False, False],
            'refusal_or_resistance': [False, False, False],
            'seq_id': ['sequence_1', 'sequence_1', 'sequence_1'],
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            importer = MirrorLoopImporter(data_path=temp_path)
            df = importer.load_data()

            assert df is not None, "Failed to load data with seq_id column"
            assert len(df) == 3
            assert 'iteration' in df.columns
            assert 'model' in df.columns
            assert 'response' in df.columns
            assert 'sequence_id' in df.columns

            # Verify mapping
            assert all(df['response'] == data['output_text'])
            assert all(df['sequence_id'] == data['seq_id'])
        finally:
            os.unlink(temp_path)

    def test_violation_state_with_real_schema(self):
        """Test ViolationStateImporter with real GitHub schema"""
        # Real schema from violation-state repo
        data = pd.DataFrame({
            'thread_id': ['thread1', 'thread1', 'thread2', 'thread2'],
            'condition': ['violation', 'violation', 'control', 'control'],
            'user_turn_index': [0, 1, 0, 1],
            'assistant_turn_index': [0, 1, 0, 1],
            'prompt_id': ['p1', 'p1', 'p2', 'p2'],
            'user_text': ['User msg 1', 'User msg 2', 'User msg 3', 'User msg 4'],
            'assistant_text': ['Assistant 1', 'Assistant 2', 'Assistant 3', 'Assistant 4'],
            'response_class': ['normal', 'contaminated', 'normal', 'normal'],
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            importer = ViolationStateImporter(data_path=temp_path)
            df = importer.load_data()

            assert df is not None, "Failed to load data with real violation-state schema"
            assert len(df) == 4
            assert 'conversation_id' in df.columns
            assert 'turn_number' in df.columns
            assert 'content' in df.columns
            assert 'response_type' in df.columns
            assert 'condition' in df.columns

            # Verify mapping
            assert all(df['conversation_id'] == data['thread_id'])
            assert all(df['turn_number'] == data['assistant_turn_index'])
            assert all(df['content'] == data['assistant_text'])
            assert all(df['response_type'] == data['response_class'])
        finally:
            os.unlink(temp_path)

    def test_echo_chamber_with_real_schema(self):
        """Test EchoChamberImporter with real GitHub schema"""
        # Real schema from echo-chamber-zero repo
        data = pd.DataFrame({
            'mean_degree': [4, 4, 4, 6, 6, 6],
            'p': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            'SRI': [0.25, 0.35, 0.45, 0.30, 0.40, 0.50],
            'RE': [0.85, 0.75, 0.65, 0.80, 0.70, 0.60],
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            importer = EchoChamberImporter(data_path=temp_path)
            df = importer.load_data()

            assert df is not None, "Failed to load data with real echo-chamber schema"
            assert len(df) == 6
            assert 'simulation_id' in df.columns
            assert 'step' in df.columns
            assert 'SRI' in df.columns
            assert 'RE' in df.columns

            # Verify mapping
            assert all(df['simulation_id'] == data['mean_degree'])
            assert all(df['step'] == data['p'])
        finally:
            os.unlink(temp_path)

    def test_confabulation_handles_404(self):
        """Test ConfabulationImporter handles missing data gracefully"""
        # Test that load_data returns None when no data is available
        importer = ConfabulationImporter(data_path="/nonexistent/path.csv")
        df = importer.load_data()

        assert df is None, "Should return None for nonexistent file"
        assert importer.data_source == "not_loaded"
