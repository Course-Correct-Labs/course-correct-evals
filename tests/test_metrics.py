"""
Tests for metrics modules
"""

import pytest
import pandas as pd
import numpy as np

from course_correct_evals.metrics import (
    delta_i_edit_distance,
    ngram_novelty,
    analyze_sequence,
    word_count,
    concreteness_score,
    proper_noun_density,
    detect_compression,
    classify_response_type,
    detect_contamination,
)


class TestInformationChange:
    """Test information change metrics"""

    def test_delta_i_edit_distance_identical(self):
        """Test ΔI for identical texts"""
        text = "The quick brown fox jumps over the lazy dog"
        di = delta_i_edit_distance(text, text)
        assert di == 0.0

    def test_delta_i_edit_distance_different(self):
        """Test ΔI for different texts"""
        text1 = "Hello world"
        text2 = "Goodbye world"
        di = delta_i_edit_distance(text1, text2)
        assert di > 0.0
        assert di <= 1.0  # Normalized

    def test_delta_i_edit_distance_empty(self):
        """Test ΔI for empty texts"""
        di = delta_i_edit_distance("", "")
        assert di == 0.0

    def test_ngram_novelty_all_novel(self):
        """Test n-gram novelty when all n-grams are new"""
        text = "completely new content here"
        previous = ["old content", "different text"]
        novelty = ngram_novelty(text, previous, n=2)
        assert novelty > 0.0
        assert novelty <= 1.0

    def test_ngram_novelty_no_novel(self):
        """Test n-gram novelty when text is repeated"""
        text = "same text"
        previous = ["same text"]
        novelty = ngram_novelty(text, previous, n=2)
        assert novelty == 0.0

    def test_analyze_sequence(self):
        """Test sequence analysis"""
        texts = [
            "This is the first iteration.",
            "This is the second iteration.",
            "This is the third iteration.",
        ]
        analysis = analyze_sequence(texts)

        assert 'delta_i_edit' in analysis
        assert 'ngram_novelty' in analysis
        assert 'collapse_detected' in analysis
        assert len(analysis['delta_i_edit']) == len(texts) - 1


class TestSemanticCompression:
    """Test semantic compression metrics"""

    def test_word_count(self):
        """Test word counting"""
        assert word_count("Hello world") == 2
        assert word_count("") == 0
        assert word_count("One two three four five") == 5

    def test_proper_noun_density(self):
        """Test proper noun density calculation"""
        # Text with proper nouns
        text = "I visited Paris and London last summer."
        density = proper_noun_density(text)
        assert density > 0.0

        # Text without proper nouns
        text = "i visited cities last summer."
        density = proper_noun_density(text)
        assert density == 0.0

    def test_concreteness_score(self):
        """Test concreteness scoring"""
        # Concrete text (names, numbers)
        concrete = "John Smith scored 95 points in 2024."
        score_concrete = concreteness_score(concrete)

        # Abstract text
        abstract = "it was nice and good."
        score_abstract = concreteness_score(abstract)

        assert score_concrete > score_abstract

    def test_detect_compression(self):
        """Test compression detection"""
        # Compressing sequence (decreasing word count)
        texts = [
            "This is a very long and detailed explanation of the concept.",
            "This is a shorter explanation.",
            "Short.",
        ]
        result = detect_compression(texts, method='slope')

        assert 'compression_detected' in result
        assert 'word_counts' in result
        assert len(result['word_counts']) == len(texts)


class TestSessionContamination:
    """Test session contamination metrics"""

    def test_classify_response_type_refusal(self):
        """Test refusal classification"""
        refusal_text = "I cannot help with that request."
        classification = classify_response_type(refusal_text)
        assert classification == 'refusal'

    def test_classify_response_type_compliance(self):
        """Test compliance classification"""
        compliance_text = "Here is the answer to your question."
        classification = classify_response_type(compliance_text)
        assert classification == 'compliance'

    def test_detect_contamination_positive(self):
        """Test contamination detection (positive case)"""
        contaminated_text = "This involves harmful and dangerous activities that are illegal."
        result = detect_contamination(contaminated_text, min_keyword_matches=2)

        assert result['contaminated'] == True
        assert result['keyword_matches'] >= 2

    def test_detect_contamination_negative(self):
        """Test contamination detection (negative case)"""
        clean_text = "This is a perfectly normal and helpful response."
        result = detect_contamination(clean_text, min_keyword_matches=2)

        assert result['contaminated'] == False


def test_imports():
    """Test that all metric functions are importable"""
    from course_correct_evals.metrics import (
        delta_i_edit_distance,
        delta_i_embedding,
        ngram_novelty,
        analyze_sequence,
        word_count,
        concreteness_score,
        proper_noun_density,
        detect_compression,
        calculate_persistence_rate,
        detect_fabrication_propagation,
        classify_response_type,
        detect_contamination,
        analyze_echo_metrics,
        detect_threshold_crossing,
    )
    assert True  # If imports work, test passes
