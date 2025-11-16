"""
Semantic Compression Metrics

Metrics for detecting semantic compression in text outputs.
Used to identify when model outputs become less specific or concrete.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import re


def word_count(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    if not text:
        return 0
    return len(text.split())


def proper_noun_density(text: str) -> float:
    """
    Estimate proper noun density (capitalized words that aren't sentence-initial).

    This is a simple heuristic for concreteness.

    Args:
        text: Input text

    Returns:
        Ratio of proper nouns to total words
    """
    if not text:
        return 0.0

    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)

    proper_nouns = 0
    total_words = 0

    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue

        # Skip first word of sentence (likely capitalized for grammar)
        for word in words[1:]:
            # Check if starts with capital letter
            if word and word[0].isupper():
                # Simple filter: must be at least 2 chars and alphanumeric
                if len(word) >= 2 and any(c.isalpha() for c in word):
                    proper_nouns += 1

        total_words += len(words)

    if total_words == 0:
        return 0.0

    return proper_nouns / total_words


def concreteness_score(text: str) -> float:
    """
    Calculate concreteness score based on proper nouns and numbers.

    Higher scores indicate more concrete, specific content.

    Args:
        text: Input text

    Returns:
        Concreteness score (0-1 scale, higher = more concrete)
    """
    if not text:
        return 0.0

    words = text.split()
    if len(words) == 0:
        return 0.0

    # Count proper nouns (heuristic)
    proper_noun_count = 0
    for i, word in enumerate(words):
        # Skip first word of text
        if i == 0:
            continue
        if word and word[0].isupper() and len(word) >= 2:
            proper_noun_count += 1

    # Count numbers
    number_count = sum(1 for word in words if any(c.isdigit() for c in word))

    # Count specific indicators (dates, measurements, etc.)
    specific_patterns = [
        r'\d{4}',  # Years
        r'\d+%',   # Percentages
        r'\$\d+',  # Money
        r'\d+\.\d+',  # Decimals
    ]

    specific_count = sum(len(re.findall(pattern, text)) for pattern in specific_patterns)

    # Combine metrics
    total_concrete = proper_noun_count + number_count + specific_count

    # Normalize by word count (with ceiling to prevent > 1)
    score = min(total_concrete / len(words), 1.0)

    return score


def detect_compression(
    texts: List[str],
    method: str = 'slope',
    threshold: float = -0.1
) -> Dict[str, Any]:
    """
    Detect semantic compression in a sequence of texts.

    Compression is indicated by:
    - Decreasing word count
    - Decreasing concreteness
    - Decreasing proper noun density

    Args:
        texts: List of texts in chronological order
        method: Detection method ('slope', 'threshold')
        threshold: Threshold for compression detection (negative slope or ratio)

    Returns:
        Dictionary with compression analysis
    """
    if not texts or len(texts) < 2:
        return {
            "compression_detected": False,
            "error": "Need at least 2 texts"
        }

    # Calculate metrics for each text
    word_counts = [word_count(t) for t in texts]
    concreteness = [concreteness_score(t) for t in texts]
    proper_noun_densities = [proper_noun_density(t) for t in texts]

    # Detect compression based on method
    compression_detected = False
    compression_point = None

    if method == 'slope':
        # Calculate slopes (simple linear regression)
        x = np.arange(len(texts))

        # Word count slope
        if len(word_counts) > 1:
            wc_slope = np.polyfit(x, word_counts, 1)[0]
        else:
            wc_slope = 0

        # Concreteness slope
        if len(concreteness) > 1:
            conc_slope = np.polyfit(x, concreteness, 1)[0]
        else:
            conc_slope = 0

        # Compression if slopes are negative beyond threshold
        if wc_slope < threshold or conc_slope < threshold:
            compression_detected = True

        result = {
            "compression_detected": compression_detected,
            "method": method,
            "word_count_slope": float(wc_slope),
            "concreteness_slope": float(conc_slope),
            "threshold": threshold,
        }

    elif method == 'threshold':
        # Check if metrics drop below threshold of initial value
        initial_wc = word_counts[0] if word_counts else 1
        initial_conc = concreteness[0] if concreteness else 1

        for i in range(1, len(texts)):
            wc_ratio = (word_counts[i] - initial_wc) / initial_wc if initial_wc > 0 else 0
            conc_ratio = (concreteness[i] - initial_conc) / initial_conc if initial_conc > 0 else 0

            if wc_ratio < threshold or conc_ratio < threshold:
                compression_detected = True
                compression_point = i
                break

        result = {
            "compression_detected": compression_detected,
            "compression_point": compression_point,
            "method": method,
            "threshold": threshold,
        }

    else:
        raise ValueError(f"Unknown method: {method}")

    # Add detailed metrics
    result.update({
        "word_counts": word_counts,
        "concreteness_scores": concreteness,
        "proper_noun_densities": proper_noun_densities,
        "word_count_mean": float(np.mean(word_counts)),
        "concreteness_mean": float(np.mean(concreteness)),
    })

    return result


def analyze_compression_dataframe(
    df: pd.DataFrame,
    text_column: str = 'response',
    sequence_id_column: str = 'sequence_id',
    iteration_column: str = 'iteration',
    method: str = 'slope',
    threshold: float = -0.1
) -> pd.DataFrame:
    """
    Analyze compression for all sequences in a DataFrame.

    Args:
        df: DataFrame with sequence data
        text_column: Name of text column
        sequence_id_column: Name of sequence ID column
        iteration_column: Name of iteration column
        method: Detection method
        threshold: Compression threshold

    Returns:
        DataFrame with compression analysis per sequence
    """
    results = []

    for seq_id, group in df.groupby(sequence_id_column):
        group = group.sort_values(iteration_column)
        texts = group[text_column].tolist()

        analysis = detect_compression(texts, method=method, threshold=threshold)
        analysis['sequence_id'] = seq_id

        results.append(analysis)

    return pd.DataFrame(results)
