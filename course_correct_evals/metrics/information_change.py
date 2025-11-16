"""
Information Change Metrics (ΔI)

Metrics for detecting information collapse in iterative reasoning sequences.
Used primarily for the Mirror Loop study.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import Levenshtein
import warnings


def delta_i_edit_distance(text1: str, text2: str, normalize: bool = True) -> float:
    """
    Calculate information change using normalized Levenshtein distance.

    Args:
        text1: First text
        text2: Second text
        normalize: If True, normalize by max length

    Returns:
        Edit distance (0-1 if normalized, raw count otherwise)
    """
    if not text1 or not text2:
        return 0.0 if (not text1 and not text2) else 1.0

    distance = Levenshtein.distance(text1, text2)

    if normalize:
        max_len = max(len(text1), len(text2))
        return distance / max_len if max_len > 0 else 0.0

    return float(distance)


def delta_i_embedding(
    text1: str,
    text2: str,
    model_name: str = 'all-MiniLM-L6-v2',
    metric: str = 'cosine'
) -> Optional[float]:
    """
    Calculate information change using embedding similarity.

    OPTIONAL: Requires sentence-transformers model download.
    Falls back gracefully if model unavailable.

    Args:
        text1: First text
        text2: Second text
        model_name: Sentence transformer model name
        metric: Distance metric ('cosine', 'euclidean')

    Returns:
        Distance between embeddings (0-1 for cosine, or None if unavailable)
    """
    try:
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import cosine, euclidean

        # Load model (cached after first call)
        model = SentenceTransformer(model_name)

        # Generate embeddings
        emb1 = model.encode(text1, convert_to_numpy=True)
        emb2 = model.encode(text2, convert_to_numpy=True)

        # Calculate distance
        if metric == 'cosine':
            # Convert cosine similarity to distance
            return float(cosine(emb1, emb2))
        elif metric == 'euclidean':
            return float(euclidean(emb1, emb2))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    except ImportError as e:
        warnings.warn(
            f"Embedding-based ΔI unavailable: {e}. "
            "Install sentence-transformers for this metric."
        )
        return None
    except Exception as e:
        warnings.warn(f"Error computing embedding distance: {e}")
        return None


def ngram_novelty(
    text: str,
    previous_texts: List[str],
    n: int = 3,
    normalize: bool = True
) -> float:
    """
    Calculate n-gram novelty - fraction of n-grams not seen in previous texts.

    Args:
        text: Current text to analyze
        previous_texts: List of previous texts in sequence
        n: N-gram size
        normalize: If True, return fraction; otherwise count

    Returns:
        Novelty score (0-1 if normalized)
    """
    if not text or not previous_texts:
        return 1.0 if normalize else 0.0

    # Tokenize (simple whitespace split)
    def get_ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        if len(words) < n:
            return {' '.join(words)}
        return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

    current_ngrams = get_ngrams(text, n)

    if len(current_ngrams) == 0:
        return 0.0

    # Collect all previous n-grams
    previous_ngrams = set()
    for prev_text in previous_texts:
        previous_ngrams.update(get_ngrams(prev_text, n))

    # Count novel n-grams
    novel_ngrams = current_ngrams - previous_ngrams
    novel_count = len(novel_ngrams)

    if normalize:
        return novel_count / len(current_ngrams)

    return float(novel_count)


def analyze_sequence(
    texts: List[str],
    sequence_id: Optional[str] = None,
    use_embeddings: bool = False,
    collapse_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze a complete sequence for information collapse.

    Args:
        texts: List of texts in chronological order
        sequence_id: Optional identifier for the sequence
        use_embeddings: If True, compute embedding-based ΔI
        collapse_threshold: Threshold below which ΔI indicates collapse

    Returns:
        Dictionary with analysis results
    """
    if not texts or len(texts) < 2:
        return {
            "sequence_id": sequence_id,
            "length": len(texts) if texts else 0,
            "error": "Need at least 2 texts to analyze"
        }

    n_texts = len(texts)

    # Calculate ΔI metrics for each consecutive pair
    delta_i_edit = []
    delta_i_emb = [] if use_embeddings else None
    ngram_nov = []

    for i in range(1, n_texts):
        # Edit distance
        di_edit = delta_i_edit_distance(texts[i-1], texts[i])
        delta_i_edit.append(di_edit)

        # Embedding distance (optional)
        if use_embeddings:
            di_emb = delta_i_embedding(texts[i-1], texts[i])
            if di_emb is not None:
                delta_i_emb.append(di_emb)

        # N-gram novelty
        novelty = ngram_novelty(texts[i], texts[:i])
        ngram_nov.append(novelty)

    # Detect collapse
    initial_delta_i = delta_i_edit[0] if delta_i_edit else 0.0
    collapse_detected = False
    collapse_iteration = None

    for i, di in enumerate(delta_i_edit):
        # Collapse if ΔI drops below threshold * initial ΔI
        if initial_delta_i > 0 and di < collapse_threshold * initial_delta_i:
            collapse_detected = True
            collapse_iteration = i + 1  # +1 because delta_i_edit starts at iteration 1
            break
        # Or if absolute ΔI is very small
        elif di < collapse_threshold:
            collapse_detected = True
            collapse_iteration = i + 1
            break

    # Calculate statistics
    result = {
        "sequence_id": sequence_id,
        "length": n_texts,
        "delta_i_edit": delta_i_edit,
        "delta_i_edit_mean": float(np.mean(delta_i_edit)),
        "delta_i_edit_std": float(np.std(delta_i_edit)),
        "delta_i_edit_min": float(np.min(delta_i_edit)),
        "delta_i_edit_max": float(np.max(delta_i_edit)),
        "ngram_novelty": ngram_nov,
        "ngram_novelty_mean": float(np.mean(ngram_nov)),
        "collapse_detected": collapse_detected,
        "collapse_iteration": collapse_iteration,
        "collapse_threshold": collapse_threshold,
    }

    # Add embedding metrics if available
    if use_embeddings and delta_i_emb and len(delta_i_emb) > 0:
        result["delta_i_embedding"] = delta_i_emb
        result["delta_i_embedding_mean"] = float(np.mean(delta_i_emb))
        result["delta_i_embedding_std"] = float(np.std(delta_i_emb))

    return result


def analyze_dataframe_sequences(
    df: pd.DataFrame,
    text_column: str = 'response',
    sequence_id_column: str = 'sequence_id',
    iteration_column: str = 'iteration',
    use_embeddings: bool = False,
    collapse_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Analyze all sequences in a DataFrame.

    Args:
        df: DataFrame with sequence data
        text_column: Name of column containing text
        sequence_id_column: Name of column with sequence IDs
        iteration_column: Name of column with iteration numbers
        use_embeddings: Whether to compute embedding metrics
        collapse_threshold: Threshold for collapse detection

    Returns:
        DataFrame with one row per sequence and analysis results
    """
    results = []

    # Group by sequence
    for seq_id, group in df.groupby(sequence_id_column):
        # Sort by iteration
        group = group.sort_values(iteration_column)

        # Extract texts
        texts = group[text_column].tolist()

        # Analyze
        analysis = analyze_sequence(
            texts=texts,
            sequence_id=seq_id,
            use_embeddings=use_embeddings,
            collapse_threshold=collapse_threshold
        )

        results.append(analysis)

    return pd.DataFrame(results)
