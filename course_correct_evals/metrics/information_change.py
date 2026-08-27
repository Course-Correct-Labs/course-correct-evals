"""
Information Change Metrics (ΔI)

Metrics for detecting information collapse in iterative reasoning sequences.
Used primarily for the Mirror Loop study.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
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

    # Imported here, not at module scope: this is the only function in the
    # package that needs Levenshtein, and it is a noncanonical/legacy
    # utility (see module docstring below) -- canonical Observatory import
    # must not require this optional dependency.
    import Levenshtein

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


# ---------------------------------------------------------------------------
# Canonical, released-measurement-based Mirror Loop plateau analysis.
#
# The functions above (delta_i_edit_distance, ngram_novelty, analyze_sequence,
# analyze_dataframe_sequences) recompute ΔI/novelty from raw response text.
# They remain valid generic/noncanonical utilities, but they are NOT used to
# reconstruct Mirror Loop's canonical released measurements: a direct
# numerical check established that delta_i_edit_distance() does not
# reproduce the released edit_change column, and a separate check
# established that ngram_novelty() does not reproduce the released
# ngram_novelty column either. The functions below use the released columns
# directly.
#
# PROVENANCE: the rolling-three-step plateau definition, the primary
# tau=0.05 threshold, the tau=0.02 sensitivity threshold, the
# per-sequence-then-aggregate interpretation, and the grounding-rebound
# definition all come from the Mirror Loop manuscript, which is supplied
# outside the cloned mirror-loop GitHub repository (that repository
# explicitly states the manuscript is not included in it). Only the
# released numerical measurements themselves (edit_change, ngram_novelty)
# are verified directly from data/mirror_loop_results_all.csv.
# ---------------------------------------------------------------------------

def detect_sequence_plateau(
    seq_df: pd.DataFrame,
    tau: float = 0.05,
    window: int = 3,
    iteration_col: str = 'iteration',
    value_col: str = 'edit_change',
) -> Optional[int]:
    """
    Manuscript-defined per-sequence plateau detection.

    Sorts the sequence by iteration, drops rows with missing ΔI (iteration 0
    has no defined edit_change), computes a TRAILING rolling `window`-step
    mean of the released ΔI values, and returns the iteration at the END of
    the first window whose mean is strictly below `tau`.

    The window-label convention (trailing, labeled at the window's last
    iteration) is not independently specified in the manuscript excerpt
    available for this implementation -- it is the standard trailing
    rolling-average convention, and it exactly reproduces the manuscript's
    reported GPT-4o-mini x ungrounded reference result (9/24 plateaued,
    median iteration 5, IQR 5-6) when applied to the released data. That
    reproduction is the evidence for this specific convention, not an
    independent manuscript citation for the labeling rule itself.

    Returns:
        The plateau iteration (int), or None if no qualifying window exists
        (the sequence is classified as NOT plateaued -- never a fabricated
        iteration).
    """
    seq_df = seq_df.sort_values(iteration_col)
    di = seq_df.set_index(iteration_col)[value_col]
    di = di[np.isfinite(di)]
    if len(di) < window:
        return None
    rolling_avg = di.rolling(window).mean()
    below_tau = rolling_avg[rolling_avg < tau]
    if len(below_tau) == 0:
        return None
    return int(below_tau.index[0])


def analyze_mirror_loop_plateau(
    df: pd.DataFrame,
    tau: float = 0.05,
    window: int = 3,
    sequence_id_col: str = 'sequence_id',
    iteration_col: str = 'iteration',
    value_col: str = 'edit_change',
    group_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Canonical Mirror Loop plateau analysis: PER-SEQUENCE detection first,
    then aggregation across sequences within each group. Never computes a
    rolling average on a pooled/averaged group trajectory.

    Args:
        df: released Mirror Loop data (one row per sequence x iteration),
            using the released `value_col` (edit_change) as ΔI.
        tau: plateau threshold (strict less-than).
        window: rolling window size (manuscript: 3).
        group_cols: columns defining the aggregation groups (default:
            ['model', 'condition'], matching the manuscript's own reporting
            granularity, e.g. "GPT-4o-mini ungrounded").

    Returns:
        Dict with:
          - 'sequence_results': DataFrame, one row per sequence, columns
            group_cols + [sequence_id_col, 'plateaued', 'plateau_iteration'].
          - 'group_summary': dict keyed by the group tuple, each value a
            dict with n_sequences, n_plateaued, plateau_rate,
            median_plateau_iteration, plateau_iteration_iqr (the latter two
            computed ONLY over the plateaued sequences; the denominator
            n_sequences/n_plateaued is preserved separately, never
            manufactured by assigning fake iterations to non-plateauing
            sequences).
    """
    if group_cols is None:
        group_cols = ['model', 'condition']

    seq_rows = []
    for seq_id, seq_df in df.groupby(sequence_id_col):
        plateau_iter = detect_sequence_plateau(
            seq_df, tau=tau, window=window,
            iteration_col=iteration_col, value_col=value_col,
        )
        row = {sequence_id_col: seq_id, 'plateaued': plateau_iter is not None,
               'plateau_iteration': plateau_iter}
        for gc in group_cols:
            row[gc] = seq_df[gc].iloc[0]
        seq_rows.append(row)

    sequence_results = pd.DataFrame(seq_rows)

    group_summary: Dict[Any, Dict[str, Any]] = {}
    for group_key, group_df in sequence_results.groupby(group_cols):
        n_sequences = len(group_df)
        plateaued_df = group_df[group_df['plateaued']]
        n_plateaued = len(plateaued_df)
        plateau_iters = plateaued_df['plateau_iteration'].dropna().tolist()

        summary = {
            'n_sequences': int(n_sequences),
            'n_plateaued': int(n_plateaued),
            'plateau_rate': float(n_plateaued / n_sequences) if n_sequences > 0 else 0.0,
            'median_plateau_iteration': float(np.median(plateau_iters)) if plateau_iters else None,
            'plateau_iteration_iqr': (
                (float(np.percentile(plateau_iters, 25)), float(np.percentile(plateau_iters, 75)))
                if plateau_iters else None
            ),
        }
        group_summary[group_key if isinstance(group_key, tuple) else (group_key,)] = summary

    return {
        'sequence_results': sequence_results,
        'group_summary': group_summary,
        'group_cols': group_cols,
        'tau': tau,
        'window': window,
    }


def compute_grounding_rebound(
    df: pd.DataFrame,
    condition: str = 'grounded',
    condition_col: str = 'condition',
    iteration_col: str = 'iteration',
    value_col: str = 'edit_change',
    iteration_from: int = 2,
    iteration_to: int = 4,
) -> Dict[str, Any]:
    """
    Manuscript-defined grounding-rebound statistic: DISTINCT from plateau
    detection. Pools released ΔI across all sequences within the given
    condition (default 'grounded'), takes the pooled mean at iteration_from
    and iteration_to, and reports the percentage increase. This is a
    pooled two-point comparison, not a per-sequence detection-then-aggregate
    statistic -- do not derive it from, or fold it into, the plateau
    structure.

    Returns:
        Dict with 'condition', 'iteration_from', 'iteration_to',
        'delta_i_from', 'delta_i_to', 'pct_increase'.
    """
    sub = df[df[condition_col] == condition]
    pooled = sub.groupby(iteration_col)[value_col].mean()

    delta_i_from = float(pooled.loc[iteration_from])
    delta_i_to = float(pooled.loc[iteration_to])
    pct_increase = ((delta_i_to - delta_i_from) / delta_i_from) * 100 if delta_i_from else None

    return {
        'condition': condition,
        'iteration_from': iteration_from,
        'iteration_to': iteration_to,
        'delta_i_from': delta_i_from,
        'delta_i_to': delta_i_to,
        'pct_increase': pct_increase,
    }
