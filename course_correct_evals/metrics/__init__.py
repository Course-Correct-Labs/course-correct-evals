"""
Metrics modules for CCL studies.

Each module implements specific metrics for analyzing reasoning stability.
"""

from .information_change import (
    delta_i_edit_distance,
    delta_i_embedding,
    ngram_novelty,
    analyze_sequence,
    analyze_dataframe_sequences,
)

from .semantic_compression import (
    word_count,
    concreteness_score,
    proper_noun_density,
    detect_compression,
)

from .persistence import (
    calculate_persistence_rate,
    detect_fabrication_propagation,
    calculate_intervention_effectiveness,
)

from .session_contamination import (
    classify_response_type,
    detect_contamination,
    classify_responses_dataframe,
    detect_contamination_dataframe,
    analyze_contamination_spread,
    calculate_refusal_rates,
)

from .percolation import (
    analyze_echo_metrics,
    detect_threshold_crossing,
    calculate_convergence_statistics,
    analyze_metric_trajectories,
)

__all__ = [
    # Information change
    "delta_i_edit_distance",
    "delta_i_embedding",
    "ngram_novelty",
    "analyze_sequence",
    "analyze_dataframe_sequences",
    # Semantic compression
    "word_count",
    "concreteness_score",
    "proper_noun_density",
    "detect_compression",
    # Persistence
    "calculate_persistence_rate",
    "detect_fabrication_propagation",
    "calculate_intervention_effectiveness",
    # Session contamination
    "classify_response_type",
    "detect_contamination",
    "classify_responses_dataframe",
    "detect_contamination_dataframe",
    "analyze_contamination_spread",
    "calculate_refusal_rates",
    # Percolation
    "analyze_echo_metrics",
    "detect_threshold_crossing",
    "calculate_convergence_statistics",
    "analyze_metric_trajectories",
]
