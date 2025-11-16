"""
Metrics modules for CCL studies.

Each module implements specific metrics for analyzing reasoning stability.
"""

from .information_change import (
    delta_i_edit_distance,
    delta_i_embedding,
    ngram_novelty,
    analyze_sequence,
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
)

from .session_contamination import (
    classify_response_type,
    detect_contamination,
)

from .percolation import (
    analyze_echo_metrics,
    detect_threshold_crossing,
)

__all__ = [
    # Information change
    "delta_i_edit_distance",
    "delta_i_embedding",
    "ngram_novelty",
    "analyze_sequence",
    # Semantic compression
    "word_count",
    "concreteness_score",
    "proper_noun_density",
    "detect_compression",
    # Persistence
    "calculate_persistence_rate",
    "detect_fabrication_propagation",
    # Session contamination
    "classify_response_type",
    "detect_contamination",
    # Percolation
    "analyze_echo_metrics",
    "detect_threshold_crossing",
]
