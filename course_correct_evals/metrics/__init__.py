"""
Metrics modules for CCL studies.

Each module implements specific metrics for analyzing reasoning stability.
"""

from .information_change import (
    # noncanonical/legacy generic utilities -- do not recompute Mirror
    # Loop's canonical released measurements; see information_change.py
    delta_i_edit_distance,
    delta_i_embedding,
    ngram_novelty,
    analyze_sequence,
    analyze_dataframe_sequences,
    # canonical -- released-measurement-based Mirror Loop plateau analysis
    detect_sequence_plateau,
    analyze_mirror_loop_plateau,
    compute_grounding_rebound,
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
    # noncanonical/legacy — generic text-pattern classifier, not the
    # Violation State study's construct; see session_contamination.py
    classify_response_type,
    detect_contamination,
    classify_responses_dataframe,
    detect_contamination_dataframe,
    analyze_contamination_spread,
    calculate_refusal_rates,
    # canonical — structured-field-based Violation State analysis
    collapse_violation_state_prompts,
    analyze_violation_state_structured,
    BENIGN_IMAGE_PROMPTS,
)

from .percolation import (
    # noncanonical/opt-in — Echo Chamber Zero is not part of the
    # Observatory's canonical study set; see percolation.py
    analyze_echo_metrics,
    detect_threshold_crossing,
    calculate_convergence_statistics,
    analyze_metric_trajectories,
)

__all__ = [
    # Information change (noncanonical/legacy generic utilities)
    "delta_i_edit_distance",
    "delta_i_embedding",
    "ngram_novelty",
    "analyze_sequence",
    "analyze_dataframe_sequences",
    # Mirror Loop (canonical, released-measurement-based)
    "detect_sequence_plateau",
    "analyze_mirror_loop_plateau",
    "compute_grounding_rebound",
    # Semantic compression
    "word_count",
    "concreteness_score",
    "proper_noun_density",
    "detect_compression",
    # Persistence
    "calculate_persistence_rate",
    "detect_fabrication_propagation",
    "calculate_intervention_effectiveness",
    # Session contamination (noncanonical/legacy generic classifier)
    "classify_response_type",
    "detect_contamination",
    "classify_responses_dataframe",
    "detect_contamination_dataframe",
    "analyze_contamination_spread",
    "calculate_refusal_rates",
    # Violation State (canonical, structured-field-based)
    "collapse_violation_state_prompts",
    "analyze_violation_state_structured",
    "BENIGN_IMAGE_PROMPTS",
    # Percolation
    "analyze_echo_metrics",
    "detect_threshold_crossing",
    "calculate_convergence_statistics",
    "analyze_metric_trajectories",
]
