"""
Fabrication Persistence Metrics

Metrics for analyzing how fabrications persist across conversational turns.
Used primarily for the Recursive Confabulation study.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict


def calculate_persistence_rate(
    df: pd.DataFrame,
    conversation_id_col: str = 'conversation_id',
    turn_col: str = 'turn_number',
    fabrication_col: str = 'fabrication_present',
    intervention_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate fabrication persistence rate.

    Persistence = fraction of fabrications that appear in the next turn
    after they first appear.

    Args:
        df: DataFrame with conversation data
        conversation_id_col: Column name for conversation ID
        turn_col: Column name for turn number
        fabrication_col: Column name for fabrication indicator (boolean)
        intervention_col: Optional column for intervention arm

    Returns:
        Dictionary with persistence statistics
    """
    persistence_counts = {
        'total_fabrications': 0,
        'persistent_fabrications': 0,
        'non_persistent': 0,
    }

    by_intervention = defaultdict(lambda: {
        'total_fabrications': 0,
        'persistent_fabrications': 0,
        'non_persistent': 0,
    })

    # Analyze each conversation
    for conv_id, conv_df in df.groupby(conversation_id_col):
        conv_df = conv_df.sort_values(turn_col)

        intervention_arm = None
        if intervention_col and intervention_col in conv_df.columns:
            intervention_arm = conv_df[intervention_col].iloc[0]

        turns = conv_df[turn_col].values
        fabrications = conv_df[fabrication_col].values

        # Check each turn for fabrication
        for i in range(len(turns) - 1):
            if fabrications[i]:
                # Fabrication found
                persistence_counts['total_fabrications'] += 1

                if intervention_arm:
                    by_intervention[intervention_arm]['total_fabrications'] += 1

                # Check if it persists to next turn
                if fabrications[i + 1]:
                    persistence_counts['persistent_fabrications'] += 1

                    if intervention_arm:
                        by_intervention[intervention_arm]['persistent_fabrications'] += 1
                else:
                    persistence_counts['non_persistent'] += 1

                    if intervention_arm:
                        by_intervention[intervention_arm]['non_persistent'] += 1

    # Calculate rates
    total = persistence_counts['total_fabrications']
    if total > 0:
        persistence_counts['persistence_rate'] = persistence_counts['persistent_fabrications'] / total
    else:
        persistence_counts['persistence_rate'] = 0.0

    # Calculate rates by intervention
    for arm, counts in by_intervention.items():
        arm_total = counts['total_fabrications']
        if arm_total > 0:
            counts['persistence_rate'] = counts['persistent_fabrications'] / arm_total
        else:
            counts['persistence_rate'] = 0.0

    result = {
        'overall': persistence_counts,
    }

    if by_intervention:
        result['by_intervention'] = dict(by_intervention)

    return result


def detect_fabrication_propagation(
    df: pd.DataFrame,
    conversation_id_col: str = 'conversation_id',
    turn_col: str = 'turn_number',
    fabrication_col: str = 'fabrication_present',
    content_col: str = 'content',
    min_propagation_length: int = 3
) -> pd.DataFrame:
    """
    Detect fabrication propagation chains.

    A propagation chain is a sequence of consecutive turns where
    fabrication is present.

    Args:
        df: DataFrame with conversation data
        conversation_id_col: Column for conversation ID
        turn_col: Column for turn number
        fabrication_col: Column for fabrication indicator
        content_col: Column with message content
        min_propagation_length: Minimum chain length to report

    Returns:
        DataFrame with detected propagation chains
    """
    chains = []

    for conv_id, conv_df in df.groupby(conversation_id_col):
        conv_df = conv_df.sort_values(turn_col)

        turns = conv_df[turn_col].values
        fabrications = conv_df[fabrication_col].values

        # Find consecutive fabrication sequences
        chain_start = None
        chain_length = 0

        for i, (turn, is_fab) in enumerate(zip(turns, fabrications)):
            if is_fab:
                if chain_start is None:
                    chain_start = turn
                chain_length += 1
            else:
                # Chain ended
                if chain_length >= min_propagation_length:
                    chains.append({
                        'conversation_id': conv_id,
                        'chain_start_turn': chain_start,
                        'chain_end_turn': turns[i-1],
                        'chain_length': chain_length,
                    })

                chain_start = None
                chain_length = 0

        # Check if chain extends to end of conversation
        if chain_length >= min_propagation_length:
            chains.append({
                'conversation_id': conv_id,
                'chain_start_turn': chain_start,
                'chain_end_turn': turns[-1],
                'chain_length': chain_length,
            })

    return pd.DataFrame(chains)


def calculate_intervention_effectiveness(
    df: pd.DataFrame,
    intervention_col: str = 'intervention_arm',
    fabrication_col: str = 'fabrication_present',
    baseline_name: str = 'baseline'
) -> pd.DataFrame:
    """
    Calculate intervention effectiveness relative to baseline.

    Args:
        df: DataFrame with conversation data
        intervention_col: Column for intervention arm
        fabrication_col: Column for fabrication indicator
        baseline_name: Name of baseline condition

    Returns:
        DataFrame with effectiveness metrics per intervention
    """
    if intervention_col not in df.columns:
        raise ValueError(f"Column '{intervention_col}' not found in data")

    # Calculate fabrication rate by intervention
    results = []

    for arm, arm_df in df.groupby(intervention_col):
        total_turns = len(arm_df)
        fabrication_count = arm_df[fabrication_col].sum()
        fabrication_rate = fabrication_count / total_turns if total_turns > 0 else 0.0

        results.append({
            'intervention_arm': arm,
            'total_turns': total_turns,
            'fabrication_count': int(fabrication_count),
            'fabrication_rate': float(fabrication_rate),
        })

    results_df = pd.DataFrame(results)

    # Calculate effectiveness relative to baseline
    baseline_rate = results_df[results_df['intervention_arm'] == baseline_name]['fabrication_rate'].values

    if len(baseline_rate) > 0:
        baseline_rate = baseline_rate[0]

        results_df['relative_effectiveness'] = results_df['fabrication_rate'].apply(
            lambda x: (baseline_rate - x) / baseline_rate if baseline_rate > 0 else 0.0
        )
    else:
        results_df['relative_effectiveness'] = 0.0

    return results_df


def analyze_fabrication_context(
    df: pd.DataFrame,
    conversation_id_col: str = 'conversation_id',
    turn_col: str = 'turn_number',
    fabrication_col: str = 'fabrication_present',
    content_col: str = 'content',
    context_window: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract context around fabrications for qualitative analysis.

    Args:
        df: DataFrame with conversation data
        conversation_id_col: Column for conversation ID
        turn_col: Column for turn number
        fabrication_col: Column for fabrication indicator
        content_col: Column with message content
        context_window: Number of turns before/after to include

    Returns:
        List of dictionaries with fabrication contexts
    """
    contexts = []

    for conv_id, conv_df in df.groupby(conversation_id_col):
        conv_df = conv_df.sort_values(turn_col).reset_index(drop=True)

        for idx in range(len(conv_df)):
            if conv_df.loc[idx, fabrication_col]:
                # Extract context
                start_idx = max(0, idx - context_window)
                end_idx = min(len(conv_df), idx + context_window + 1)

                context = {
                    'conversation_id': conv_id,
                    'fabrication_turn': conv_df.loc[idx, turn_col],
                    'fabrication_content': conv_df.loc[idx, content_col],
                    'context_turns': [],
                }

                for ctx_idx in range(start_idx, end_idx):
                    context['context_turns'].append({
                        'turn': conv_df.loc[ctx_idx, turn_col],
                        'content': conv_df.loc[ctx_idx, content_col],
                        'is_fabrication': conv_df.loc[ctx_idx, fabrication_col],
                    })

                contexts.append(context)

    return contexts
