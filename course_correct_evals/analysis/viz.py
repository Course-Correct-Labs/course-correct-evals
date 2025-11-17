"""
Visualization Module

Publication-quality visualizations for CCL Observatory.
"""

from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

# Set default style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_four_panel_comparison(
    observatory: 'CrossStudyAnalysis',
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create the flagship four-panel comparison figure.

    Panel 1: Mirror Loop ΔI curves
    Panel 2: Confabulation persistence across intervention arms
    Panel 3: Violation contamination/refusal rates
    Panel 4: Echo Chamber GR/SRI trajectories

    Args:
        observatory: CrossStudyAnalysis instance with loaded data
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Mirror Loop
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_mirror_loop_panel(observatory, ax1)

    # Panel 2: Confabulation
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_confabulation_panel(observatory, ax2)

    # Panel 3: Violation State
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_violation_state_panel(observatory, ax3)

    # Panel 4: Echo Chamber
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_echo_chamber_panel(observatory, ax4)

    # Main title
    fig.suptitle(
        'CCL Reasoning Stability Observatory: Cross-Study Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    return fig


def _plot_mirror_loop_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """Plot Mirror Loop ΔI curves."""
    if not observatory._data_loaded['mirror_loop']:
        ax.text(0.5, 0.5, 'Mirror Loop Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 1: Mirror Loop - Information Collapse')
        return

    # Get sequence analysis
    if observatory.mirror_loop_analysis is None:
        observatory.analyze_mirror_loop()

    seq_analysis = observatory.mirror_loop_analysis['sequence_analysis']

    # Plot ΔI trajectories for sample of sequences
    sample_size = min(10, len(seq_analysis))
    sample_sequences = seq_analysis.sample(n=sample_size, random_state=42)

    for _, row in sample_sequences.iterrows():
        delta_i_values = row['delta_i_edit']
        iterations = range(1, len(delta_i_values) + 1)

        alpha = 0.3 if not row['collapse_detected'] else 0.7
        color = 'red' if row['collapse_detected'] else 'blue'
        label = 'Collapsed' if row['collapse_detected'] else 'Stable'

        ax.plot(iterations, delta_i_values, alpha=alpha, color=color)

    # Add mean trajectory
    max_len = max(len(row['delta_i_edit']) for _, row in seq_analysis.iterrows())
    mean_trajectory = []

    for i in range(max_len):
        values_at_i = [row['delta_i_edit'][i] for _, row in seq_analysis.iterrows()
                      if i < len(row['delta_i_edit'])]
        if values_at_i:
            mean_trajectory.append(np.mean(values_at_i))

    if mean_trajectory:
        ax.plot(range(1, len(mean_trajectory) + 1), mean_trajectory,
               'k--', linewidth=2, label='Mean', alpha=0.8)

    # Collapse threshold line
    threshold = seq_analysis.iloc[0]['collapse_threshold']
    ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5,
              label=f'Collapse Threshold ({threshold})')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ΔI (Edit Distance)', fontsize=11)
    ax.set_title('Panel 1: Mirror Loop - Information Collapse', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)


def _plot_confabulation_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """Plot Confabulation persistence across intervention arms."""
    if not observatory._data_loaded['confabulation']:
        ax.text(0.5, 0.5, 'Confabulation Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 2: Recursive Confabulation - Persistence')
        return

    # Get analysis
    if observatory.confabulation_analysis is None:
        observatory.analyze_confabulation()

    pers_stats = observatory.confabulation_analysis['persistence_statistics']

    # Plot by intervention arm if available
    if 'by_intervention' in pers_stats and pers_stats['by_intervention']:
        arms = []
        rates = []

        for arm, stats in pers_stats['by_intervention'].items():
            arms.append(arm)
            rates.append(stats['persistence_rate'])

        # Bar plot
        colors = sns.color_palette("husl", len(arms))
        bars = ax.bar(arms, rates, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Persistence Rate', fontsize=11)
        ax.set_xlabel('Intervention Arm', fontsize=11)
        ax.set_ylim(0, max(rates) * 1.2 if rates else 1)

    else:
        # Overall rate only
        overall_rate = pers_stats['overall']['persistence_rate']
        ax.bar(['Overall'], [overall_rate], color='steelblue', alpha=0.7, edgecolor='black')
        ax.text(0, overall_rate, f'{overall_rate:.1%}',
               ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('Persistence Rate', fontsize=11)
        ax.set_ylim(0, overall_rate * 1.2 if overall_rate > 0 else 1)

    ax.set_title('Panel 2: Recursive Confabulation - Persistence', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


def _plot_violation_state_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """Plot Violation contamination/refusal rates."""
    if not observatory._data_loaded['violation_state']:
        ax.text(0.5, 0.5, 'Violation State Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 3: Violation State - Contamination')
        return

    # Get analysis
    if observatory.violation_state_analysis is None:
        observatory.analyze_violation_state()

    refusal_stats = observatory.violation_state_analysis['refusal_statistics']

    # If model-level stats available
    if 'model' in refusal_stats.columns and len(refusal_stats) > 1:
        # Plot by model
        models = refusal_stats['model'].tolist()
        refusal_rates = refusal_stats['refusal_rate'].tolist()

        colors = sns.color_palette("husl", len(models))
        bars = ax.barh(models, refusal_rates, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.1%}',
                   ha='left', va='center', fontsize=9)

        ax.set_xlabel('Refusal Rate', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)
        ax.set_xlim(0, max(refusal_rates) * 1.2 if refusal_rates else 1)

    else:
        # Overall stats
        overall = refusal_stats.iloc[0]
        categories = ['Refusal', 'Compliance', 'Rate Limit', 'Error']
        counts = [
            overall.get('refusals', 0),
            overall.get('compliance', 0),
            overall.get('rate_limits', 0),
            overall.get('errors', 0)
        ]

        colors = ['red', 'green', 'orange', 'gray']
        ax.pie(counts, labels=categories, autopct='%1.1f%%',
              colors=colors, startangle=90)

    ax.set_title('Panel 3: Violation State - Response Distribution', fontsize=12, fontweight='bold')


def _plot_echo_chamber_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """Plot Echo Chamber GR/SRI trajectories."""
    if not observatory._data_loaded['echo_chamber']:
        ax.text(0.5, 0.5, 'Echo Chamber Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 4: Echo Chamber - Percolation')
        return

    # Get analysis
    if observatory.echo_chamber_analysis is None:
        observatory.analyze_echo_chamber()

    # Plot mean GR and SRI over time
    data = observatory.echo_chamber_data

    # Build aggregation spec dynamically - only include columns that exist
    agg_spec = {}
    if 'GR' in data.columns:
        agg_spec['GR'] = 'mean'
    if 'SRI' in data.columns:
        agg_spec['SRI'] = 'mean'
    if 'RE' in data.columns:
        agg_spec['RE'] = 'mean'

    # If no metrics available, show fallback message
    if not agg_spec:
        ax.text(0.5, 0.5, 'Echo metrics not available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 4: Echo Chamber - Percolation Metrics', fontsize=12, fontweight='bold')
        return

    # Aggregate by step
    agg_data = data.groupby('step').agg(agg_spec).reset_index()

    steps = agg_data['step']

    if 'GR' in data.columns and agg_data['GR'].notna().any():
        ax.plot(steps, agg_data['GR'], 'o-', label='GR (Group Radicalization)',
               linewidth=2, markersize=4, alpha=0.8)

    if 'SRI' in data.columns and agg_data['SRI'].notna().any():
        ax.plot(steps, agg_data['SRI'], 's-', label='SRI (Self-Reinforcement)',
               linewidth=2, markersize=4, alpha=0.8)

    if 'RE' in data.columns and agg_data['RE'].notna().any():
        ax.plot(steps, agg_data['RE'], '^-', label='RE (Reasoning Entropy)',
               linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Panel 4: Echo Chamber - Percolation Metrics', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)


def plot_leaderboard(
    leaderboard_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model leaderboard as a heatmap.

    Args:
        leaderboard_df: DataFrame from create_leaderboard()
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Select numeric columns
    numeric_cols = leaderboard_df.select_dtypes(include=[np.number]).columns.tolist()
    plot_data = leaderboard_df[['model'] + numeric_cols].set_index('model')

    # Create heatmap
    sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
               cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)

    ax.set_title('Cross-Study Model Leaderboard', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Model', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Leaderboard saved to {save_path}")

    return fig


def plot_mirror_loop_detail(
    observatory: 'CrossStudyAnalysis',
    sequence_id: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Detailed plot for a single Mirror Loop sequence.

    Args:
        observatory: CrossStudyAnalysis instance
        sequence_id: Specific sequence to plot (if None, picks one)
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Figure object
    """
    if not observatory._data_loaded['mirror_loop']:
        raise ValueError("Mirror Loop data not loaded")

    # Get a sequence
    if sequence_id is None:
        sequence_id = observatory.mirror_loop_data['sequence_id'].iloc[0]

    seq_data = observatory.mirror_loop_data[
        observatory.mirror_loop_data['sequence_id'] == sequence_id
    ].sort_values('iteration')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    iterations = seq_data['iteration'].values
    responses = seq_data['response'].values

    # Calculate ΔI
    delta_i_values = []
    for i in range(1, len(responses)):
        from ..metrics import delta_i_edit_distance
        di = delta_i_edit_distance(responses[i-1], responses[i])
        delta_i_values.append(di)

    # Plot ΔI
    ax1.plot(iterations[1:], delta_i_values, 'o-', linewidth=2, markersize=6)
    ax1.set_ylabel('ΔI (Edit Distance)', fontsize=11)
    ax1.set_title(f'Sequence {sequence_id}: Information Change', fontweight='bold')
    ax1.grid(alpha=0.3)

    # Plot word count
    word_counts = [len(r.split()) if r else 0 for r in responses]
    ax2.plot(iterations, word_counts, 's-', color='green', linewidth=2, markersize=6)
    ax2.set_ylabel('Word Count', fontsize=11)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
