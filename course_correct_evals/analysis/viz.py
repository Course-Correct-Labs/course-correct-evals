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
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create the flagship canonical comparison figure for the three
    Observatory studies.

    Panel 1: Mirror Loop ΔI curves
    Panel 2: Confabulation persistence across intervention arms
    Panel 3: Violation contamination/refusal rates

    Echo Chamber Zero is NOT part of this canonical figure (it is
    noncanonical/opt-in — see _plot_echo_chamber_panel(), which remains
    available as a standalone, non-canonical plotting helper).

    Note: function name retained as plot_four_panel_comparison for
    backward compatibility with existing imports; it now draws three
    canonical panels.

    Args:
        observatory: CrossStudyAnalysis instance with loaded data
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Mirror Loop
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_mirror_loop_panel(observatory, ax1)

    # Panel 2: Confabulation
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_confabulation_panel(observatory, ax2)

    # Panel 3: Violation State
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_violation_state_panel(observatory, ax3)

    # Main title
    fig.suptitle(
        'CCL Reasoning Stability Observatory: Cross-Study Comparison',
        fontsize=16,
        fontweight='bold',
        y=1.03
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")

    return fig


def _plot_mirror_loop_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """
    Plot Mirror Loop's manuscript-defined plateau rate by model x condition
    (PRIMARY canonical threshold, tau=0.05), derived from
    analyze_mirror_loop()'s per-sequence-then-aggregated plateau structure
    -- never a pooled-trajectory crossing.

    This panel intentionally does NOT plot pooled/sampled ΔI trajectories
    with a single "collapse threshold" line: that geometry cannot honestly
    represent a per-sequence rolling-window statistic (it visually implies
    a single group-level crossing, which is exactly the pooled-trajectory
    construct the manuscript's statistic is not). Bar heights come directly
    from the canonical plateau_rate values; see the notebook deep dive for
    individual released ΔI trajectories and median/IQR detail.
    """
    if not observatory._data_loaded['mirror_loop']:
        ax.text(0.5, 0.5, 'Mirror Loop Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 1: Mirror Loop - Plateau Rate')
        return

    if observatory.mirror_loop_analysis is None:
        observatory.analyze_mirror_loop()

    group_summary = observatory.mirror_loop_analysis['plateau']['group_summary']

    # Collect (model, condition) -> plateau_rate, preserving model order
    # of first appearance and a fixed condition order.
    models = []
    for (model, *_rest) in group_summary.keys():
        if model not in models:
            models.append(model)
    conditions = ['grounded', 'ungrounded']

    x = np.arange(len(models))
    width = 0.8 / len(conditions)
    colors = {'grounded': '#3b82f6', 'ungrounded': '#f59e0b'}

    for i, cond in enumerate(conditions):
        heights = [group_summary.get((m, cond), {}).get('plateau_rate', 0.0) for m in models]
        offset = (i - (len(conditions) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=cond,
                      color=colors.get(cond), alpha=0.8, edgecolor='black')
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.0%}',
                   ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Plateau Rate (τ=0.05, primary)', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Panel 1: Mirror Loop - Plateau Rate by Model × Condition',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


def _plot_confabulation_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """
    Plot Recursive Confabulation's manuscript-defined pooled intervention
    comparison: baseline / fact_table / belief_audit, N-weighted across
    models, derived from analyze_confabulation()'s
    pooled_intervention_comparison (itself derived from the canonical
    released model x arm table -- no bar height is hardcoded).

    grounding_pilot is deliberately NOT rendered as a fourth pooled bar --
    the source study's own pooled comparison excludes it, and its effect
    is model-specific (heterogeneous), not a pooled result. A brief text
    annotation notes this and points to the notebook deep dive for the
    full model x arm breakdown.
    """
    if not observatory._data_loaded['confabulation']:
        ax.text(0.5, 0.5, 'Confabulation Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 2: Recursive Confabulation - Pooled Intervention Comparison')
        return

    # Get analysis
    if observatory.confabulation_analysis is None:
        observatory.analyze_confabulation()

    pooled = observatory.confabulation_analysis.get('pooled_intervention_comparison', {})

    if not pooled:
        ax.text(0.5, 0.5, 'Pooled intervention comparison not available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 2: Recursive Confabulation - Pooled Intervention Comparison')
        return

    # Fixed manuscript order (not alphabetical): baseline -> fact_table -> belief_audit
    arm_order = ['baseline', 'fact_table', 'belief_audit']
    arms = [a for a in arm_order if a in pooled]
    rates = [pooled[a]['persist_rate'] for a in arms]

    colors = sns.color_palette("husl", len(arms))
    bars = ax.bar(arms, rates, color=colors, alpha=0.7, edgecolor='black')

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{rate:.1%}',
               ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Persistence Rate (N-weighted, pooled across models)', fontsize=10)
    ax.set_xlabel('Intervention Arm', fontsize=11)
    ax.set_ylim(0, max(rates) * 1.2 if rates else 1)

    # Grounding-confabulation heterogeneity annotation -- explicitly NOT a
    # pooled result, and explicitly about confab_rate (whether the model
    # confabulates at all), not persist_rate (whether a fabrication
    # persists -- the metric the three bars above use). See the METRIC
    # IDENTITY GUARD in CrossStudyAnalysis.analyze_confabulation().
    ax.text(0.5, -0.32,
           "grounding_pilot reduced CONFABULATION heterogeneously by model\n"
           "(not part of this pooled persistence comparison) -- see notebook\n"
           "deep dive for the full model × arm breakdown.",
           transform=ax.transAxes, ha='center', va='top', fontsize=8,
           style='italic', color='dimgray')

    ax.set_title('Panel 2: Recursive Confabulation - Pooled Intervention Comparison',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


def _plot_violation_state_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """
    Plot Violation State structured outcomes: contaminated vs control,
    across the raw response categories (policy_refusal, capability_refusal,
    image_success, terminal rate_limit).

    Bar heights come from raw_structured_outcomes (the released, as-observed
    data) — NOT from the published/historical aggregate. The published
    96.67%/0% figures are annotated as text, not used to size the bars, so
    the raw-vs-published distinction stays visually apparent.
    """
    if not observatory._data_loaded['violation_state']:
        ax.text(0.5, 0.5, 'Violation State Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Panel 3: Violation State - Structured Outcomes')
        return

    # Get analysis
    if observatory.violation_state_analysis is None:
        observatory.analyze_violation_state()

    structured = observatory.violation_state_analysis['structured']
    raw = structured['raw_structured_outcomes']
    published = structured['published_aggregate']

    categories = ['policy_refusal', 'capability_refusal', 'image_success', 'rate_limit']
    conditions = [c for c in ['contaminated', 'control'] if c in raw]

    x = np.arange(len(categories))
    width = 0.8 / max(len(conditions), 1)
    colors = {'contaminated': '#e74c3c', 'control': '#2ecc71'}

    for i, cond in enumerate(conditions):
        counts = raw[cond]['counts']
        heights = [counts.get(cat, 0) for cat in categories]
        offset = (i - (len(conditions) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=cond,
                      color=colors.get(cond, None), alpha=0.8, edgecolor='black')
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                       str(h), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Policy\nRefusal', 'Capability\nRefusal', 'Image\nSuccess', 'Rate Limit\n(terminal)'],
                       fontsize=9)
    ax.set_ylabel('Count (raw structured outcomes)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    # Annotate the published/historical aggregate separately — text only,
    # does not drive bar heights.
    annotation_lines = ["Published/historical aggregate:"]
    for cond in conditions:
        p = published[cond]
        annotation_lines.append(f"  {cond}: {p['refused']}/{p['n']} = {p['refusal_rate']:.2%}")
    ax.text(0.98, 0.98, "\n".join(annotation_lines),
           transform=ax.transAxes, ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_title('Panel 3: Violation State - Raw Structured Outcomes', fontsize=12, fontweight='bold')


def _plot_echo_chamber_panel(observatory: 'CrossStudyAnalysis', ax: plt.Axes):
    """
    Plot Echo Chamber Zero GR/SRI/RE trajectories.

    NONCANONICAL / STANDALONE: this panel is not part of the canonical
    Observatory figure (plot_four_panel_comparison draws only the three
    canonical studies). Retained for opt-in/provenance use only; call it
    directly against an observatory that has loaded Echo Chamber data via
    load_all_studies(include_echo_chamber=True).
    """
    if not observatory._data_loaded['echo_chamber']:
        ax.text(0.5, 0.5, 'Echo Chamber Data Not Available',
                ha='center', va='center', fontsize=12)
        ax.set_title('Echo Chamber Zero - Percolation (noncanonical)')
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
        ax.set_title('Echo Chamber Zero - Percolation Metrics (noncanonical)', fontsize=12, fontweight='bold')
        return

    # Aggregate by step
    agg_data = data.groupby('step').agg(agg_spec).reset_index()

    steps = agg_data['step']

    if 'GR' in data.columns and agg_data['GR'].notna().any():
        ax.plot(steps, agg_data['GR'], 'o-', label='GR (Groundedness Ratio)',
               linewidth=2, markersize=4, alpha=0.8)

    if 'SRI' in data.columns and agg_data['SRI'].notna().any():
        ax.plot(steps, agg_data['SRI'], 's-', label='SRI (Synthetic Recurrence Index)',
               linewidth=2, markersize=4, alpha=0.8)

    if 'RE' in data.columns and agg_data['RE'].notna().any():
        ax.plot(steps, agg_data['RE'], '^-', label='RE (Referential Entropy)',
               linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Echo Chamber Zero - Percolation Metrics (noncanonical)', fontsize=12, fontweight='bold')
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

    Uses the released `edit_change` column directly (not a
    delta_i_edit_distance recomputation from response text -- a direct
    numerical check established that recomputation does not reproduce the
    released measurement). Marks this individual sequence's own
    manuscript-defined plateau iteration (tau=0.05, rolling-3-step), if
    any, computed via detect_sequence_plateau() on this single released
    sequence -- not a pooled/group statistic.

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

    from ..metrics import detect_sequence_plateau

    iterations = seq_data['iteration'].values
    responses = seq_data['response'].values
    delta_i = seq_data[['iteration', 'edit_change']].dropna()

    # Plot released ΔI
    ax1.plot(delta_i['iteration'], delta_i['edit_change'], 'o-', linewidth=2, markersize=6)

    plateau_iter = detect_sequence_plateau(seq_data, tau=0.05, window=3)
    if plateau_iter is not None:
        ax1.axvline(plateau_iter, color='red', linestyle='--', alpha=0.7,
                   label=f'Plateau (τ=0.05) at iter {plateau_iter}')
        ax1.legend(loc='upper right', fontsize=9)

    ax1.set_ylabel('ΔI (released edit_change)', fontsize=11)
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
