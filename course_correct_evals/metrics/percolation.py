"""
Echo Chamber Zero Percolation Metrics

NONCANONICAL / OPT-IN: retained for provenance/backward compatibility.
Echo Chamber Zero is an independent Course Correct Labs theoretical/
systemic research project (synthetic epistemic drift modeled as
percolation on a provenance graph), not part of the canonical Reasoning
Stability Observatory evaluation set (Mirror Loop, Recursive
Confabulation, Violation State), and not a peer behavioral model
evaluation. Used only when explicitly opted into via
CrossStudyAnalysis.load_all_studies(include_echo_chamber=True).

IMPORTANT: This module primarily uses PRECOMPUTED metrics (GR, SRI, RE) from
the simulation_results.csv. NetworkX reconstruction is optional and for future data only.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import warnings


def analyze_echo_metrics(
    df: pd.DataFrame,
    simulation_id_col: str = 'simulation_id',
    step_col: str = 'step',
    gr_col: str = 'GR',
    sri_col: str = 'SRI',
    re_col: str = 'RE'
) -> Dict[str, Any]:
    """
    Analyze precomputed echo chamber metrics (GR, SRI, RE).

    IMPORTANT: Uses precomputed metrics directly from data.

    Args:
        df: DataFrame with simulation data
        simulation_id_col: Column for simulation ID
        step_col: Column for step/iteration
        gr_col: Column for Groundedness Ratio metric
        sri_col: Column for Synthetic Recurrence Index
        re_col: Column for Referential Entropy

    Returns:
        Dictionary with metric analysis
    """
    # Verify required columns exist
    required_cols = [simulation_id_col, step_col]
    metric_cols = []

    for col_name, col_var in [(gr_col, 'GR'), (sri_col, 'SRI'), (re_col, 'RE')]:
        if col_name in df.columns:
            metric_cols.append(col_name)
        else:
            warnings.warn(f"Metric column '{col_name}' not found in data")

    if not metric_cols:
        raise ValueError("No metric columns (GR, SRI, RE) found in data")

    results = {
        'total_simulations': df[simulation_id_col].nunique(),
        'total_steps': len(df),
        'step_range': (int(df[step_col].min()), int(df[step_col].max())),
        'metrics': {},
    }

    # Analyze each metric
    for metric_col in metric_cols:
        metric_stats = {
            'mean': float(df[metric_col].mean()),
            'std': float(df[metric_col].std()),
            'min': float(df[metric_col].min()),
            'max': float(df[metric_col].max()),
            'median': float(df[metric_col].median()),
        }

        # Calculate trajectory statistics (change over time)
        trajectories = []
        for sim_id, sim_df in df.groupby(simulation_id_col):
            sim_df = sim_df.sort_values(step_col)
            values = sim_df[metric_col].values

            if len(values) > 1:
                # Calculate trend (positive = increasing)
                trend = values[-1] - values[0]
                trajectories.append(trend)

        if trajectories:
            metric_stats['trend_mean'] = float(np.mean(trajectories))
            metric_stats['trend_std'] = float(np.std(trajectories))
            metric_stats['increasing_count'] = int(sum(1 for t in trajectories if t > 0))
            metric_stats['decreasing_count'] = int(sum(1 for t in trajectories if t < 0))

        results['metrics'][metric_col] = metric_stats

    return results


def detect_threshold_crossing(
    df: pd.DataFrame,
    metric_col: str,
    threshold: float,
    simulation_id_col: str = 'simulation_id',
    step_col: str = 'step',
    direction: str = 'above'
) -> pd.DataFrame:
    """
    Detect when a metric crosses a threshold.

    Args:
        df: DataFrame with simulation data
        metric_col: Column containing the metric (e.g., 'GR', 'SRI')
        threshold: Threshold value
        simulation_id_col: Column for simulation ID
        step_col: Column for step number
        direction: 'above' or 'below' - direction of crossing

    Returns:
        DataFrame with crossing events (simulation_id, crossing_step, metric_value)
    """
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found")

    crossings = []

    for sim_id, sim_df in df.groupby(simulation_id_col):
        sim_df = sim_df.sort_values(step_col)

        steps = sim_df[step_col].values
        values = sim_df[metric_col].values

        # Find first crossing
        for i in range(len(values)):
            if direction == 'above' and values[i] > threshold:
                crossings.append({
                    'simulation_id': sim_id,
                    'crossing_step': steps[i],
                    'metric': metric_col,
                    'threshold': threshold,
                    'value': values[i],
                })
                break
            elif direction == 'below' and values[i] < threshold:
                crossings.append({
                    'simulation_id': sim_id,
                    'crossing_step': steps[i],
                    'metric': metric_col,
                    'threshold': threshold,
                    'value': values[i],
                })
                break

    return pd.DataFrame(crossings)


def calculate_convergence_statistics(
    df: pd.DataFrame,
    simulation_id_col: str = 'simulation_id',
    step_col: str = 'step',
    convergence_col: Optional[str] = 'convergence_reached'
) -> Dict[str, Any]:
    """
    Calculate convergence statistics from simulations.

    Args:
        df: DataFrame with simulation data
        simulation_id_col: Column for simulation ID
        step_col: Column for step number
        convergence_col: Column indicating convergence (if available)

    Returns:
        Dictionary with convergence statistics
    """
    stats = {
        'total_simulations': df[simulation_id_col].nunique(),
    }

    if convergence_col and convergence_col in df.columns:
        # Find simulations that converged
        converged_sims = df[df[convergence_col] == True][simulation_id_col].unique()
        stats['converged_count'] = len(converged_sims)
        stats['convergence_rate'] = len(converged_sims) / stats['total_simulations']

        # Find step at which convergence occurred
        convergence_steps = []
        for sim_id in converged_sims:
            sim_df = df[df[simulation_id_col] == sim_id].sort_values(step_col)
            conv_step = sim_df[sim_df[convergence_col] == True][step_col].min()
            if pd.notna(conv_step):
                convergence_steps.append(conv_step)

        if convergence_steps:
            stats['convergence_step_mean'] = float(np.mean(convergence_steps))
            stats['convergence_step_median'] = float(np.median(convergence_steps))
            stats['convergence_step_std'] = float(np.std(convergence_steps))
            stats['convergence_step_min'] = int(np.min(convergence_steps))
            stats['convergence_step_max'] = int(np.max(convergence_steps))

    return stats


def analyze_metric_trajectories(
    df: pd.DataFrame,
    simulation_id_col: str = 'simulation_id',
    step_col: str = 'step',
    metric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze metric trajectories for each simulation.

    Args:
        df: DataFrame with simulation data
        simulation_id_col: Column for simulation ID
        step_col: Column for step number
        metric_cols: List of metric columns to analyze (default: ['GR', 'SRI', 'RE'])

    Returns:
        DataFrame with trajectory analysis per simulation
    """
    if metric_cols is None:
        metric_cols = ['GR', 'SRI', 'RE']

    # Filter to metrics that exist
    metric_cols = [col for col in metric_cols if col in df.columns]

    if not metric_cols:
        raise ValueError("No valid metric columns found in data")

    results = []

    for sim_id, sim_df in df.groupby(simulation_id_col):
        sim_df = sim_df.sort_values(step_col)

        trajectory = {
            'simulation_id': sim_id,
            'num_steps': len(sim_df),
        }

        for metric_col in metric_cols:
            values = sim_df[metric_col].values

            # Calculate trajectory statistics
            trajectory[f'{metric_col}_initial'] = float(values[0])
            trajectory[f'{metric_col}_final'] = float(values[-1])
            trajectory[f'{metric_col}_mean'] = float(np.mean(values))
            trajectory[f'{metric_col}_std'] = float(np.std(values))
            trajectory[f'{metric_col}_max'] = float(np.max(values))
            trajectory[f'{metric_col}_min'] = float(np.min(values))
            trajectory[f'{metric_col}_trend'] = float(values[-1] - values[0])

            # Monotonicity check
            is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
            is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))

            trajectory[f'{metric_col}_monotonic_increasing'] = is_increasing
            trajectory[f'{metric_col}_monotonic_decreasing'] = is_decreasing

        results.append(trajectory)

    return pd.DataFrame(results)


# Optional: NetworkX-based reconstruction (for future data, not current analysis)

def reconstruct_agent_network(
    df: pd.DataFrame,
    simulation_id: str,
    simulation_id_col: str = 'simulation_id',
    agent_col: str = 'agent_id',
    step_col: str = 'step'
) -> Optional[Any]:
    """
    OPTIONAL: Reconstruct agent interaction network.

    This is for future data analysis only. Current analysis should use
    precomputed GR/SRI/RE metrics directly.

    Args:
        df: DataFrame with simulation data
        simulation_id: Simulation to reconstruct
        simulation_id_col: Column for simulation ID
        agent_col: Column for agent ID
        step_col: Column for step number

    Returns:
        NetworkX graph (if networkx available), or None
    """
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("NetworkX not available for network reconstruction")
        return None

    # This is a placeholder - actual reconstruction would depend on
    # the specific interaction data available
    sim_df = df[df[simulation_id_col] == simulation_id].sort_values(step_col)

    if agent_col not in sim_df.columns:
        warnings.warn(f"Agent column '{agent_col}' not found")
        return None

    G = nx.Graph()

    # Add agents as nodes
    agents = sim_df[agent_col].unique()
    G.add_nodes_from(agents)

    # Note: Edge creation would require interaction data
    # This is just a skeleton for future implementation

    return G
