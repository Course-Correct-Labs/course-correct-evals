"""
Echo Chamber Study Data Importer

Loads simulation data from the Echo Chamber study.
This importer is READ ONLY - it does not modify source data.

IMPORTANT: This study includes PRECOMPUTED metrics (GR, SRI, RE) in the CSV.
We use these directly rather than reconstructing from raw data.
"""

import os
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class EchoChamberImporter:
    """
    Importer for Echo Chamber study data.

    This study examines belief convergence and radicalization in multi-agent systems.

    CRITICAL: The simulation_results.csv contains PRECOMPUTED metrics:
    - GR (Group Radicalization)
    - SRI (Self-Reinforcement Index)
    - RE (Reasoning Entropy)

    We use these precomputed values directly as ground truth.
    """

    DEFAULT_FILENAME = "simulation_results.csv"

    # GitHub fallback URL for automatic data fetching
    FALLBACK_URL = (
        "https://raw.githubusercontent.com/Course-Correct-Labs/"
        "echo-chamber-zero/main/data/simulation_results.csv"
    )

    DEFAULT_SEARCH_PATHS = [
        "simulation_results.csv",
        "../echo-chamber-zero/simulation_results.csv",
        "../echo-chamber-zero/results/simulation_results.csv",
        "../echo-chamber-zero/data/simulation_results.csv",
        "data/echo_chamber/simulation_results.csv",
    ]

    REQUIRED_COLUMNS = {
        "simulation_id": ["simulation_id", "run_id", "session_id"],
        "step": ["step", "iteration", "turn"],
    }

    # These are the precomputed metrics we rely on
    METRIC_COLUMNS = {
        "GR": ["GR", "group_radicalization", "gr"],
        "SRI": ["SRI", "self_reinforcement_index", "sri"],
        "RE": ["RE", "reasoning_entropy", "re"],
    }

    OPTIONAL_COLUMNS = {
        "agent_id": ["agent_id", "agent"],
        "belief_state": ["belief_state", "message", "content", "output"],
        "model": ["model", "model_name"],
        "initial_prompt": ["initial_prompt", "seed_prompt"],
        "convergence_reached": ["convergence_reached", "converged"],
        "timestamp": ["timestamp", "created_at", "time"],
    }

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the Echo Chamber importer.

        Args:
            data_path: Explicit path to the data file. If None, will auto-discover.
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self._column_mapping: Dict[str, str] = {}
        self._has_precomputed_metrics = False
        self.data_source: Optional[str] = None  # Track where data was loaded from

    def _find_data_file(self) -> Optional[str]:
        """Find the data file by searching common locations."""
        if self.data_path:
            if os.path.exists(self.data_path):
                self.data_source = f"explicit_path:{self.data_path}"
                return self.data_path
            return None

        env_path = os.getenv("ECHO_CHAMBER_DATA_PATH")
        if env_path and os.path.exists(env_path):
            self.data_source = f"env_var:{env_path}"
            return env_path

        for search_path in self.DEFAULT_SEARCH_PATHS:
            if os.path.exists(search_path):
                self.data_source = f"local:{search_path}"
                return search_path

        return None

    def _normalize_column_name(self, df: pd.DataFrame, standard_name: str, variants: List[str]) -> Optional[str]:
        """Find a column by checking variants (case-insensitive)."""
        df_columns_lower = {col.lower(): col for col in df.columns}

        for variant in variants:
            if variant.lower() in df_columns_lower:
                actual_col = df_columns_lower[variant.lower()]
                self._column_mapping[standard_name] = actual_col
                return actual_col

        return None

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the DataFrame schema."""
        normalized_cols = {}
        missing_columns = []

        # Check required columns
        for standard_name, variants in self.REQUIRED_COLUMNS.items():
            actual_col = self._normalize_column_name(df, standard_name, variants)
            if actual_col is None:
                missing_columns.append(f"{standard_name} (tried: {', '.join(variants)})")
            else:
                normalized_cols[actual_col] = standard_name

        if missing_columns:
            raise ValueError(
                f"Missing required columns:\n" +
                "\n".join(f"  - {col}" for col in missing_columns) +
                f"\n\nAvailable columns: {', '.join(df.columns)}"
            )

        # Check for precomputed metrics (these are critical!)
        metric_columns_found = {}
        missing_metrics = []

        for metric_name, variants in self.METRIC_COLUMNS.items():
            actual_col = self._normalize_column_name(df, metric_name, variants)
            if actual_col is None:
                missing_metrics.append(f"{metric_name} (tried: {', '.join(variants)})")
            else:
                normalized_cols[actual_col] = metric_name
                metric_columns_found[metric_name] = True

        # Warn if metrics are missing (they should be present!)
        if missing_metrics:
            warnings.warn(
                "IMPORTANT: Precomputed metrics not found:\n" +
                "\n".join(f"  - {m}" for m in missing_metrics) +
                "\n\nThese metrics should be present in the data. " +
                "Analysis will be limited without them."
            )
        else:
            self._has_precomputed_metrics = True
            print("✓ Found all precomputed metrics (GR, SRI, RE)")

        # Check optional columns
        for standard_name, variants in self.OPTIONAL_COLUMNS.items():
            actual_col = self._normalize_column_name(df, standard_name, variants)
            if actual_col:
                normalized_cols[actual_col] = standard_name

        # Rename columns to standard names
        df_normalized = df.rename(columns=normalized_cols)

        # Validate data types
        df_normalized['step'] = pd.to_numeric(df_normalized['step'], errors='coerce')

        # Validate metric columns are numeric
        for metric in ['GR', 'SRI', 'RE']:
            if metric in df_normalized.columns:
                df_normalized[metric] = pd.to_numeric(df_normalized[metric], errors='coerce')
                null_count = df_normalized[metric].isnull().sum()
                if null_count > 0:
                    warnings.warn(f"Metric '{metric}' has {null_count} null values")

        # Convert convergence_reached to boolean if present
        if 'convergence_reached' in df_normalized.columns:
            if df_normalized['convergence_reached'].dtype == 'object':
                df_normalized['convergence_reached'] = df_normalized['convergence_reached'].map({
                    'true': True, 'True': True, '1': True, 1: True,
                    'false': False, 'False': False, '0': False, 0: False,
                    True: True, False: False
                })

        return df_normalized

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load and validate the Echo Chamber data.

        Tries local paths first, then falls back to GitHub if needed.

        Returns:
            Normalized DataFrame with standard column names and precomputed metrics, or None if data unavailable
        """
        # Build list of candidates to try
        candidates = []

        # 1) Explicit path
        if self.data_path:
            candidates.append(("explicit_path", self.data_path))

        # 2) Environment variable
        env_path = os.getenv("ECHO_CHAMBER_DATA_PATH")
        if env_path:
            candidates.append(("env:ECHO_CHAMBER_DATA_PATH", env_path))

        # 3) Local default paths
        for local_path in self.DEFAULT_SEARCH_PATHS:
            candidates.append(("local", local_path))

        # Try all local/explicit candidates
        for source_type, path in candidates:
            if path and os.path.exists(path):
                print(f"[EchoChamberImporter] Loading data from {source_type}: {path}")
                try:
                    df = pd.read_csv(path)
                    if df is None or len(df) == 0:
                        print(f"[EchoChamberImporter] Data file is empty: {path}")
                        continue
                    self.data_source = f"{source_type}:{path}"
                    df_normalized = self._validate_and_normalize(df)
                    self.df = df_normalized
                    print(f"[EchoChamberImporter] ✓ Loaded {len(df_normalized)} rows from {source_type}")
                    return df_normalized
                except Exception as e:
                    print(f"[EchoChamberImporter] Failed to load from {path}: {e}")
                    continue

        # 4) GitHub raw fallback
        if REQUESTS_AVAILABLE:
            try:
                print(f"[EchoChamberImporter] Trying GitHub fallback: {self.FALLBACK_URL}")
                resp = requests.get(self.FALLBACK_URL, timeout=15)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                if df is None or len(df) == 0:
                    print("[EchoChamberImporter] Remote data file is empty")
                else:
                    self.data_source = f"remote:{self.FALLBACK_URL}"
                    df_normalized = self._validate_and_normalize(df)
                    self.df = df_normalized
                    print(f"[EchoChamberImporter] ✓ Loaded {len(df_normalized)} rows from GitHub")
                    return df_normalized
            except Exception as e:
                print(f"[EchoChamberImporter] GitHub fallback failed: {e}")

        # 5) Nothing worked
        self.data_source = "not_loaded"
        print("[EchoChamberImporter] Data not available from any source")
        return None

    def get_data_info(self) -> Dict[str, Any]:
        """Get summary information about the loaded data."""
        if self.df is None:
            return {"status": "not loaded"}

        info = {
            "status": "loaded",
            "total_rows": len(self.df),
            "num_simulations": self.df['simulation_id'].nunique(),
            "step_range": (int(self.df['step'].min()), int(self.df['step'].max())),
            "has_precomputed_metrics": self._has_precomputed_metrics,
            "column_mapping": self._column_mapping,
        }

        # Add metric statistics if available
        if 'GR' in self.df.columns:
            info['GR_range'] = (float(self.df['GR'].min()), float(self.df['GR'].max()))
            info['GR_mean'] = float(self.df['GR'].mean())

        if 'SRI' in self.df.columns:
            info['SRI_range'] = (float(self.df['SRI'].min()), float(self.df['SRI'].max()))
            info['SRI_mean'] = float(self.df['SRI'].mean())

        if 'RE' in self.df.columns:
            info['RE_range'] = (float(self.df['RE'].min()), float(self.df['RE'].max()))
            info['RE_mean'] = float(self.df['RE'].mean())

        if 'convergence_reached' in self.df.columns:
            convergence_count = self.df.groupby('simulation_id')['convergence_reached'].any().sum()
            info['simulations_converged'] = int(convergence_count)

        if 'agent_id' in self.df.columns:
            info['num_agents'] = self.df['agent_id'].nunique()

        if 'model' in self.df.columns:
            info['models'] = sorted(self.df['model'].unique().tolist())

        return info

    def get_simulation(self, simulation_id: str) -> pd.DataFrame:
        """Get data for a specific simulation."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        sim_df = self.df[self.df['simulation_id'] == simulation_id].copy()
        sim_df = sim_df.sort_values('step')

        return sim_df

    def get_convergent_simulations(self) -> pd.DataFrame:
        """Get all simulations that reached convergence."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'convergence_reached' not in self.df.columns:
            raise ValueError("convergence_reached column not available in data")

        # Get simulation IDs that have convergence_reached = True at any step
        converged_sims = self.df[self.df['convergence_reached'] == True]['simulation_id'].unique()

        return self.df[self.df['simulation_id'].isin(converged_sims)].copy()

    def get_metrics_over_time(self, simulation_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get metrics (GR, SRI, RE) over time.

        Args:
            simulation_id: If specified, get metrics for one simulation.
                          Otherwise, aggregate across all simulations.

        Returns:
            DataFrame with step and metric columns
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if not self._has_precomputed_metrics:
            warnings.warn("Precomputed metrics not available in data")
            return pd.DataFrame()

        if simulation_id:
            data = self.get_simulation(simulation_id)
        else:
            data = self.df

        # Extract relevant columns
        metric_cols = ['step']
        for metric in ['GR', 'SRI', 'RE']:
            if metric in data.columns:
                metric_cols.append(metric)

        if simulation_id:
            result = data[metric_cols].copy()
        else:
            # Aggregate across simulations (mean per step)
            result = data.groupby('step')[metric_cols[1:]].mean().reset_index()

        return result
