"""
Recursive Confabulation Study Data Importer

Loads aggregate metrics from the Recursive Confabulation study.
This study publishes IRR-validated statistics, not raw conversation sequences.
This importer is READ ONLY - it does not modify source data.

NOTE: RC publishes aggregate metrics (confab rates, persistence rates) per model
and intervention arm, not per-turn sequences. We compute a summary persistence
metric per model for cross-study comparison.
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


class ConfabulationImporter:
    """
    Importer for Recursive Confabulation study data.

    This study examines fabrication persistence across conversational turns
    with different intervention strategies. The public repo exposes aggregate
    statistics per model and intervention arm, not raw sequences.

    We load summary_by_model_arm.csv and compute an aggregate persistence rate
    per model for inclusion in cross-study comparisons.

    Persistence Rate Metric:
    ------------------------
    The confab_persistence_rate is derived from the summary_by_model_arm.csv file,
    which contains per-model, per-intervention-arm statistics including:
    - confab_rate: Initial confabulation rate
    - persist_rate: Persistence rate (how often confabulations persist)

    We compute the average persistence rate across all intervention arms for each model.
    This gives a single scalar metric per model suitable for cross-study comparison.
    """

    # GitHub URL for aggregate summary data
    SUMMARY_URL = (
        "https://raw.githubusercontent.com/Course-Correct-Labs/"
        "recursive-confabulation/main/data/summary_by_model_arm.csv"
    )

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the Confabulation importer.

        Args:
            data_path: Explicit path to summary CSV. If None, will fetch from GitHub.
        """
        self.data_path = data_path
        self.summary_df: Optional[pd.DataFrame] = None  # Per-model summary
        self.data_source: Optional[str] = None

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load aggregate confabulation data from RC study.

        Returns per-model persistence metrics suitable for cross-study comparison.

        Returns:
            DataFrame with columns: model, confab_persistence_rate
            Or None if data unavailable
        """
        # Try explicit path first
        if self.data_path and os.path.exists(self.data_path):
            print(f"[ConfabulationImporter] Loading from explicit path: {self.data_path}")
            try:
                df = pd.read_csv(self.data_path)
                self.data_source = f"explicit_path:{self.data_path}"
                return self._compute_model_summary(df)
            except Exception as e:
                print(f"[ConfabulationImporter] Failed to load from {self.data_path}: {e}")

        # Try GitHub fallback
        if REQUESTS_AVAILABLE:
            try:
                print(f"[ConfabulationImporter] Trying GitHub fallback: {self.SUMMARY_URL}")
                resp = requests.get(self.SUMMARY_URL, timeout=15)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                self.data_source = f"remote:{self.SUMMARY_URL}"
                return self._compute_model_summary(df)
            except Exception as e:
                print(f"[ConfabulationImporter] GitHub fallback failed: {e}")

        # Nothing worked
        self.data_source = "not_loaded"
        print("[ConfabulationImporter] Data not available from any source")
        return None

    def _compute_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-model persistence rate from summary_by_model_arm.csv.

        The input CSV has columns: model, arm, n, confab_rate, persist_rate, ...
        We compute the mean persistence rate across all arms for each model.

        Args:
            df: Raw summary_by_model_arm DataFrame

        Returns:
            DataFrame with columns: model, confab_persistence_rate
        """
        # Validate required columns exist
        required_cols = ['model', 'persist_rate']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in summary CSV: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Group by model and compute mean persistence rate across all arms
        model_summary = df.groupby('model').agg({
            'persist_rate': 'mean',
            'confab_rate': 'mean',  # Also include confab rate for reference
            'n': 'sum'  # Total sample size
        }).reset_index()

        # Rename to match Observatory naming convention
        model_summary = model_summary.rename(columns={
            'persist_rate': 'confab_persistence_rate',
        })

        self.summary_df = model_summary
        print(f"[ConfabulationImporter] ✓ Loaded aggregate metrics for {len(model_summary)} models")

        return model_summary

    def get_data_info(self) -> Dict[str, Any]:
        """Get summary information about the loaded data."""
        if self.summary_df is None:
            return {"status": "not loaded"}

        info = {
            "status": "loaded (aggregate)",
            "num_models": len(self.summary_df),
            "models": sorted(self.summary_df['model'].tolist()),
            "mean_persistence_rate": float(self.summary_df['confab_persistence_rate'].mean()),
            "data_source": self.data_source,
        }

        return info
