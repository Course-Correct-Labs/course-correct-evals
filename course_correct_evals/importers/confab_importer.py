"""
Recursive Confabulation Study Data Importer

Loads the released model×arm summary table from the Recursive Confabulation
study. This study publishes IRR-validated per-model, per-intervention-arm
statistics, not raw conversation sequences.
This importer is READ ONLY - it does not modify source data.

CANONICAL REPRESENTATION: the released summary_by_model_arm.csv (one row per
(model, arm), 3 models x 4 arms = 12 rows) is the single canonical source
representation. This importer preserves it unmodified -- it does NOT group
across model or arm, and does NOT rename persist_rate into a collapsed
metric. Any pooled or model-specific views (e.g. the manuscript's N-weighted
three-arm intervention comparison, or the model-specific grounding_pilot
finding) are computed downstream, from this same preserved table, in
CrossStudyAnalysis.analyze_confabulation() -- not here.
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
    under four intervention arms (baseline, fact_table, belief_audit,
    grounding_pilot), for 3 models. The public repo exposes per-model,
    per-intervention-arm statistics (summary_by_model_arm.csv), not raw
    per-turn sequences.

    load_data() returns the released model x arm table UNMODIFIED (one row
    per (model, arm), 12 rows total) -- no grouping across model or arm, no
    collapsed/renamed metric. This is the single canonical source
    representation; downstream analysis (CrossStudyAnalysis) derives the
    manuscript's pooled intervention comparison and the model-specific
    grounding_pilot finding from this same table.
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
        Load the released model x arm summary table from the RC study.

        Returns the table UNMODIFIED: one row per (model, arm), preserving
        every released column (model, arm, n, confab_rate, confab_ci,
        persist_rate, persist_ci, latency_mean, latency_std, blame_mean,
        blame_std). No aggregation across model or arm is performed here.

        Returns:
            DataFrame with the released model x arm schema, or None if
            data unavailable.
        """
        # Try explicit path first
        if self.data_path and os.path.exists(self.data_path):
            print(f"[ConfabulationImporter] Loading from explicit path: {self.data_path}")
            try:
                df = pd.read_csv(self.data_path)
                self.data_source = f"explicit_path:{self.data_path}"
                return self._validate_model_arm_table(df)
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
                return self._validate_model_arm_table(df)
            except Exception as e:
                print(f"[ConfabulationImporter] GitHub fallback failed: {e}")

        # Nothing worked
        self.data_source = "not_loaded"
        print("[ConfabulationImporter] Data not available from any source")
        return None

    def _validate_model_arm_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and return the released model x arm table UNMODIFIED.

        The input CSV has columns: model, arm, n, confab_rate, confab_ci,
        persist_rate, persist_ci, latency_mean, latency_std, blame_mean,
        blame_std -- one row per (model, arm). This is the single canonical
        source representation: no grouping, no renaming, no collapsing
        across model or arm happens here. Pooled/model-specific downstream
        views are computed elsewhere from this same preserved table.

        Args:
            df: Raw summary_by_model_arm DataFrame

        Returns:
            The same DataFrame, validated, with columns/rows unmodified.
        """
        # Validate required columns exist
        required_cols = ['model', 'arm', 'n', 'persist_rate']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in summary CSV: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        self.summary_df = df
        print(f"[ConfabulationImporter] ✓ Loaded released model x arm table: "
              f"{len(df)} rows ({df['model'].nunique()} models x {df['arm'].nunique()} arms)")

        return df

    def get_data_info(self) -> Dict[str, Any]:
        """Get summary information about the loaded data.

        Deliberately does NOT report any across-arm persistence scalar --
        the released table has no single "the persistence rate"; see
        CrossStudyAnalysis.analyze_confabulation() for the manuscript's
        pooled comparison and the model-specific grounding finding.
        """
        if self.summary_df is None:
            return {"status": "not loaded"}

        info = {
            "status": "loaded (model x arm table)",
            "num_rows": len(self.summary_df),
            "models": sorted(self.summary_df['model'].unique().tolist()),
            "arms": sorted(self.summary_df['arm'].unique().tolist()),
            "data_source": self.data_source,
        }

        return info
