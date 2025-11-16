"""
Recursive Confabulation Study Data Importer

Loads IRR-validated conversation data from the Recursive Confabulation study.
This importer is READ ONLY - it does not modify source data.
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
    with different intervention strategies.
    """

    DEFAULT_FILENAME = "confabulation_results.csv"

    # GitHub fallback URL for automatic data fetching
    FALLBACK_URL = (
        "https://raw.githubusercontent.com/Course-Correct-Labs/"
        "recursive-confabulation/main/data/confabulation_results.csv"
    )

    DEFAULT_SEARCH_PATHS = [
        "confabulation_results.csv",
        "../recursive-confabulation/confabulation_results.csv",
        "../recursive-confabulation/results/confabulation_results.csv",
        "../recursive-confabulation/data/confabulation_results.csv",
        "data/confabulation/confabulation_results.csv",
    ]

    REQUIRED_COLUMNS = {
        "conversation_id": ["conversation_id", "session_id", "run_id"],
        "turn_number": ["turn_number", "turn", "iteration"],
        "content": ["content", "message", "text", "response"],
        "role": ["role", "speaker"],
        "fabrication_present": ["fabrication_present", "is_fabrication", "fabricated"],
    }

    OPTIONAL_COLUMNS = {
        "intervention_arm": ["intervention_arm", "condition", "treatment"],
        "model": ["model", "model_name"],
        "timestamp": ["timestamp", "created_at", "time"],
        "annotator_confidence": ["confidence", "annotator_confidence"],
    }

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the Confabulation importer.

        Args:
            data_path: Explicit path to the data file. If None, will auto-discover.
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self._column_mapping: Dict[str, str] = {}
        self.data_source: Optional[str] = None  # Track where data was loaded from

    def _find_data_file(self) -> Optional[str]:
        """Find the data file by searching common locations."""
        if self.data_path:
            if os.path.exists(self.data_path):
                self.data_source = f"explicit_path:{self.data_path}"
                return self.data_path
            return None

        env_path = os.getenv("CONFABULATION_DATA_PATH")
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

        for standard_name, variants in self.OPTIONAL_COLUMNS.items():
            actual_col = self._normalize_column_name(df, standard_name, variants)
            if actual_col:
                normalized_cols[actual_col] = standard_name
            else:
                warnings.warn(f"Optional column '{standard_name}' not found")

        df_normalized = df.rename(columns=normalized_cols)

        # Validate data types
        df_normalized['turn_number'] = pd.to_numeric(df_normalized['turn_number'], errors='coerce')

        # Convert fabrication_present to boolean if needed
        if df_normalized['fabrication_present'].dtype == 'object':
            # Handle various string representations
            df_normalized['fabrication_present'] = df_normalized['fabrication_present'].map({
                'true': True, 'True': True, '1': True, 1: True,
                'false': False, 'False': False, '0': False, 0: False,
                True: True, False: False
            })

        # Check for nulls in required columns
        for col in self.REQUIRED_COLUMNS.keys():
            null_count = df_normalized[col].isnull().sum()
            if null_count > 0:
                warnings.warn(f"Column '{col}' has {null_count} null values")

        return df_normalized

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load and validate the Confabulation data.

        Tries local paths first, then falls back to GitHub if needed.

        Returns:
            Normalized DataFrame with standard column names, or None if data unavailable
        """
        # Build list of candidates to try
        candidates = []

        # 1) Explicit path
        if self.data_path:
            candidates.append(("explicit_path", self.data_path))

        # 2) Environment variable
        env_path = os.getenv("CONFABULATION_DATA_PATH")
        if env_path:
            candidates.append(("env:CONFABULATION_DATA_PATH", env_path))

        # 3) Local default paths
        for local_path in self.DEFAULT_SEARCH_PATHS:
            candidates.append(("local", local_path))

        # Try all local/explicit candidates
        for source_type, path in candidates:
            if path and os.path.exists(path):
                print(f"[ConfabulationImporter] Loading data from {source_type}: {path}")
                try:
                    df = pd.read_csv(path)
                    if df is None or len(df) == 0:
                        print(f"[ConfabulationImporter] Data file is empty: {path}")
                        continue
                    self.data_source = f"{source_type}:{path}"
                    df_normalized = self._validate_and_normalize(df)
                    self.df = df_normalized
                    print(f"[ConfabulationImporter] ✓ Loaded {len(df_normalized)} rows from {source_type}")
                    return df_normalized
                except Exception as e:
                    print(f"[ConfabulationImporter] Failed to load from {path}: {e}")
                    continue

        # 4) GitHub raw fallback
        if REQUESTS_AVAILABLE:
            try:
                print(f"[ConfabulationImporter] Trying GitHub fallback: {self.FALLBACK_URL}")
                resp = requests.get(self.FALLBACK_URL, timeout=15)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                if df is None or len(df) == 0:
                    print("[ConfabulationImporter] Remote data file is empty")
                else:
                    self.data_source = f"remote:{self.FALLBACK_URL}"
                    df_normalized = self._validate_and_normalize(df)
                    self.df = df_normalized
                    print(f"[ConfabulationImporter] ✓ Loaded {len(df_normalized)} rows from GitHub")
                    return df_normalized
            except Exception as e:
                print(f"[ConfabulationImporter] GitHub fallback failed: {e}")

        # 5) Nothing worked
        self.data_source = "not_loaded"
        print("[ConfabulationImporter] Data not available from any source")
        return None

    def get_data_info(self) -> Dict[str, Any]:
        """Get summary information about the loaded data."""
        if self.df is None:
            return {"status": "not loaded"}

        fabrication_count = self.df['fabrication_present'].sum() if 'fabrication_present' in self.df else 0

        info = {
            "status": "loaded",
            "total_rows": len(self.df),
            "num_conversations": self.df['conversation_id'].nunique(),
            "fabrication_count": int(fabrication_count),
            "fabrication_rate": float(fabrication_count / len(self.df)) if len(self.df) > 0 else 0.0,
            "turn_range": (int(self.df['turn_number'].min()), int(self.df['turn_number'].max())),
            "column_mapping": self._column_mapping,
        }

        if 'intervention_arm' in self.df.columns:
            info['intervention_arms'] = sorted(self.df['intervention_arm'].unique().tolist())

        if 'model' in self.df.columns:
            info['models'] = sorted(self.df['model'].unique().tolist())

        return info

    def get_conversation(self, conversation_id: str) -> pd.DataFrame:
        """Get data for a specific conversation."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        conv_df = self.df[self.df['conversation_id'] == conversation_id].copy()
        conv_df = conv_df.sort_values('turn_number')

        return conv_df

    def get_fabricated_turns(self) -> pd.DataFrame:
        """Get all turns where fabrication is present."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.df[self.df['fabrication_present'] == True].copy()

    def get_by_intervention_arm(self, arm: str) -> pd.DataFrame:
        """Get all data for a specific intervention arm."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'intervention_arm' not in self.df.columns:
            raise ValueError("intervention_arm column not available in data")

        return self.df[self.df['intervention_arm'] == arm].copy()
