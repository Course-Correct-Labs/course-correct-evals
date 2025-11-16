"""
Violation State Study Data Importer

Loads conversation data from the Violation State study.
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


class ViolationStateImporter:
    """
    Importer for Violation State study data.

    This study examines how violation requests and refusals contaminate
    subsequent conversational turns.
    """

    DEFAULT_FILENAME = "parsed_turns.csv"

    # GitHub fallback URL for automatic data fetching
    FALLBACK_URL = (
        "https://raw.githubusercontent.com/Course-Correct-Labs/"
        "violation-state/main/data/parsed_turns.csv"
    )

    DEFAULT_SEARCH_PATHS = [
        "parsed_turns.csv",
        "violation_results.csv",
        "../violation-state/parsed_turns.csv",
        "../violation-state/results/parsed_turns.csv",
        "../violation-state/data/parsed_turns.csv",
        "data/violation_state/parsed_turns.csv",
    ]

    REQUIRED_COLUMNS = {
        "conversation_id": ["conversation_id", "session_id", "run_id"],
        "turn_number": ["turn_number", "turn", "iteration"],
        "content": ["content", "message", "text", "response"],
    }

    OPTIONAL_COLUMNS = {
        "violation_type": ["violation_type", "violation", "request_type"],
        "model": ["model", "model_name"],
        "response_type": ["response_type", "classification", "category"],
        "contamination_detected": ["contamination_detected", "contaminated", "is_contaminated"],
        "timestamp": ["timestamp", "created_at", "time"],
        "role": ["role", "speaker"],
    }

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the Violation State importer.

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

        env_path = os.getenv("VIOLATION_STATE_DATA_PATH")
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

        # Convert contamination_detected to boolean if needed
        if 'contamination_detected' in df_normalized.columns:
            if df_normalized['contamination_detected'].dtype == 'object':
                df_normalized['contamination_detected'] = df_normalized['contamination_detected'].map({
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
        Load and validate the Violation State data.

        Tries local paths first, then falls back to GitHub if needed.

        Returns:
            Normalized DataFrame with standard column names, or None if data unavailable
        """
        df = None

        # Try to find data file locally
        file_path = self._find_data_file()

        if file_path:
            print(f"Loading Violation State data from: {file_path}")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"[ViolationStateImporter] Failed to read local file: {e}")
                df = None

        # If not found locally, try GitHub fallback
        if df is None and REQUESTS_AVAILABLE:
            print(f"[ViolationStateImporter] Trying GitHub fallback: {self.FALLBACK_URL}")
            try:
                response = requests.get(self.FALLBACK_URL, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                self.data_source = f"remote:{self.FALLBACK_URL}"
                print(f"✓ Loaded Violation State data from GitHub")
            except Exception as e:
                print(f"[ViolationStateImporter] GitHub fallback failed: {e}")
                df = None

        # If still no data, return None
        if df is None:
            print("[ViolationStateImporter] Data not available from any source")
            return None

        if len(df) == 0:
            print("[ViolationStateImporter] Data file is empty")
            return None

        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        try:
            df_normalized = self._validate_and_normalize(df)
        except Exception as e:
            print(f"[ViolationStateImporter] Data validation failed: {e}")
            return None

        self.df = df_normalized

        print(f"✓ Data validated: {len(df_normalized)} turns across {df_normalized['conversation_id'].nunique()} conversations")
        print(f"  Source: {self.data_source}")

        return df_normalized

    def get_data_info(self) -> Dict[str, Any]:
        """Get summary information about the loaded data."""
        if self.df is None:
            return {"status": "not loaded"}

        info = {
            "status": "loaded",
            "total_rows": len(self.df),
            "num_conversations": self.df['conversation_id'].nunique(),
            "turn_range": (int(self.df['turn_number'].min()), int(self.df['turn_number'].max())),
            "column_mapping": self._column_mapping,
        }

        if 'contamination_detected' in self.df.columns:
            contamination_count = self.df['contamination_detected'].sum()
            info['contamination_count'] = int(contamination_count)
            info['contamination_rate'] = float(contamination_count / len(self.df)) if len(self.df) > 0 else 0.0

        if 'response_type' in self.df.columns:
            info['response_types'] = self.df['response_type'].value_counts().to_dict()

        if 'violation_type' in self.df.columns:
            info['violation_types'] = sorted(self.df['violation_type'].dropna().unique().tolist())

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

    def get_contaminated_sessions(self) -> pd.DataFrame:
        """Get all sessions with contamination detected."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'contamination_detected' not in self.df.columns:
            raise ValueError("contamination_detected column not available in data")

        return self.df[self.df['contamination_detected'] == True].copy()

    def get_by_response_type(self, response_type: str) -> pd.DataFrame:
        """Get all turns with a specific response type."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'response_type' not in self.df.columns:
            raise ValueError("response_type column not available in data")

        return self.df[self.df['response_type'] == response_type].copy()

    def get_refusals(self) -> pd.DataFrame:
        """Get all turns classified as refusals."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'response_type' not in self.df.columns:
            raise ValueError("response_type column not available in data")

        # Common refusal classifications
        refusal_types = ['refusal', 'refused', 'decline', 'declined']
        return self.df[self.df['response_type'].str.lower().isin(refusal_types)].copy()
