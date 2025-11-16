"""
Mirror Loop Study Data Importer

Loads data from the Mirror Loop empirical study.
This importer is READ ONLY - it does not modify source data.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import warnings


class MirrorLoopImporter:
    """
    Importer for Mirror Loop study data.

    The Mirror Loop study examines information collapse in iterative self-critique.
    This importer loads sequences of model outputs and prepares them for ΔI analysis.
    """

    DEFAULT_SEARCH_PATHS = [
        "mirror_loop_results_all.csv",
        "../mirror-loop/mirror_loop_results_all.csv",
        "../mirror-loop/results/mirror_loop_results_all.csv",
        "../mirror-loop/data/mirror_loop_results_all.csv",
        "data/mirror_loop/mirror_loop_results_all.csv",
    ]

    REQUIRED_COLUMNS = {
        "iteration": ["iteration", "turn", "step", "turn_number"],
        "model": ["model", "model_name", "model_id"],
        "response": ["response", "output", "output_text", "content", "text"],
        "sequence_id": ["sequence_id", "run_id", "session_id", "conversation_id"],
    }

    OPTIONAL_COLUMNS = {
        "prompt": ["prompt", "input", "input_text", "query"],
        "provider": ["provider", "api_provider"],
        "timestamp": ["timestamp", "created_at", "time"],
    }

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the Mirror Loop importer.

        Args:
            data_path: Explicit path to the data file. If None, will auto-discover.
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self._column_mapping: Dict[str, str] = {}

    def _find_data_file(self) -> str:
        """
        Find the data file by searching common locations.

        Returns:
            Path to the data file

        Raises:
            FileNotFoundError: If data file cannot be found
        """
        # Check explicit path first
        if self.data_path:
            if os.path.exists(self.data_path):
                return self.data_path
            raise FileNotFoundError(
                f"Specified data path does not exist: {self.data_path}"
            )

        # Check environment variable
        env_path = os.getenv("MIRROR_LOOP_DATA_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Search default locations
        for search_path in self.DEFAULT_SEARCH_PATHS:
            if os.path.exists(search_path):
                return search_path

        # Not found - raise helpful error
        raise FileNotFoundError(
            "Mirror Loop data file not found. Searched locations:\n" +
            "\n".join(f"  - {p}" for p in self.DEFAULT_SEARCH_PATHS) +
            "\n\nPlease provide data_path explicitly or set MIRROR_LOOP_DATA_PATH environment variable."
        )

    def _normalize_column_name(self, df: pd.DataFrame, standard_name: str, variants: List[str]) -> Optional[str]:
        """
        Find a column by checking variants (case-insensitive).

        Args:
            df: DataFrame to search
            standard_name: Standard column name
            variants: List of possible variants

        Returns:
            Actual column name in DataFrame, or None if not found
        """
        df_columns_lower = {col.lower(): col for col in df.columns}

        for variant in variants:
            if variant.lower() in df_columns_lower:
                actual_col = df_columns_lower[variant.lower()]
                self._column_mapping[standard_name] = actual_col
                return actual_col

        return None

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize the DataFrame schema.

        Args:
            df: Raw DataFrame

        Returns:
            Normalized DataFrame with standard column names

        Raises:
            ValueError: If required columns are missing
        """
        # Find all required columns
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

        # Find optional columns
        for standard_name, variants in self.OPTIONAL_COLUMNS.items():
            actual_col = self._normalize_column_name(df, standard_name, variants)
            if actual_col:
                normalized_cols[actual_col] = standard_name
            else:
                warnings.warn(f"Optional column '{standard_name}' not found")

        # Rename columns to standard names
        df_normalized = df.rename(columns=normalized_cols)

        # Validate data types
        df_normalized['iteration'] = pd.to_numeric(df_normalized['iteration'], errors='coerce')

        # Check for nulls in required columns
        for col in ['iteration', 'model', 'response', 'sequence_id']:
            null_count = df_normalized[col].isnull().sum()
            if null_count > 0:
                warnings.warn(f"Column '{col}' has {null_count} null values")

        return df_normalized

    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the Mirror Loop data.

        Returns:
            Normalized DataFrame with standard column names

        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data validation fails
        """
        # Find data file
        file_path = self._find_data_file()
        print(f"Loading Mirror Loop data from: {file_path}")

        # Load CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Validate not empty
        if len(df) == 0:
            raise ValueError("Data file is empty")

        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Validate and normalize
        df_normalized = self._validate_and_normalize(df)

        # Store for later access
        self.df = df_normalized

        print(f"✓ Data validated: {len(df_normalized)} rows across {df_normalized['sequence_id'].nunique()} sequences")

        return df_normalized

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded data.

        Returns:
            Dictionary with data statistics
        """
        if self.df is None:
            return {"status": "not loaded"}

        return {
            "status": "loaded",
            "total_rows": len(self.df),
            "num_sequences": self.df['sequence_id'].nunique(),
            "models": sorted(self.df['model'].unique().tolist()),
            "iteration_range": (self.df['iteration'].min(), self.df['iteration'].max()),
            "avg_sequence_length": self.df.groupby('sequence_id').size().mean(),
            "column_mapping": self._column_mapping,
        }

    def get_sequence(self, sequence_id: str) -> pd.DataFrame:
        """
        Get data for a specific sequence.

        Args:
            sequence_id: Sequence identifier

        Returns:
            DataFrame containing only the specified sequence, sorted by iteration
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        sequence_df = self.df[self.df['sequence_id'] == sequence_id].copy()
        sequence_df = sequence_df.sort_values('iteration')

        return sequence_df

    def get_sequences_by_model(self, model: str) -> pd.DataFrame:
        """
        Get all sequences for a specific model.

        Args:
            model: Model identifier

        Returns:
            DataFrame containing sequences for the specified model
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.df[self.df['model'] == model].copy()
