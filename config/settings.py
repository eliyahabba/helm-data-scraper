"""
Centralized project configuration and registries.

Edit this file to update environment-specific paths, dataset/model registries,
and frequently changed constants. This module is intentionally minimal and free
of side-effects so it can be safely imported anywhere.

Guidelines:
- Keep values deterministic and static; do not compute values from runtime state.
- If you change paths, ensure they exist or are created by call sites (not here).
- If you add dataset/model entries, follow existing key naming conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------------------
# Paths (environment-specific)
# --------------------------------------------------------------------------------------

# NOTE: Updated paths to work with new project structure
HF_MAP_DATA_DIR: Path = Path(__file__).parent.parent / "data" / "hf_map_data"

# Path to model metadata CSV file used by model utilities  
MODEL_METADATA_CSV: Path = Path(__file__).parent / "model_metadata.csv"


# --------------------------------------------------------------------------------------
# Centralized Data Directory Management
# --------------------------------------------------------------------------------------

# Default base directory for all data files.
# This can be overridden at runtime via command-line arguments in scripts like main_processor.py.
DEFAULT_DATA_DIR: Path = Path(__file__).parent.parent / "data"

# Subdirectory names. Centralized here to ensure consistency across the project.
DOWNLOADS_SUBDIR: str = "downloads"
PROCESSED_SUBDIR: str = "processed"
AGGREGATED_SUBDIR: str = "aggregated"
BENCHMARK_LINES_SUBDIR: str = "benchmark_lines"


def get_data_directories(base_data_dir: Path) -> Dict[str, Path]:
    """
    Generates a dictionary of all critical data directory paths based on a provided
    base directory. This allows for easy redirection of all data storage.
    
    Args:
        base_data_dir: The base data directory path.
        
    Returns:
        A dictionary mapping directory roles to their full Path objects.
    """
    return {
        'data_dir': base_data_dir,
        'downloads_dir': base_data_dir / DOWNLOADS_SUBDIR,
        'processed_dir': base_data_dir / PROCESSED_SUBDIR,
        'aggregated_dir': base_data_dir / AGGREGATED_SUBDIR,
        'benchmark_csvs_dir': base_data_dir / BENCHMARK_LINES_SUBDIR,
    }


# Generate default directory paths for easy import elsewhere in the application
# that do not need dynamic path resolution.
DEFAULT_DIRS = get_data_directories(DEFAULT_DATA_DIR)

# --------------------------------------------------------------------------------------
# HELM download & processing settings
# --------------------------------------------------------------------------------------

# Versions to search for when downloading HELM files (kept identical ordering)
HELM_1_VERSIONS: List[str] = [f"v1.{i}.0" for i in range(14)]  # v1.0.0 to v1.13.0
HELM_0_VERSIONS: List[str] = [f"v0.{i}.0" for i in range(3, 14)]  # v1.0.0 to v1.13.0
HELM_VERSIONS: List[str] = HELM_1_VERSIONS + HELM_0_VERSIONS
# Default starting version
DEFAULT_START_VERSION: str = "v1.0.0"

# Base URL template for HELM assets
HELM_URL_WITH_BENCHMARK_TEMPLATE: str = (
    "https://storage.googleapis.com/crfm-helm-public/{benchmark}/benchmark_output/runs/{version}"
)

HELM_URL_WITHOUT_BENCHMARK_TEMPLATE: str = (
    "https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/{version}"
)

INSTRUCT_HELM_URL_WITHOUT_VERSION_TEMPLATE: str = (
    "https://storage.googleapis.com/crfm-helm-public/{benchmark}/benchmark_output/runs/instruction_following/"
)

# File types fetched per task (unchanged)
HELM_FILE_TYPES: List[str] = [
    "run_spec",
    "stats",
    "per_instance_stats",
    "instances",
    "scenario_state",
    "display_predictions",
    "display_requests",
    "scenario",
]

# Concurrency for `ProcessPoolExecutor` in `helm_data_processor.py`
PROCESS_POOL_MAX_WORKERS: int = 8

# --------------------------------------------------------------------------------------
# Small utility constants
# --------------------------------------------------------------------------------------

# Progress bar format used in `helm_data_processor.py` (kept identical)
TQDM_BAR_FORMAT: str = (
    "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
