"""
Utility functions for data processing and environment management.
"""
import os
import math
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

from .config import ENV_FILE


def sanitize_for_json(obj):
    """
    Recursively sanitize an object to ensure it's JSON-serializable.
    Converts nan/inf floats to None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string for unknown types
        try:
            return str(obj)
        except:
            return None


def sanitize_data(data):
    """
    Recursively replace NaN/Inf and non-JSON types (like pd.NA) for JSON compliance.
    More comprehensive than sanitize_for_json.
    """
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif pd.isna(data):  # Handles None, np.nan, pd.NA, pd.NaT
        return None
    elif isinstance(data, (np.float64, np.float32, float)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (np.int64, np.int32, int)):
        return int(data)
    elif isinstance(data, (datetime, pd.Timestamp)):
        return data.strftime('%Y-%m-%d %H:%M:%S')
    return data


def load_env_file() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    """
    env_vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def save_env_file(updates: Dict[str, str]):
    """
    Update environment variables in .env file.
    Preserves existing variables and comments.
    """
    lines = []
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

    key_map = {}
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#") and "=" in line:
            k = line.split("=", 1)[0].strip()
            key_map[k] = i

    for key, value in updates.items():
        if value is None:
            continue

        new_line = f"{key}={value}\n"
        if key in key_map:
            lines[key_map[key]] = new_line
        else:
            lines.append(new_line)

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


def mask_api_key(key: str) -> str:
    """
    Mask an API key for display, showing only first and last 4 characters.
    """
    if not key:
        return ""
    if len(key) > 8:
        return key[:4] + "..." + key[-4:]
    return key
