"""
Path configuration for the application.
"""
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Report storage directory
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Configuration directory
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Environment file
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Static files directory (frontend build)
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Cache files
MARKET_FUNDS_CACHE = os.path.join(CONFIG_DIR, "market_funds_cache.json")
MARKET_STOCKS_CACHE = os.path.join(CONFIG_DIR, "market_stocks_cache.json")

# Ensure directories exist
for directory in [REPORT_DIR, CONFIG_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
