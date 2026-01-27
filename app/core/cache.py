"""
Global caching mechanisms for market data and computed values.
"""
import time
from threading import Lock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class IndicesCache:
    """
    Thread-safe cache for market indices data.
    Handles different cache TTL for active vs inactive market hours.
    """
    def __init__(self):
        self.data = []
        self.expiry = 0
        self._lock = Lock()

    def get(self) -> list:
        """Get cached data if still valid."""
        with self._lock:
            now_ts = time.time()
            now_dt = datetime.now()
            current_hm = now_dt.hour * 100 + now_dt.minute

            # Active Hours: 08:00 - 15:00 OR 21:30 - 05:00
            is_active_session_1 = 800 <= current_hm < 1500
            is_active_session_2 = (2130 <= current_hm) or (current_hm < 500)
            is_active = is_active_session_1 or is_active_session_2

            # If inactive and we have data, use it indefinitely
            if not is_active and self.data:
                return self.data

            # Otherwise (Active OR Empty Cache), check standard expiry
            if self.data and now_ts < self.expiry:
                return self.data

            return None

    def set(self, data: list, ttl_seconds: int = 60):
        """Set cache with TTL."""
        with self._lock:
            self.data = data
            self.expiry = time.time() + ttl_seconds

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.data = []
            self.expiry = 0


class StockFeatureCache:
    """
    Cache for stock professional features (financials, shareholders, etc.)
    with configurable TTL per feature type.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, cache_key: str, ttl_minutes: int) -> Optional[Dict]:
        """Get cached data if not expired."""
        with self._lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if datetime.now() - cached['timestamp'] < timedelta(minutes=ttl_minutes):
                    return cached['data']
        return None

    def set(self, cache_key: str, data: Dict):
        """Set cache with timestamp."""
        with self._lock:
            self._cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }

    def clear(self, key_prefix: str = None):
        """Clear cache, optionally by key prefix."""
        with self._lock:
            if key_prefix:
                keys_to_remove = [k for k in self._cache if k.startswith(key_prefix)]
                for k in keys_to_remove:
                    del self._cache[k]
            else:
                self._cache.clear()


# Global cache instances
indices_cache = IndicesCache()
stock_feature_cache = StockFeatureCache()
