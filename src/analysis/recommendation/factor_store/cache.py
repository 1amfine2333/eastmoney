"""
Factor Cache - Multi-level caching for recommendation factors.

Cache hierarchy:
1. Memory cache: 5-minute TTL for hot data
2. Database cache: 24-hour TTL for factor data

Optimized for 2H4G server with limited resources.
"""
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from functools import wraps


class MemoryCache:
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize memory cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if time.time() > entry['expires_at']:
                del self._cache[key]
                return None

            return entry['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional custom TTL."""
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expires_at': time.time() + (ttl or self._default_ttl)
            }

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items()
                if now > v['expires_at']
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            valid_count = sum(1 for v in self._cache.values() if now <= v['expires_at'])
            return {
                'total_entries': len(self._cache),
                'valid_entries': valid_count,
                'expired_entries': len(self._cache) - valid_count
            }


class FactorCache:
    """
    Multi-level factor cache for recommendation system.

    Provides fast access to stock and fund factors with automatic
    fallback from memory to database.
    """

    # Cache TTL settings
    MEMORY_TTL_HOT = 300  # 5 minutes for frequently accessed data
    MEMORY_TTL_WARM = 900  # 15 minutes for less frequent data
    DB_TTL_DAYS = 1  # 1 day for database cache

    def __init__(self):
        """Initialize factor cache."""
        self._stock_cache = MemoryCache(default_ttl=self.MEMORY_TTL_HOT)
        self._fund_cache = MemoryCache(default_ttl=self.MEMORY_TTL_HOT)
        self._metadata_cache = MemoryCache(default_ttl=self.MEMORY_TTL_WARM)

    def _make_key(self, prefix: str, code: str, trade_date: str) -> str:
        """Generate cache key."""
        return f"{prefix}:{code}:{trade_date}"

    # =========================================================================
    # Stock Factor Cache
    # =========================================================================

    def get_stock_factors(self, code: str, trade_date: str) -> Optional[Dict]:
        """
        Get stock factors with cache fallback.

        Args:
            code: Stock code
            trade_date: Trade date (YYYY-MM-DD)

        Returns:
            Factor dictionary or None if not found
        """
        # Import here to avoid circular imports
        from src.storage.db import get_stock_factors as db_get_stock_factors

        # Try memory cache first
        cache_key = self._make_key('stock', code, trade_date)
        cached = self._stock_cache.get(cache_key)
        if cached is not None:
            return cached

        # Fall back to database
        db_result = db_get_stock_factors(code, trade_date)
        if db_result:
            self._stock_cache.set(cache_key, db_result)
            return db_result

        return None

    def get_stock_factors_batch(
        self,
        codes: List[str],
        trade_date: str
    ) -> Dict[str, Dict]:
        """
        Get stock factors for multiple codes efficiently.

        Args:
            codes: List of stock codes
            trade_date: Trade date

        Returns:
            Dictionary mapping code to factors
        """
        from src.storage.db import get_stock_factors_batch as db_get_batch

        result = {}
        missing_codes = []

        # Check memory cache for each code
        for code in codes:
            cache_key = self._make_key('stock', code, trade_date)
            cached = self._stock_cache.get(cache_key)
            if cached is not None:
                result[code] = cached
            else:
                missing_codes.append(code)

        # Batch fetch missing from database
        if missing_codes:
            db_results = db_get_batch(missing_codes, trade_date)
            for factors in db_results:
                code = factors['code']
                result[code] = factors
                cache_key = self._make_key('stock', code, trade_date)
                self._stock_cache.set(cache_key, factors)

        return result

    def set_stock_factors(
        self,
        code: str,
        trade_date: str,
        factors: Dict,
        persist: bool = True
    ) -> None:
        """
        Set stock factors in cache (and optionally database).

        Args:
            code: Stock code
            trade_date: Trade date
            factors: Factor dictionary
            persist: Whether to persist to database
        """
        from src.storage.db import upsert_stock_factors

        # Update memory cache
        cache_key = self._make_key('stock', code, trade_date)
        self._stock_cache.set(cache_key, factors)

        # Persist to database
        if persist:
            factors_with_key = {**factors, 'code': code, 'trade_date': trade_date}
            upsert_stock_factors(factors_with_key)

    def get_top_stocks(
        self,
        trade_date: str,
        score_type: str = 'short_term',
        limit: int = 20,
        min_score: float = 0
    ) -> List[Dict]:
        """
        Get top-ranked stocks from cache/database.

        Args:
            trade_date: Trade date
            score_type: 'short_term' or 'long_term'
            limit: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            List of stock factors sorted by score
        """
        from src.storage.db import get_top_stocks_by_score

        # Try metadata cache first
        cache_key = f"top_stocks:{score_type}:{trade_date}:{limit}:{min_score}"
        cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        # Fetch from database
        results = get_top_stocks_by_score(trade_date, score_type, limit, min_score)

        # Cache results
        self._metadata_cache.set(cache_key, results)

        # Also cache individual entries
        for factors in results:
            code = factors['code']
            stock_key = self._make_key('stock', code, trade_date)
            self._stock_cache.set(stock_key, factors)

        return results

    # =========================================================================
    # Fund Factor Cache
    # =========================================================================

    def get_fund_factors(self, code: str, trade_date: str) -> Optional[Dict]:
        """
        Get fund factors with cache fallback.

        Args:
            code: Fund code
            trade_date: Trade date

        Returns:
            Factor dictionary or None if not found
        """
        from src.storage.db import get_fund_factors as db_get_fund_factors

        cache_key = self._make_key('fund', code, trade_date)
        cached = self._fund_cache.get(cache_key)
        if cached is not None:
            return cached

        db_result = db_get_fund_factors(code, trade_date)
        if db_result:
            self._fund_cache.set(cache_key, db_result)
            return db_result

        return None

    def get_fund_factors_batch(
        self,
        codes: List[str],
        trade_date: str
    ) -> Dict[str, Dict]:
        """
        Get fund factors for multiple codes efficiently.

        Args:
            codes: List of fund codes
            trade_date: Trade date

        Returns:
            Dictionary mapping code to factors
        """
        from src.storage.db import get_fund_factors_batch as db_get_batch

        result = {}
        missing_codes = []

        for code in codes:
            cache_key = self._make_key('fund', code, trade_date)
            cached = self._fund_cache.get(cache_key)
            if cached is not None:
                result[code] = cached
            else:
                missing_codes.append(code)

        if missing_codes:
            db_results = db_get_batch(missing_codes, trade_date)
            for factors in db_results:
                code = factors['code']
                result[code] = factors
                cache_key = self._make_key('fund', code, trade_date)
                self._fund_cache.set(cache_key, factors)

        return result

    def set_fund_factors(
        self,
        code: str,
        trade_date: str,
        factors: Dict,
        persist: bool = True
    ) -> None:
        """
        Set fund factors in cache (and optionally database).

        Args:
            code: Fund code
            trade_date: Trade date
            factors: Factor dictionary
            persist: Whether to persist to database
        """
        from src.storage.db import upsert_fund_factors

        cache_key = self._make_key('fund', code, trade_date)
        self._fund_cache.set(cache_key, factors)

        if persist:
            factors_with_key = {**factors, 'code': code, 'trade_date': trade_date}
            upsert_fund_factors(factors_with_key)

    def get_top_funds(
        self,
        trade_date: str,
        score_type: str = 'short_term',
        limit: int = 20,
        min_score: float = 0
    ) -> List[Dict]:
        """
        Get top-ranked funds from cache/database.

        Args:
            trade_date: Trade date
            score_type: 'short_term' or 'long_term'
            limit: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            List of fund factors sorted by score
        """
        from src.storage.db import get_top_funds_by_score

        cache_key = f"top_funds:{score_type}:{trade_date}:{limit}:{min_score}"
        cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        results = get_top_funds_by_score(trade_date, score_type, limit, min_score)

        self._metadata_cache.set(cache_key, results)

        for factors in results:
            code = factors['code']
            fund_key = self._make_key('fund', code, trade_date)
            self._fund_cache.set(fund_key, factors)

        return results

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_all(self) -> None:
        """Clear all caches."""
        self._stock_cache.clear()
        self._fund_cache.clear()
        self._metadata_cache.clear()

    def clear_for_date(self, trade_date: str) -> None:
        """Clear cache entries for a specific date (used when recomputing)."""
        # Memory caches use key patterns, so we can't selectively clear by date
        # without iterating. For simplicity, clear all.
        self.clear_all()

    def cleanup(self) -> Dict[str, int]:
        """Clean up expired entries and return cleanup stats."""
        return {
            'stock_cache': self._stock_cache.cleanup_expired(),
            'fund_cache': self._fund_cache.cleanup_expired(),
            'metadata_cache': self._metadata_cache.cleanup_expired()
        }

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache statistics."""
        return {
            'stock_cache': self._stock_cache.get_stats(),
            'fund_cache': self._fund_cache.get_stats(),
            'metadata_cache': self._metadata_cache.get_stats()
        }


# Global cache instance
factor_cache = FactorCache()


def cached_factor(cache_type: str = 'stock', ttl: int = 300):
    """
    Decorator for caching factor computation results.

    Args:
        cache_type: 'stock' or 'fund'
        ttl: Cache TTL in seconds

    Usage:
        @cached_factor('stock', ttl=300)
        def compute_technical_factors(code: str, trade_date: str) -> Dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(code: str, trade_date: str, *args, **kwargs):
            cache = factor_cache._stock_cache if cache_type == 'stock' else factor_cache._fund_cache
            cache_key = f"{func.__name__}:{code}:{trade_date}"

            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            result = func(code, trade_date, *args, **kwargs)

            if result is not None:
                cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
