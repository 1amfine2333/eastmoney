"""
Risk Metrics Calculator

Calculates institutional-grade risk metrics:
- Portfolio Beta (relative to benchmark)
- Sharpe Ratio (risk-adjusted returns)
- Value at Risk (VaR)
- Health Score (AI composite score)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats


class RiskMetricsCalculator:
    """
    Calculates portfolio-level risk metrics for institutional-grade analysis.
    """

    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    TRADING_DAYS = 252  # Trading days per year

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_risk_summary(
        self,
        positions: List[Dict],
        price_histories: Dict[str, List[Dict]],
        benchmark_history: List[Dict],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk summary including Beta, Sharpe, VaR, and Health Score.

        Args:
            positions: List of positions with asset_code, total_shares, average_cost
            price_histories: Dict mapping asset_code to price history [{date, price}]
            benchmark_history: Benchmark index price history [{date, price}]
            current_prices: Dict mapping asset_code to current price

        Returns:
            Risk summary with beta, sharpe, var, health_score, and details
        """
        if not positions or not price_histories:
            return self._empty_risk_summary()

        # Calculate position weights
        weights, total_value = self._calculate_weights(positions, current_prices)

        if total_value == 0:
            return self._empty_risk_summary()

        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            positions, weights, price_histories
        )

        if len(portfolio_returns) < 20:
            return self._empty_risk_summary("Insufficient historical data")

        # Calculate benchmark returns
        benchmark_returns = self._calculate_benchmark_returns(benchmark_history)

        # Align portfolio and benchmark returns
        portfolio_returns, benchmark_returns = self._align_returns(
            portfolio_returns, benchmark_returns
        )

        if len(portfolio_returns) < 20:
            return self._empty_risk_summary("Insufficient aligned data")

        # Calculate metrics
        
        var_95 = self._calculate_var(portfolio_returns, total_value)
        volatility = self._calculate_volatility(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)

        return {
            "var_95": round(var_95, 2),
            "var_95_pct": round(var_95 / total_value * 100, 2) if total_value > 0 else 0,
            "volatility": round(volatility * 100, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "total_value": round(total_value, 2),
            "position_count": len(positions),
            "analysis_days": len(portfolio_returns),
            "computed_at": datetime.now().isoformat()
        }

    def _empty_risk_summary(self, message: str = "No data available") -> Dict[str, Any]:
        """Return empty risk summary with default values."""
        return {
            "beta": None,
            "beta_status": "unknown",
            "sharpe_ratio": None,
            "sharpe_status": "unknown",
            "var_95": None,
            "var_95_pct": None,
            "volatility": None,
            "max_drawdown": None,
            "health_score": 50,
            "health_grade": "N/A",
            "total_value": 0,
            "position_count": 0,
            "analysis_days": 0,
            "message": message,
            "computed_at": datetime.now().isoformat()
        }

    def _calculate_weights(
        self,
        positions: List[Dict],
        current_prices: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """Calculate position weights based on current values."""
        total_value = 0
        position_values = {}

        for pos in positions:
            code = pos.get("asset_code")
            shares = float(pos.get("total_shares", 0))
            price = current_prices.get(code, float(pos.get("average_cost", 0)))

            value = shares * price
            position_values[code] = value
            total_value += value

        weights = {}
        if total_value > 0:
            weights = {code: val / total_value for code, val in position_values.items()}

        return weights, total_value

    def _calculate_portfolio_returns(
        self,
        positions: List[Dict],
        weights: Dict[str, float],
        price_histories: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """Calculate weighted portfolio daily returns."""
        # Get all dates across all positions
        all_dates = set()
        price_maps = {}

        for pos in positions:
            code = pos.get("asset_code")
            history = price_histories.get(code, [])
            price_map = {}
            for h in history:
                date = h.get("date")
                price = h.get("price") or h.get("close") or h.get("value")
                if date and price:
                    price_map[date] = float(price)
                    all_dates.add(date)
            price_maps[code] = price_map

        # Calculate daily portfolio returns
        sorted_dates = sorted(all_dates)
        portfolio_returns = {}

        for i in range(1, len(sorted_dates)):
            date = sorted_dates[i]
            prev_date = sorted_dates[i - 1]

            daily_return = 0
            valid_weight = 0

            for code, weight in weights.items():
                price_map = price_maps.get(code, {})
                curr_price = price_map.get(date)
                prev_price = price_map.get(prev_date)

                if curr_price and prev_price and prev_price > 0:
                    asset_return = (curr_price / prev_price) - 1
                    daily_return += asset_return * weight
                    valid_weight += weight

            if valid_weight > 0.5:  # At least 50% of portfolio has data
                portfolio_returns[date] = daily_return

        return portfolio_returns

    def _calculate_benchmark_returns(
        self,
        benchmark_history: List[Dict]
    ) -> Dict[str, float]:
        """Calculate benchmark daily returns."""
        returns = {}
        sorted_history = sorted(benchmark_history, key=lambda x: x.get("date", ""))

        for i in range(1, len(sorted_history)):
            curr = sorted_history[i]
            prev = sorted_history[i - 1]

            date = curr.get("date")
            curr_price = float(curr.get("price") or curr.get("close") or curr.get("value", 0))
            prev_price = float(prev.get("price") or prev.get("close") or prev.get("value", 0))

            if date and curr_price and prev_price > 0:
                returns[date] = (curr_price / prev_price) - 1

        return returns

    def _align_returns(
        self,
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align portfolio and benchmark returns to common dates."""
        common_dates = set(portfolio_returns.keys()) & set(benchmark_returns.keys())
        sorted_dates = sorted(common_dates)

        port_ret = np.array([portfolio_returns[d] for d in sorted_dates])
        bench_ret = np.array([benchmark_returns[d] for d in sorted_dates])

        return port_ret, bench_ret

    def _calculate_beta(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate portfolio beta relative to benchmark.
        Beta = Covariance(Rp, Rm) / Variance(Rm)
        """
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return 1.0

        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)

        if variance == 0:
            return 1.0

        return covariance / variance

    def _calculate_sharpe(self, portfolio_returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.
        Sharpe = (Rp - Rf) / Std(Rp)
        """
        if len(portfolio_returns) < 2:
            return 0

        annual_return = np.mean(portfolio_returns) * self.TRADING_DAYS
        annual_std = np.std(portfolio_returns) * np.sqrt(self.TRADING_DAYS)

        if annual_std == 0:
            return 0

        return (annual_return - self.risk_free_rate) / annual_std

    def _calculate_var(
        self,
        portfolio_returns: np.ndarray,
        total_value: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk at given confidence level.
        Using historical simulation method.
        """
        if len(portfolio_returns) < 10:
            return 0

        # Calculate VaR percentile (e.g., 5th percentile for 95% confidence)
        var_pct = np.percentile(portfolio_returns, (1 - confidence) * 100)

        # VaR in dollar terms (positive number representing potential loss)
        return abs(var_pct) * total_value

    def _calculate_volatility(self, portfolio_returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(portfolio_returns) < 2:
            return 0
        return np.std(portfolio_returns) * np.sqrt(self.TRADING_DAYS)

    def _calculate_max_drawdown(self, portfolio_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_returns) < 2:
            return 0

        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return abs(np.min(drawdowns))

    def _calculate_health_score(
        self,
        beta: float,
        sharpe: float,
        var_pct: float,
        max_drawdown: float,
        position_count: int
    ) -> float:
        """
        Calculate composite health score (0-100).

        Components:
        - Risk-adjusted return (Sharpe): 30 points
        - Market exposure (Beta): 20 points
        - Downside protection (VaR, Max DD): 30 points
        - Diversification: 20 points
        """
        score = 0

        # Sharpe score (30 points)
        # Sharpe > 2: full points, < 0: 0 points
        sharpe_score = min(30, max(0, (sharpe / 2) * 30))
        score += sharpe_score

        # Beta score (20 points)
        # Optimal beta around 0.8-1.0
        if 0.8 <= beta <= 1.0:
            beta_score = 20
        elif 0.6 <= beta <= 1.2:
            beta_score = 15
        elif 0.4 <= beta <= 1.5:
            beta_score = 10
        else:
            beta_score = 5
        score += beta_score

        # Downside protection score (30 points)
        # Lower VaR and max drawdown = better
        var_score = max(0, 15 - var_pct * 100)  # VaR < 15% = full points
        dd_score = max(0, 15 - max_drawdown * 100)  # Max DD < 15% = full points
        score += var_score + dd_score

        # Diversification score (20 points)
        # Based on position count
        if position_count >= 10:
            div_score = 20
        elif position_count >= 7:
            div_score = 15
        elif position_count >= 5:
            div_score = 10
        elif position_count >= 3:
            div_score = 5
        else:
            div_score = 0
        score += div_score

        return min(100, max(0, score))

    def _get_beta_status(self, beta: float) -> str:
        """Get beta status indicator."""
        if beta is None:
            return "unknown"
        if beta < 1.0:
            return "low"  # Green - defensive
        elif beta <= 1.2:
            return "medium"  # Yellow - moderate
        else:
            return "high"  # Red - aggressive

    def _get_sharpe_status(self, sharpe: float) -> str:
        """Get Sharpe ratio status indicator."""
        if sharpe is None:
            return "unknown"
        if sharpe > 2:
            return "excellent"
        elif sharpe > 1:
            return "good"
        elif sharpe > 0.5:
            return "fair"
        else:
            return "poor"

    def _get_health_grade(self, score: float) -> str:
        """Convert health score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def calculate_sparkline_data(
        self,
        portfolio_id: int,
        snapshots: List[Dict],
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate sparkline data for portfolio value over time.

        Args:
            portfolio_id: Portfolio ID
            snapshots: List of portfolio snapshots with date and total_value
            days: Number of days for sparkline (default 7)

        Returns:
            Sparkline data with values and change metrics
        """
        if not snapshots:
            return {
                "portfolio_id": portfolio_id,
                "values": [],
                "dates": [],
                "change": 0,
                "change_pct": 0,
                "trend": "flat"
            }

        # Sort by date and get last N days
        sorted_snapshots = sorted(snapshots, key=lambda x: x.get("snapshot_date", ""))
        recent = sorted_snapshots[-days:] if len(sorted_snapshots) >= days else sorted_snapshots

        values = [s.get("total_value", 0) for s in recent]
        dates = [s.get("snapshot_date", "") for s in recent]

        # Calculate change
        if len(values) >= 2:
            first_val = values[0]
            last_val = values[-1]
            change = last_val - first_val
            change_pct = (change / first_val * 100) if first_val > 0 else 0
        else:
            change = 0
            change_pct = 0

        # Determine trend
        if change_pct > 1:
            trend = "up"
        elif change_pct < -1:
            trend = "down"
        else:
            trend = "flat"

        return {
            "portfolio_id": portfolio_id,
            "values": [round(v, 2) for v in values],
            "dates": dates,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "trend": trend,
            "days": len(values)
        }
