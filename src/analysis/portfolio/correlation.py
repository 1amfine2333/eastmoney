"""
Correlation Analysis Module

Calculates correlation matrix for portfolio positions:
- 90-day return correlation matrix
- AI-powered correlation interpretation
- Risk concentration warnings
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats


class CorrelationAnalyzer:
    """
    Analyzes correlation between portfolio positions.
    """

    # Correlation thresholds
    HIGH_POSITIVE_THRESHOLD = 0.7
    LOW_CORRELATION_THRESHOLD = 0.3
    NEGATIVE_THRESHOLD = -0.3

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days

    def calculate_correlation_matrix(
        self,
        positions: List[Dict],
        price_histories: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix for all positions.

        Args:
            positions: List of positions with asset_code, asset_name
            price_histories: Dict mapping asset_code to price history

        Returns:
            Correlation matrix data with labels and interpretations
        """
        if not positions or len(positions) < 2:
            return self._empty_correlation_result("Need at least 2 positions")

        # Build returns DataFrame
        returns_data = {}
        asset_names = {}

        for pos in positions:
            code = pos.get("asset_code")
            name = pos.get("asset_name", code)
            history = price_histories.get(code, [])

            if not history:
                print(f"[CorrelationAnalyzer] {code}: No history data")
                continue

            # Calculate returns
            returns = self._calculate_returns(history)
            print(f"[CorrelationAnalyzer] {code}: {len(history)} history points -> {len(returns)} returns")

            # Debug: print first few dates
            if returns:
                sample_dates = list(returns.keys())[:3]
                print(f"[CorrelationAnalyzer] {code} sample dates: {sample_dates}")

            if len(returns) >= 20:  # Minimum 20 data points
                returns_data[code] = returns
                asset_names[code] = name

        if len(returns_data) < 2:
            return self._empty_correlation_result("Insufficient price data")

        # Create DataFrame with aligned dates
        df = pd.DataFrame(returns_data)
        print(f"[CorrelationAnalyzer] DataFrame shape before dropna: {df.shape}")

        df = df.dropna()
        print(f"[CorrelationAnalyzer] DataFrame shape after dropna: {df.shape}")

        if len(df) < 20:
            return self._empty_correlation_result("Insufficient aligned data")

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Convert to list format for frontend
        codes = list(corr_matrix.columns)
        labels = [asset_names.get(c, c) for c in codes]

        matrix_data = []
        for i, row_code in enumerate(codes):
            for j, col_code in enumerate(codes):
                corr_val = corr_matrix.loc[row_code, col_code]
                matrix_data.append({
                    "x": j,
                    "y": i,
                    "value": round(corr_val, 3),
                    "row_code": row_code,
                    "col_code": col_code,
                    "row_name": labels[i],
                    "col_name": labels[j]
                })

        # Identify high correlation pairs
        high_correlations = self._find_high_correlations(corr_matrix, asset_names)

        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(corr_matrix)

        # Generate AI interpretations
        interpretations = self._generate_interpretations(
            corr_matrix, asset_names, high_correlations
        )

        return {
            "matrix": matrix_data,
            "labels": labels,
            "codes": codes,
            "size": len(codes),
            "high_correlations": high_correlations,
            "diversification_score": round(diversification_score, 1),
            "diversification_status": self._get_diversification_status(diversification_score),
            "interpretations": interpretations,
            "analysis_days": len(df),
            "computed_at": datetime.now().isoformat()
        }

    def _empty_correlation_result(self, message: str) -> Dict[str, Any]:
        """Return empty correlation result."""
        return {
            "matrix": [],
            "labels": [],
            "codes": [],
            "size": 0,
            "high_correlations": [],
            "diversification_score": 0,
            "diversification_status": "unknown",
            "interpretations": [],
            "message": message,
            "computed_at": datetime.now().isoformat()
        }

    def _calculate_returns(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate daily returns from price history."""
        sorted_history = sorted(history, key=lambda x: x.get("date", ""))
        returns = {}

        for i in range(1, len(sorted_history)):
            curr = sorted_history[i]
            prev = sorted_history[i - 1]

            date = curr.get("date")
            curr_price = float(curr.get("price") or curr.get("close") or curr.get("value", 0))
            prev_price = float(prev.get("price") or prev.get("close") or prev.get("value", 0))

            if date and curr_price and prev_price > 0:
                # Normalize date format to YYYY-MM-DD
                normalized_date = self._normalize_date(date)
                if normalized_date:
                    returns[normalized_date] = (curr_price / prev_price) - 1

        return returns

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format."""
        if not date_str:
            return ""

        # Already in correct format
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str

        # Try various formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y%m%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If all else fails, try to extract date parts
        try:
            # Remove any non-digit characters and try to parse
            digits = ''.join(c for c in date_str if c.isdigit())
            if len(digits) == 8:
                return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        except:
            pass

        return date_str  # Return as-is if can't normalize

    def _find_high_correlations(
        self,
        corr_matrix: pd.DataFrame,
        asset_names: Dict[str, str]
    ) -> List[Dict]:
        """Find pairs with high positive correlation (potential concentration risk)."""
        high_corr = []
        codes = list(corr_matrix.columns)

        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                code_a = codes[i]
                code_b = codes[j]
                corr_val = corr_matrix.loc[code_a, code_b]

                if corr_val >= self.HIGH_POSITIVE_THRESHOLD:
                    high_corr.append({
                        "pair": [code_a, code_b],
                        "pair_names": [
                            asset_names.get(code_a, code_a),
                            asset_names.get(code_b, code_b)
                        ],
                        "correlation": round(corr_val, 3),
                        "risk_level": "high" if corr_val >= 0.85 else "medium",
                        "message": self._generate_pair_message(
                            asset_names.get(code_a, code_a),
                            asset_names.get(code_b, code_b),
                            corr_val
                        )
                    })

        # Sort by correlation (highest first)
        high_corr.sort(key=lambda x: x["correlation"], reverse=True)
        return high_corr

    def _generate_pair_message(
        self,
        name_a: str,
        name_b: str,
        correlation: float
    ) -> str:
        """Generate interpretation message for a correlation pair."""
        if correlation >= 0.9:
            return f"{name_a} 与 {name_b} 相关性 {correlation:.2f}，走势几乎完全同步，分散风险有限"
        elif correlation >= 0.8:
            return f"{name_a} 与 {name_b} 相关性 {correlation:.2f}，同质化严重，建议考虑减持其一"
        elif correlation >= 0.7:
            return f"{name_a} 与 {name_b} 相关性 {correlation:.2f}，走势相似度较高，注意集中风险"
        else:
            return f"{name_a} 与 {name_b} 相关性 {correlation:.2f}"

    def _calculate_diversification_score(self, corr_matrix: pd.DataFrame) -> float:
        """
        Calculate portfolio diversification score (0-100).

        Based on average pairwise correlation:
        - Lower average correlation = better diversification
        - Score = 100 * (1 - avg_correlation)
        """
        n = len(corr_matrix)
        if n < 2:
            return 0

        # Get upper triangle (excluding diagonal)
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(corr_matrix.iloc[i, j])

        if not upper_triangle:
            return 50

        avg_correlation = np.mean(upper_triangle)

        # Score: lower correlation = higher score
        # avg_corr of 0 = score of 100
        # avg_corr of 1 = score of 0
        # avg_corr of 0.5 = score of 50
        score = (1 - avg_correlation) * 100

        return max(0, min(100, score))

    def _get_diversification_status(self, score: float) -> str:
        """Get diversification status based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "critical"

    def _generate_interpretations(
        self,
        corr_matrix: pd.DataFrame,
        asset_names: Dict[str, str],
        high_correlations: List[Dict]
    ) -> List[Dict]:
        """Generate AI interpretations for the correlation matrix."""
        interpretations = []

        # Overall diversification interpretation
        n = len(corr_matrix)
        if n >= 2:
            upper_vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper_vals.append(corr_matrix.iloc[i, j])

            avg_corr = np.mean(upper_vals)
            max_corr = max(upper_vals)
            min_corr = min(upper_vals)

            interpretations.append({
                "type": "overview",
                "title": "整体相关性",
                "content": f"组合平均相关性 {avg_corr:.2f}，最高 {max_corr:.2f}，最低 {min_corr:.2f}",
                "severity": "info" if avg_corr < 0.5 else "warning"
            })

        # High correlation warnings
        if high_correlations:
            worst_pair = high_correlations[0]
            interpretations.append({
                "type": "warning",
                "title": "集中风险警告",
                "content": f"发现 {len(high_correlations)} 对高相关资产，最高 {worst_pair['pair_names'][0]} 与 {worst_pair['pair_names'][1]} ({worst_pair['correlation']:.2f})",
                "severity": "critical" if worst_pair["correlation"] >= 0.85 else "warning"
            })

        # Look for hedging opportunities (negative correlations)
        codes = list(corr_matrix.columns)
        for i in range(n):
            for j in range(i + 1, n):
                corr = corr_matrix.iloc[i, j]
                if corr < self.NEGATIVE_THRESHOLD:
                    interpretations.append({
                        "type": "opportunity",
                        "title": "对冲组合",
                        "content": f"{asset_names.get(codes[i], codes[i])} 与 {asset_names.get(codes[j], codes[j])} 负相关 ({corr:.2f})，具有对冲效果",
                        "severity": "positive"
                    })

        return interpretations

    def get_correlation_tooltip(
        self,
        code_a: str,
        code_b: str,
        name_a: str,
        name_b: str,
        correlation: float
    ) -> Dict[str, Any]:
        """
        Generate detailed tooltip content for a correlation cell.

        Returns interpretation based on correlation value.
        """
        # Determine category
        if correlation >= 0.8:
            category = "very_high_positive"
            risk_level = "high"
            description = "走势高度同步，分散效果差"
            recommendation = "考虑减持其中一只以分散风险"
            color = "#ff4d4f"
        elif correlation >= 0.5:
            category = "moderate_positive"
            risk_level = "medium"
            description = "走势有一定相关性"
            recommendation = "可保持，但注意仓位平衡"
            color = "#faad14"
        elif correlation >= -0.2:
            category = "low_correlation"
            risk_level = "low"
            description = "走势相对独立，分散效果好"
            recommendation = "良好的分散组合"
            color = "#d9d9d9"
        elif correlation >= -0.5:
            category = "moderate_negative"
            risk_level = "hedge"
            description = "走势存在负相关"
            recommendation = "可作为对冲组合"
            color = "#52c41a"
        else:
            category = "strong_negative"
            risk_level = "hedge"
            description = "走势高度负相关，具有对冲效果"
            recommendation = "优秀的对冲组合"
            color = "#1890ff"

        return {
            "code_a": code_a,
            "code_b": code_b,
            "name_a": name_a,
            "name_b": name_b,
            "correlation": correlation,
            "category": category,
            "risk_level": risk_level,
            "description": description,
            "recommendation": recommendation,
            "color": color
        }
