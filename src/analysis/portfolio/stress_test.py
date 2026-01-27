"""
Stress Testing Engine

Provides institutional-grade stress testing capabilities:
- Predefined macro scenarios (Fed rate hikes, currency moves, market crashes)
- Custom factor sensitivity analysis
- VaR and Expected Shortfall under stress
- Top losers identification
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """Predefined stress test scenarios."""
    FED_RATE_HIKE_50BP = "fed_rate_hike_50bp"
    FED_RATE_CUT_50BP = "fed_rate_cut_50bp"
    CNY_DEPRECIATION_2PCT = "cny_depreciation_2pct"
    CNY_APPRECIATION_2PCT = "cny_appreciation_2pct"
    INDEX_DROP_5PCT = "index_drop_5pct"
    INDEX_DROP_10PCT = "index_drop_10pct"
    SEMICONDUCTOR_CRASH_10PCT = "semiconductor_crash_10pct"
    OIL_SPIKE_20PCT = "oil_spike_20pct"


@dataclass
class StressScenario:
    """Stress test scenario configuration."""
    interest_rate_change_bp: float = 0  # Basis points
    fx_change_pct: float = 0  # Percentage
    index_change_pct: float = 0  # Percentage
    oil_change_pct: float = 0  # Percentage
    sector_shocks: Dict[str, float] = None  # Sector-specific shocks


# Predefined scenario configurations
PREDEFINED_SCENARIOS = {
    ScenarioType.FED_RATE_HIKE_50BP: StressScenario(
        interest_rate_change_bp=50,
        index_change_pct=-2.0
    ),
    ScenarioType.FED_RATE_CUT_50BP: StressScenario(
        interest_rate_change_bp=-50,
        index_change_pct=1.5
    ),
    ScenarioType.CNY_DEPRECIATION_2PCT: StressScenario(
        fx_change_pct=-2.0,
        index_change_pct=-1.0
    ),
    ScenarioType.CNY_APPRECIATION_2PCT: StressScenario(
        fx_change_pct=2.0,
        index_change_pct=0.5
    ),
    ScenarioType.INDEX_DROP_5PCT: StressScenario(
        index_change_pct=-5.0
    ),
    ScenarioType.INDEX_DROP_10PCT: StressScenario(
        index_change_pct=-10.0
    ),
    ScenarioType.SEMICONDUCTOR_CRASH_10PCT: StressScenario(
        sector_shocks={"semiconductor": -10.0, "technology": -5.0}
    ),
    ScenarioType.OIL_SPIKE_20PCT: StressScenario(
        oil_change_pct=20.0,
        sector_shocks={"energy": 15.0, "airline": -10.0, "transportation": -5.0}
    ),
}


# Asset factor sensitivities (beta coefficients)
# These would ideally come from a factor model regression
DEFAULT_FACTOR_SENSITIVITIES = {
    # Format: asset_type -> {factor -> sensitivity}
    "stock": {
        "interest_rate": -0.15,  # Per 100bp
        "fx": 0.3,  # Per 1%
        "index": 1.0,  # Beta to market
        "oil": 0.1  # Per 1%
    },
    "fund": {
        "interest_rate": -0.10,
        "fx": 0.2,
        "index": 0.85,
        "oil": 0.05
    }
}

# Sector-specific sensitivities (multipliers on top of base)
SECTOR_SENSITIVITIES = {
    # Interest rate sensitive
    "银行": {"interest_rate": 0.5},
    "金融": {"interest_rate": 0.4},
    "房地产": {"interest_rate": -0.8},
    "科技": {"interest_rate": -0.5},

    # Export/import sensitive
    "出口": {"fx": 0.8},
    "进口": {"fx": -0.6},
    "外贸": {"fx": 0.5},

    # Oil sensitive
    "能源": {"oil": 1.2},
    "石油": {"oil": 1.5},
    "航空": {"oil": -0.8},
    "运输": {"oil": -0.4},

    # High beta sectors
    "半导体": {"index": 1.4},
    "科技": {"index": 1.3},
    "消费": {"index": 0.9},
    "医药": {"index": 0.8},
    "公用事业": {"index": 0.6},
}


class StressTestEngine:
    """
    Engine for portfolio stress testing and scenario analysis.
    """

    def __init__(self, factor_sensitivities: Dict = None):
        self.factor_sensitivities = factor_sensitivities or DEFAULT_FACTOR_SENSITIVITIES

    def run_stress_test(
        self,
        positions: List[Dict],
        scenario: StressScenario,
        current_prices: Dict[str, float],
        position_betas: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Run stress test on portfolio with given scenario.

        Args:
            positions: List of positions with asset_code, asset_type, total_shares, sector
            scenario: Stress scenario configuration
            current_prices: Current prices for each asset
            position_betas: Optional custom betas for each position

        Returns:
            Stress test results with projected P&L, VaR, and top losers
        """
        if not positions:
            return self._empty_stress_result()

        # Calculate current portfolio value
        total_value = 0
        position_values = []

        for pos in positions:
            code = pos.get("asset_code")
            shares = float(pos.get("total_shares", 0))
            price = current_prices.get(code, float(pos.get("average_cost", 0)))
            value = shares * price

            total_value += value
            position_values.append({
                "code": code,
                "name": pos.get("asset_name", code),
                "asset_type": pos.get("asset_type", "stock"),
                "sector": pos.get("sector", ""),
                "shares": shares,
                "price": price,
                "value": value,
                "weight": 0  # Will calculate after
            })

        # Calculate weights
        for pv in position_values:
            pv["weight"] = pv["value"] / total_value if total_value > 0 else 0

        # Apply stress scenario to each position
        results = []
        total_projected_pnl = 0

        for pv in position_values:
            # Get position-specific beta
            custom_beta = position_betas.get(pv["code"]) if position_betas else None

            # Calculate position impact
            impact = self._calculate_position_impact(
                pv["asset_type"],
                pv["sector"],
                scenario,
                custom_beta
            )

            projected_change = pv["value"] * impact
            total_projected_pnl += projected_change

            results.append({
                "code": pv["code"],
                "name": pv["name"],
                "sector": pv["sector"],
                "current_value": round(pv["value"], 2),
                "weight": round(pv["weight"] * 100, 2),
                "impact_pct": round(impact * 100, 2),
                "projected_change": round(projected_change, 2),
                "projected_value": round(pv["value"] + projected_change, 2)
            })

        # Sort by impact (worst first for losers)
        sorted_by_loss = sorted(results, key=lambda x: x["projected_change"])
        top_losers = sorted_by_loss[:3]

        sorted_by_gain = sorted(results, key=lambda x: x["projected_change"], reverse=True)
        top_gainers = [r for r in sorted_by_gain[:3] if r["projected_change"] > 0]

        # Calculate VaR under stress (simplified)
        var_95 = self._calculate_stressed_var(
            total_value,
            total_projected_pnl / total_value if total_value > 0 else 0
        )

        # Calculate expected shortfall (CVaR)
        cvar_95 = var_95 * 1.2  # Simplified approximation

        projected_pnl_pct = (total_projected_pnl / total_value * 100) if total_value > 0 else 0

        return {
            "scenario": self._serialize_scenario(scenario),
            "portfolio_value": round(total_value, 2),
            "projected_pnl": round(total_projected_pnl, 2),
            "projected_pnl_pct": round(projected_pnl_pct, 2),
            "projected_value": round(total_value + total_projected_pnl, 2),
            "var_95": round(abs(var_95), 2),
            "var_95_pct": round(abs(var_95) / total_value * 100, 2) if total_value > 0 else 0,
            "cvar_95": round(abs(cvar_95), 2),
            "top_losers": top_losers,
            "top_gainers": top_gainers,
            "position_impacts": results,
            "risk_level": self._get_risk_level(projected_pnl_pct),
            "computed_at": datetime.now().isoformat()
        }

    def run_predefined_scenario(
        self,
        positions: List[Dict],
        scenario_type: ScenarioType,
        current_prices: Dict[str, float],
        position_betas: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Run a predefined stress scenario."""
        scenario = PREDEFINED_SCENARIOS.get(scenario_type)
        if not scenario:
            return self._empty_stress_result(f"Unknown scenario: {scenario_type}")

        result = self.run_stress_test(
            positions, scenario, current_prices, position_betas
        )
        result["scenario_name"] = scenario_type.value
        return result

    def _calculate_position_impact(
        self,
        asset_type: str,
        sector: str,
        scenario: StressScenario,
        custom_beta: float = None
    ) -> float:
        """
        Calculate expected impact on a position from the stress scenario.

        Returns percentage change (e.g., -0.05 for -5%)
        """
        base_sensitivities = self.factor_sensitivities.get(
            asset_type, self.factor_sensitivities.get("stock")
        )

        # Ensure sector is a string (handle None case)
        sector_str = sector or ""

        # Get sector-specific adjustments
        sector_adj = {}
        for s, adj in SECTOR_SENSITIVITIES.items():
            if s in sector_str:
                for factor, mult in adj.items():
                    if factor in sector_adj:
                        sector_adj[factor] = max(sector_adj[factor], mult)
                    else:
                        sector_adj[factor] = mult

        total_impact = 0

        # Interest rate impact
        if scenario.interest_rate_change_bp != 0:
            base_sens = base_sensitivities.get("interest_rate", -0.1)
            sector_mult = sector_adj.get("interest_rate", 1.0)
            impact = (scenario.interest_rate_change_bp / 100) * base_sens * sector_mult
            total_impact += impact

        # FX impact
        if scenario.fx_change_pct != 0:
            base_sens = base_sensitivities.get("fx", 0.2)
            sector_mult = sector_adj.get("fx", 1.0)
            impact = (scenario.fx_change_pct / 100) * base_sens * sector_mult
            total_impact += impact

        # Index/market impact
        if scenario.index_change_pct != 0:
            if custom_beta is not None:
                beta = custom_beta
            else:
                base_beta = base_sensitivities.get("index", 1.0)
                sector_mult = sector_adj.get("index", 1.0)
                beta = base_beta * sector_mult

            impact = (scenario.index_change_pct / 100) * beta
            total_impact += impact

        # Oil impact
        if scenario.oil_change_pct != 0:
            base_sens = base_sensitivities.get("oil", 0.05)
            sector_mult = sector_adj.get("oil", 1.0)
            impact = (scenario.oil_change_pct / 100) * base_sens * sector_mult
            total_impact += impact

        # Sector-specific shocks
        if scenario.sector_shocks:
            for shock_sector, shock_pct in scenario.sector_shocks.items():
                if shock_sector.lower() in sector_str.lower():
                    total_impact += shock_pct / 100

        return total_impact

    def _calculate_stressed_var(
        self,
        total_value: float,
        expected_return: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate VaR under stress scenario.

        Uses a simplified approach:
        VaR = portfolio_value * (expected_loss + volatility_adjustment)
        """
        # Assume stressed volatility is higher
        stressed_vol = 0.02  # 2% daily vol under stress

        # VaR at 95% confidence (1.65 standard deviations)
        z_score = 1.65
        var_component = stressed_vol * z_score

        # Total VaR includes expected loss plus tail risk
        total_var = total_value * (abs(expected_return) + var_component)

        return total_var

    def _get_risk_level(self, pnl_pct: float) -> str:
        """Determine risk level based on projected P&L percentage."""
        if pnl_pct <= -10:
            return "critical"
        elif pnl_pct <= -5:
            return "high"
        elif pnl_pct <= -2:
            return "medium"
        elif pnl_pct < 0:
            return "low"
        else:
            return "positive"

    def _serialize_scenario(self, scenario: StressScenario) -> Dict:
        """Convert scenario to dictionary."""
        return {
            "interest_rate_change_bp": scenario.interest_rate_change_bp,
            "fx_change_pct": scenario.fx_change_pct,
            "index_change_pct": scenario.index_change_pct,
            "oil_change_pct": scenario.oil_change_pct,
            "sector_shocks": scenario.sector_shocks or {}
        }

    def _empty_stress_result(self, message: str = "No positions") -> Dict[str, Any]:
        """Return empty stress test result."""
        return {
            "scenario": {},
            "portfolio_value": 0,
            "projected_pnl": 0,
            "projected_pnl_pct": 0,
            "projected_value": 0,
            "var_95": 0,
            "var_95_pct": 0,
            "cvar_95": 0,
            "top_losers": [],
            "top_gainers": [],
            "position_impacts": [],
            "risk_level": "unknown",
            "message": message,
            "computed_at": datetime.now().isoformat()
        }

    @staticmethod
    def get_available_scenarios() -> List[Dict[str, Any]]:
        """Get list of available predefined scenarios for UI."""
        return [
            {
                "id": "fed_rate_hike_50bp",
                "name": "美联储加息 50bp",
                "description": "联邦基金利率上调50个基点",
                "category": "monetary_policy",
                "icon": "trending_up"
            },
            {
                "id": "fed_rate_cut_50bp",
                "name": "美联储降息 50bp",
                "description": "联邦基金利率下调50个基点",
                "category": "monetary_policy",
                "icon": "trending_down"
            },
            {
                "id": "cny_depreciation_2pct",
                "name": "人民币贬值 2%",
                "description": "人民币对美元贬值2%",
                "category": "currency",
                "icon": "currency_yuan"
            },
            {
                "id": "cny_appreciation_2pct",
                "name": "人民币升值 2%",
                "description": "人民币对美元升值2%",
                "category": "currency",
                "icon": "currency_yuan"
            },
            {
                "id": "index_drop_5pct",
                "name": "上证指数下跌 5%",
                "description": "大盘系统性下跌5%",
                "category": "market",
                "icon": "show_chart"
            },
            {
                "id": "index_drop_10pct",
                "name": "上证指数下跌 10%",
                "description": "大盘系统性下跌10%（极端情况）",
                "category": "market",
                "icon": "show_chart"
            },
            {
                "id": "semiconductor_crash_10pct",
                "name": "半导体板块回调 10%",
                "description": "半导体行业大幅回调",
                "category": "sector",
                "icon": "memory"
            },
            {
                "id": "oil_spike_20pct",
                "name": "油价飙升 20%",
                "description": "国际油价大涨，影响能源和运输板块",
                "category": "commodity",
                "icon": "local_gas_station"
            }
        ]

    @staticmethod
    def get_factor_sliders() -> List[Dict[str, Any]]:
        """Get slider configuration for custom scenarios."""
        return [
            {
                "id": "interest_rate_change_bp",
                "name": "利率变化 (基点)",
                "min": -100,
                "max": 100,
                "step": 10,
                "default": 0,
                "unit": "bp",
                "description": "影响成长股估值、银行股收益"
            },
            {
                "id": "fx_change_pct",
                "name": "汇率变化 (%)",
                "min": -5,
                "max": 5,
                "step": 0.5,
                "default": 0,
                "unit": "%",
                "description": "人民币升值为正，贬值为负"
            },
            {
                "id": "index_change_pct",
                "name": "指数变化 (%)",
                "min": -10,
                "max": 10,
                "step": 1,
                "default": 0,
                "unit": "%",
                "description": "大盘系统性涨跌"
            },
            {
                "id": "oil_change_pct",
                "name": "油价变化 (%)",
                "min": -20,
                "max": 20,
                "step": 5,
                "default": 0,
                "unit": "%",
                "description": "影响能源、航空、运输板块"
            }
        ]
