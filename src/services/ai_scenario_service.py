"""
AI Scenario Service for Stress Test Enhancement

Provides AI-powered scenario generation and conversational stress testing:
- Phase 1: Dynamic scenario parameter generation based on market context
- Phase 2: Natural language to scenario parameters conversion
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from src.llm.client import get_llm_client
from src.data_sources.akshare_api import get_market_indices


@dataclass
class AIScenario:
    """AI-generated stress test scenario."""
    id: str
    name: str
    parameters: Dict[str, float]
    reasoning: str
    confidence: str  # high, medium, low
    generated_at: str
    source: str = "ai"  # "ai" or "fallback"


@dataclass
class MarketContext:
    """Current market context for AI scenario generation."""
    indices: Dict[str, Any]
    usd_cny: Optional[float]
    timestamp: str


# Predefined fallback scenarios by category
FALLBACK_SCENARIOS = {
    "monetary_policy": AIScenario(
        id="fallback_rate_hike",
        name="Federal Reserve Rate Hike (Fallback)",
        parameters={
            "interest_rate_change_bp": 50,
            "index_change_pct": -2.0,
            "fx_change_pct": 0,
            "oil_change_pct": 0
        },
        reasoning="Default scenario: Typical Fed rate hike impact",
        confidence="medium",
        generated_at=datetime.now().isoformat(),
        source="fallback"
    ),
    "currency": AIScenario(
        id="fallback_cny_depreciation",
        name="CNY Depreciation (Fallback)",
        parameters={
            "interest_rate_change_bp": 0,
            "index_change_pct": -1.5,
            "fx_change_pct": -2.0,
            "oil_change_pct": 0
        },
        reasoning="Default scenario: Typical CNY depreciation impact",
        confidence="medium",
        generated_at=datetime.now().isoformat(),
        source="fallback"
    ),
    "market": AIScenario(
        id="fallback_market_crash",
        name="Market Correction (Fallback)",
        parameters={
            "interest_rate_change_bp": 0,
            "index_change_pct": -5.0,
            "fx_change_pct": 0,
            "oil_change_pct": 0
        },
        reasoning="Default scenario: Typical market correction",
        confidence="medium",
        generated_at=datetime.now().isoformat(),
        source="fallback"
    ),
    "sector": AIScenario(
        id="fallback_tech_crash",
        name="Tech Sector Pullback (Fallback)",
        parameters={
            "interest_rate_change_bp": 25,
            "index_change_pct": -3.0,
            "fx_change_pct": 0,
            "oil_change_pct": 0
        },
        reasoning="Default scenario: Tech sector rotation",
        confidence="medium",
        generated_at=datetime.now().isoformat(),
        source="fallback"
    ),
    "commodity": AIScenario(
        id="fallback_oil_spike",
        name="Oil Price Spike (Fallback)",
        parameters={
            "interest_rate_change_bp": 0,
            "index_change_pct": -1.0,
            "fx_change_pct": 0,
            "oil_change_pct": 15.0
        },
        reasoning="Default scenario: Typical oil supply shock",
        confidence="medium",
        generated_at=datetime.now().isoformat(),
        source="fallback"
    )
}


CATEGORY_NAMES = {
    "monetary_policy": "Monetary Policy",
    "currency": "Currency/FX",
    "market": "Market Risk",
    "sector": "Sector Risk",
    "commodity": "Commodity Risk"
}


class AIScenarioService:
    """
    Service for AI-powered stress test scenario generation.
    """

    def __init__(self):
        self._llm_client = None

    def _get_llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def _fetch_market_context(self) -> MarketContext:
        """Fetch current market data for context."""
        try:
            indices = get_market_indices()

            # Extract USD/CNY if available
            usd_cny = None
            if indices and "forex" in indices:
                usd_cny = indices["forex"].get("usd_cny")

            return MarketContext(
                indices=indices or {},
                usd_cny=usd_cny,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Error fetching market context: {e}")
            return MarketContext(
                indices={},
                usd_cny=None,
                timestamp=datetime.now().isoformat()
            )

    def _build_scenario_prompt(self, category: str, market_context: MarketContext) -> str:
        """Build prompt for AI scenario generation."""
        category_name = CATEGORY_NAMES.get(category, category)

        # Format market data
        market_summary = []
        indices = market_context.indices

        if indices:
            for key, data in indices.items():
                if isinstance(data, dict) and "close" in data:
                    name = data.get("name", key)
                    close = data.get("close", 0)
                    change_pct = data.get("pct_change", 0)
                    market_summary.append(f"{name}: {close:.2f} ({change_pct:+.2f}%)")

        market_str = "\n".join(market_summary) if market_summary else "Market data unavailable"

        if market_context.usd_cny:
            market_str += f"\nUSD/CNY: {market_context.usd_cny:.4f}"

        prompt = f"""You are a senior macro analyst. Based on the following market data, generate a realistic stress test scenario for the "{category_name}" category.

## Current Market Data
{market_str}

## Task
Generate a plausible stress scenario with specific numerical parameters. Consider recent market conditions and historical precedents.

## Output Format (JSON only, no markdown)
{{
  "scenario_name": "Descriptive name (e.g., 'Fed Rate Hike 75bp')",
  "parameters": {{
    "interest_rate_change_bp": <number between -100 and 100>,
    "fx_change_pct": <number between -5 and 5>,
    "index_change_pct": <number between -10 and 10>,
    "oil_change_pct": <number between -20 and 20>
  }},
  "reasoning": "Brief analysis (1-2 sentences) explaining why these parameters are appropriate given current market conditions",
  "confidence": "high" | "medium" | "low"
}}

Important:
- For "{category_name}" scenarios, emphasize the most relevant parameters
- Base your reasoning on actual market conditions
- Be specific with numbers, not ranges
- Output ONLY valid JSON, no additional text"""

        return prompt

    async def generate_scenario(self, category: str) -> Dict[str, Any]:
        """
        Generate AI-powered stress test scenario for a category.

        Args:
            category: Scenario category (monetary_policy, currency, market, sector, commodity)

        Returns:
            Dict with scenario and market_context
        """
        market_context = self._fetch_market_context()

        try:
            # Build and execute prompt
            prompt = self._build_scenario_prompt(category, market_context)
            client = self._get_llm_client()

            # Run LLM call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                client.generate_content,
                prompt
            )

            # Parse response
            scenario = self._parse_scenario_response(response, category)

            return {
                "scenario": asdict(scenario),
                "market_context": asdict(market_context)
            }

        except Exception as e:
            print(f"AI scenario generation failed: {e}")
            # Return fallback scenario
            fallback = FALLBACK_SCENARIOS.get(category, FALLBACK_SCENARIOS["market"])
            fallback.generated_at = datetime.now().isoformat()

            return {
                "scenario": asdict(fallback),
                "market_context": asdict(market_context),
                "error": str(e)
            }

    def _parse_scenario_response(self, response: str, category: str) -> AIScenario:
        """Parse LLM response into AIScenario."""
        try:
            # Clean response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove opening ```json or ```
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            data = json.loads(cleaned)

            # Validate and extract parameters
            params = data.get("parameters", {})
            validated_params = {
                "interest_rate_change_bp": self._clamp(params.get("interest_rate_change_bp", 0), -100, 100),
                "fx_change_pct": self._clamp(params.get("fx_change_pct", 0), -5, 5),
                "index_change_pct": self._clamp(params.get("index_change_pct", 0), -10, 10),
                "oil_change_pct": self._clamp(params.get("oil_change_pct", 0), -20, 20),
            }

            scenario_name = data.get("scenario_name", f"AI Scenario - {category}")

            return AIScenario(
                id=f"ai_{category}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name=scenario_name,
                parameters=validated_params,
                reasoning=data.get("reasoning", "AI-generated scenario"),
                confidence=data.get("confidence", "medium"),
                generated_at=datetime.now().isoformat(),
                source="ai"
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            raise ValueError(f"Failed to parse AI response: {e}")

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range."""
        try:
            v = float(value)
            return max(min_val, min(max_val, v))
        except (TypeError, ValueError):
            return 0.0


class StressTestChatService:
    """
    Service for conversational stress testing.
    Converts natural language queries to stress test parameters.
    """

    def __init__(self):
        self._llm_client = None

    def _get_llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def _build_chat_prompt(
        self,
        message: str,
        history: List[Dict[str, str]],
        portfolio_summary: Dict[str, Any]
    ) -> str:
        """Build prompt for chat-based stress testing."""

        # Format portfolio summary
        portfolio_str = f"""Portfolio Value: {portfolio_summary.get('total_value', 0):,.2f}
Position Count: {portfolio_summary.get('position_count', 0)}
Current P&L: {portfolio_summary.get('total_pnl_pct', 0):.2f}%"""

        # Format history
        history_str = ""
        if history:
            for msg in history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_str += f"{role.upper()}: {content}\n"

        prompt = f"""You are a portfolio risk analyst assistant. Help the user understand stress test scenarios.

## User's Portfolio
{portfolio_str}

## Conversation History
{history_str}

## User Question
{message}

## Your Task
1. If the user is asking about a stress scenario (e.g., "what if rates rise 50bp"), extract parameters and provide analysis
2. If asking for general advice, provide risk management insights
3. Always be specific with numbers and explain the impact on their portfolio

## Response Format (JSON)
{{
  "response": "Your natural language response explaining the scenario and its impact",
  "scenario_params": {{
    "interest_rate_change_bp": <number or null>,
    "fx_change_pct": <number or null>,
    "index_change_pct": <number or null>,
    "oil_change_pct": <number or null>
  }},
  "should_run_stress_test": true/false,
  "suggested_followups": ["Question 1", "Question 2", "Question 3"]
}}

Output ONLY valid JSON, no additional text."""

        return prompt

    async def chat(
        self,
        message: str,
        portfolio_id: int,
        portfolio_summary: Dict[str, Any],
        history: List[Dict[str, str]] = None,
        stress_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message for stress testing.

        Args:
            message: User's message
            portfolio_id: Portfolio ID for context
            portfolio_summary: Portfolio summary data
            history: Previous chat messages
            stress_result: Previous stress test result for context

        Returns:
            Chat response with optional stress test parameters
        """
        history = history or []

        try:
            prompt = self._build_chat_prompt(message, history, portfolio_summary)
            client = self._get_llm_client()

            # Run LLM call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                client.generate_content,
                prompt
            )

            return self._parse_chat_response(response, stress_result)

        except Exception as e:
            print(f"Stress test chat failed: {e}")
            return {
                "response": f"Sorry, I encountered an error processing your request: {str(e)}",
                "scenario_params": None,
                "should_run_stress_test": False,
                "suggested_followups": [
                    "What if interest rates rise?",
                    "How would a market crash affect my portfolio?",
                    "What's my biggest risk exposure?"
                ],
                "error": str(e)
            }

    def _parse_chat_response(
        self,
        response: str,
        stress_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Parse chat response from LLM."""
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            # Extract and validate scenario params
            params = data.get("scenario_params")
            if params:
                validated_params = {}
                for key in ["interest_rate_change_bp", "fx_change_pct", "index_change_pct", "oil_change_pct"]:
                    val = params.get(key)
                    if val is not None:
                        validated_params[key] = float(val)
                params = validated_params if validated_params else None

            return {
                "response": data.get("response", "I can help you analyze stress scenarios."),
                "scenario_params": params,
                "should_run_stress_test": data.get("should_run_stress_test", False),
                "suggested_followups": data.get("suggested_followups", []),
                "stress_result": stress_result
            }

        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return {
                "response": response,
                "scenario_params": None,
                "should_run_stress_test": False,
                "suggested_followups": [],
            }

    def _interpret_stress_result(self, result: Dict[str, Any]) -> str:
        """Generate natural language interpretation of stress test result."""
        if not result:
            return ""

        pnl_pct = result.get("projected_pnl_pct", 0)
        risk_level = result.get("risk_level", "unknown")
        var_95 = result.get("var_95_pct", 0)

        if pnl_pct < 0:
            direction = "lose"
            amount = abs(pnl_pct)
        else:
            direction = "gain"
            amount = pnl_pct

        interpretation = f"Under this scenario, your portfolio would {direction} approximately {amount:.1f}%. "

        if risk_level == "critical":
            interpretation += "This represents a critical risk level - consider hedging strategies."
        elif risk_level == "high":
            interpretation += "This is a high-risk scenario that warrants attention."
        elif risk_level == "medium":
            interpretation += "This represents moderate risk within acceptable limits."
        else:
            interpretation += "This is within your portfolio's normal risk tolerance."

        if var_95 > 0:
            interpretation += f" The 95% VaR suggests maximum potential loss of {var_95:.1f}%."

        return interpretation


# Singleton instances
ai_scenario_service = AIScenarioService()
stress_test_chat_service = StressTestChatService()
