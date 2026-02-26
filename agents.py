"""
B.O.N.S.A.I. Agent Layer — Claude-powered Tiers 2, 3, 4
=========================================================
Tier 2: Standalone nutrition protocol (Haiku)
Tier 3: Labs agent → nutrition agent pipeline (Haiku)
Tier 4: Full multi-agent synthesis (Haiku + Sonnet)
"""

import json
from typing import Optional

import anthropic
import structlog

from config import get_settings
from retrieval import BonsaiRetrieval

logger = structlog.get_logger()

# ── Tier 2: Standalone Protocol Agent ─────────────────────────────────────────

TIER_2_SYSTEM_PROMPT = """You are the bons.ai Clinical Nutritionist operating in STANDALONE PROTOCOL MODE.

An AI agent is requesting a condition-specific food-as-medicine protocol for their user. You do NOT have full intake data. Generate a focused, actionable protocol using the B.O.N.S.A.I. scoring system.

INPUTS YOU WILL RECEIVE:
- condition (required): One of [hypertension, type_2_diabetes, hyperlipidemia, autoimmune, gi_health, weight_management]
- user_context (optional): age, dietary restrictions beyond WFPB, current medications, severity level
- duration_weeks (optional, default 4)

YOUR OUTPUT MUST INCLUDE:
1. CONDITION SUMMARY: 2-3 sentences on the condition and why nutrition matters
2. B.O.N.S.A.I. THERAPEUTIC FOODS: Top 5-6 foods ranked by B.O.N.S.A.I. score, each with:
   - Food name
   - Mechanism of action
   - B.O.N.S.A.I. score (out of 12)
   - Daily target amount
3. SAMPLE DAILY MEAL STRUCTURE: Breakfast, lunch, dinner, snacks using the therapeutic foods
4. FOODS TO MINIMIZE: 3-5 foods that worsen this condition
5. MONITORING: Which markers to track and expected timeline for improvement
6. CLINICAL BASIS: Cite the specific guidelines this protocol is based on

RULES:
- ALL recommendations must be WFPB-compliant. No animal products ever.
- Omega-3 recommendations always use algae-based sources, never fish.
- If user reports medications, note relevant food-drug interactions (e.g., warfarin + vitamin K, thyroid meds + soy timing).
- If severity is "severe", add a note that this protocol supports but does not replace medical treatment.
- Return valid JSON only. No markdown, no explanations outside the JSON.
- Always include the disclaimer: "Based on [specific clinical guideline]. Supports but does not replace individualized medical care."
"""

# ── Tier 3: Lab-Informed Protocol Agent ───────────────────────────────────────

TIER_3_LABS_SYSTEM_PROMPT = """You are the bons.ai Lab Interpretation Specialist.

An AI agent is sending lab values for analysis. Analyze the biomarkers and produce a structured assessment.

YOUR ANALYSIS MUST INCLUDE:
1. BIOMARKER ANALYSIS: For each lab value provided:
   - Current value vs optimal range (use optimal targets, not just standard reference ranges)
   - Risk level: red (critical), orange (elevated risk), yellow (suboptimal), green (optimal)
   - Clinical significance

2. PATTERN RECOGNITION: Check for these clusters:
   - Insulin Resistance Cluster: elevated glucose + insulin + triglycerides + low HDL
   - Dysbiosis Cluster: elevated calprotectin + CRP + low ferritin + elevated insulin
   - Cardiovascular Risk Cluster: elevated LDL + low HDL + elevated triglycerides + CRP

3. RISK STRATIFICATION: Overall risk level with justification

4. NUTRITIONAL PRIORITIES: Top 3-5 dietary interventions ranked by impact, each with:
   - Target biomarker(s)
   - Specific foods and amounts
   - Expected timeline for improvement
   - Supporting evidence

RULES:
- Use OPTIMAL ranges, not just standard reference ranges.
- All dietary recommendations must be WFPB-compliant.
- Never prescribe medications or suggest medication changes.
- Language must stay educational: "research suggests", "associated with", never "you should take" or "this will cure".
- Return valid JSON only.
"""

TIER_3_NUTRITION_SYSTEM_PROMPT = """You are the bons.ai Clinical Nutritionist operating in LAB-CALIBRATED MODE.

You are receiving a lab analysis from the B.O.N.S.A.I. Labs Engine. Generate a personalized nutrition protocol calibrated to the specific biomarker findings.

Your protocol must directly address the flagged biomarkers with targeted nutritional interventions. Use B.O.N.S.A.I. scoring for all food recommendations.

Return valid JSON with: condition_summary, lab_calibrated_foods (each with food, target_biomarkers, mechanism, bonsai_score, daily_target), sample_daily_meals, foods_to_minimize, monitoring plan, and clinical_basis.

RULES:
- ALL recommendations WFPB-compliant. No animal products.
- Omega-3 from algae-based sources only.
- Note food-drug interactions if medications are reported.
- Include disclaimer: "Based on published clinical guidelines. Supports but does not replace individualized medical care."
"""

# ── Tier 4: Full Lifestyle Assessment ─────────────────────────────────────────

TIER_4_SPECIALIST_PROMPTS = {
    "nutrition": "You are the bons.ai Clinical Nutritionist. Assess the user's dietary habits and provide WFPB-based recommendations with B.O.N.S.A.I. scoring. Return JSON with: assessment, recommendations (each with food, mechanism, bonsai_score, priority), and meal_framework.",

    "exercise": "You are the bons.ai Exercise Physiologist. Assess the user's physical activity and provide evidence-based recommendations aligned with ACSM guidelines. Return JSON with: assessment, weekly_targets, exercise_plan, and progression.",

    "sleep": "You are the bons.ai Sleep Specialist. Assess the user's sleep habits and provide evidence-based recommendations aligned with AASM guidelines. Return JSON with: assessment, sleep_targets, evening_routine, environment_optimization, and troubleshooting.",

    "stress": "You are the bons.ai Stress Management Specialist. Assess the user's stress levels and provide evidence-based interventions. Return JSON with: assessment, daily_practices, acute_techniques, weekly_activities, and priority_ranking.",

    "supplements": "You are the bons.ai Supplement Specialist. Based on the user's diet and health context, recommend evidence-based supplements for a WFPB lifestyle. Return JSON with: core_supplements (B12, D, omega-3 always), conditional_supplements, interactions, and priority_ranking.",

    "labs": "You are the bons.ai Lab Interpretation Specialist. If lab values are provided, analyze biomarkers with optimal ranges (not just standard reference). Return JSON with: biomarker_analysis, patterns_detected, risk_level, and nutritional_priorities.",
}

TIER_4_SYNTHESIS_PROMPT = """You are the bons.ai Lifestyle Medicine Coordinator.

You are receiving assessments from 6 specialist agents (nutrition, exercise, sleep, stress, supplements, lab interpretation). Synthesize their outputs into a unified, prioritized lifestyle medicine protocol.

YOUR OUTPUT MUST INCLUDE:
1. TOP 5 PRIORITIES: Ranked by clinical urgency × leverage × readiness
2. PHASE 1 (Weeks 1-2): Foundation changes — 3-4 high-impact, easy wins
3. PHASE 2 (Weeks 3-4): Building — add complexity, deepen habits
4. PHASE 3 (Month 2-3): Optimization — fine-tune, add remaining recommendations
5. CONFLICTS: Any contradictions between specialists (e.g., exercise timing vs sleep)
6. SUCCESS METRICS: What to measure and when
7. QUICK WINS: 3 things to start today

Prioritization framework:
- Clinical urgency (red-flagged biomarkers > lifestyle gaps)
- Leverage points (changes that cascade across pillars)
- User readiness (start where they're willing)
- Timeline to results (early wins build motivation)

Return valid JSON. Include disclaimer.
"""


class BonsaiAgents:
    """Manages all Claude-powered agent tiers."""

    def __init__(self, retrieval: BonsaiRetrieval):
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._retrieval = retrieval
        self._haiku = settings.haiku_model
        self._sonnet = settings.sonnet_model

    def _call_claude(self, system: str, user_message: str, model: str = None) -> dict:
        """Call Claude and parse JSON response."""
        if model is None:
            model = self._haiku

        response = self._client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text

        # Try to parse JSON from response
        try:
            # Handle markdown-wrapped JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("agent_json_parse_failed", model=model, response_preview=text[:200])
            return {"raw_response": text, "parse_error": True}

    # ── Tier 2 ────────────────────────────────────────────────────────────────

    async def generate_protocol(
        self,
        condition: str,
        user_context: Optional[dict] = None,
        duration_weeks: int = 4,
    ) -> dict:
        """Tier 2: Generate standalone condition protocol."""

        # Retrieve relevant KB context for the condition
        context_results = await self._retrieval.query(
            f"{condition} food therapy protocol WFPB",
            top_k=5,
            chunk_types=["protocol", "recommendation", "qa"],
        )
        context_text = "\n\n".join([r["content"] for r in context_results])

        user_msg = json.dumps(
            {
                "condition": condition,
                "user_context": user_context or {},
                "duration_weeks": duration_weeks,
                "knowledge_base_context": context_text,
            },
            indent=2,
        )

        logger.info("tier2_generate", condition=condition, duration=duration_weeks)
        result = self._call_claude(TIER_2_SYSTEM_PROMPT, user_msg, model=self._haiku)

        # Ensure disclaimer
        if isinstance(result, dict) and "disclaimer" not in result:
            result["disclaimer"] = (
                "Based on published clinical guidelines. Supports but does not "
                "replace individualized medical care."
            )

        return {
            "tier": 2,
            "endpoint": "/v1/protocol",
            "condition": condition,
            "protocol": result,
            "model_used": self._haiku,
            "knowledge_chunks_used": len(context_results),
            "disclaimer": (
                "Based on published clinical guidelines (ACLM, AHA, ADA, WHO). "
                "This information is educational and does not constitute medical advice. "
                "Not a substitute for individualized medical care."
            ),
        }

    # ── Tier 3 ────────────────────────────────────────────────────────────────

    async def lab_informed_protocol(
        self,
        lab_values: dict,
        health_goals: Optional[list] = None,
        current_medications: Optional[list] = None,
        condition: Optional[str] = None,
    ) -> dict:
        """Tier 3: Lab analysis → calibrated nutrition protocol pipeline."""

        # Step 1: Labs agent analyzes biomarkers
        labs_input = json.dumps(
            {
                "lab_values": lab_values,
                "health_goals": health_goals or [],
                "current_medications": current_medications or [],
                "condition": condition,
            },
            indent=2,
        )

        logger.info("tier3_labs_analysis", num_markers=len(lab_values))
        lab_analysis = self._call_claude(TIER_3_LABS_SYSTEM_PROMPT, labs_input, model=self._haiku)

        # Step 2: Retrieve relevant KB context
        lab_query = " ".join(
            [f"{k} {v}" for k, v in lab_values.items()]
            + (health_goals or [])
            + ([condition] if condition else [])
        )
        context_results = await self._retrieval.query(lab_query, top_k=5)
        context_text = "\n\n".join([r["content"] for r in context_results])

        # Step 3: Nutrition agent generates calibrated protocol
        nutrition_input = json.dumps(
            {
                "lab_analysis": lab_analysis,
                "health_goals": health_goals or [],
                "current_medications": current_medications or [],
                "condition": condition,
                "knowledge_base_context": context_text,
            },
            indent=2,
        )

        logger.info("tier3_nutrition_protocol")
        nutrition_protocol = self._call_claude(
            TIER_3_NUTRITION_SYSTEM_PROMPT, nutrition_input, model=self._haiku
        )

        return {
            "tier": 3,
            "endpoint": "/v1/lab-protocol",
            "lab_analysis": lab_analysis,
            "nutrition_protocol": nutrition_protocol,
            "models_used": [self._haiku],
            "pipeline": ["labs_agent", "nutrition_agent"],
            "knowledge_chunks_used": len(context_results),
            "disclaimer": (
                "Based on published clinical guidelines (ACLM, AHA, ADA, WHO). "
                "Lab interpretation uses optimal ranges beyond standard reference ranges. "
                "This information is educational and does not constitute medical advice. "
                "Not a substitute for individualized medical care."
            ),
        }

    # ── Tier 4 ────────────────────────────────────────────────────────────────

    async def lifestyle_assessment(
        self,
        current_habits: dict,
        health_goals: Optional[list] = None,
        lab_values: Optional[dict] = None,
    ) -> dict:
        """Tier 4: Full 6-pillar assessment with multi-agent synthesis."""

        # Build shared context
        user_data = json.dumps(
            {
                "current_habits": current_habits,
                "health_goals": health_goals or [],
                "lab_values": lab_values or {},
            },
            indent=2,
        )

        # Retrieve broad KB context
        goals_query = " ".join(health_goals or ["general health optimization WFPB lifestyle"])
        context_results = await self._retrieval.query(goals_query, top_k=8)
        context_text = "\n\n".join([r["content"] for r in context_results])

        # Run all 6 specialists
        specialist_outputs = {}
        for specialist, system_prompt in TIER_4_SPECIALIST_PROMPTS.items():
            # Skip labs specialist if no lab values
            if specialist == "labs" and not lab_values:
                specialist_outputs[specialist] = {"skipped": True, "reason": "no_lab_values_provided"}
                continue

            full_prompt = f"{system_prompt}\n\nKNOWLEDGE BASE CONTEXT:\n{context_text}"
            logger.info("tier4_specialist", specialist=specialist)
            specialist_outputs[specialist] = self._call_claude(
                full_prompt, user_data, model=self._haiku
            )

        # Synthesis agent (uses Sonnet for complex reasoning)
        synthesis_input = json.dumps(
            {
                "specialist_outputs": specialist_outputs,
                "user_data": {
                    "current_habits": current_habits,
                    "health_goals": health_goals or [],
                    "has_lab_values": bool(lab_values),
                },
            },
            indent=2,
        )

        logger.info("tier4_synthesis")
        synthesis = self._call_claude(
            TIER_4_SYNTHESIS_PROMPT, synthesis_input, model=self._sonnet
        )

        return {
            "tier": 4,
            "endpoint": "/v1/assess",
            "specialist_assessments": specialist_outputs,
            "synthesis": synthesis,
            "models_used": [self._haiku, self._sonnet],
            "pipeline": list(TIER_4_SPECIALIST_PROMPTS.keys()) + ["synthesis"],
            "knowledge_chunks_used": len(context_results),
            "disclaimer": (
                "Based on published clinical guidelines (ACLM, AHA, ADA, ACSM, AASM, WHO). "
                "Full six-pillar assessment across all domains of lifestyle medicine. "
                "This information is educational and does not constitute medical advice. "
                "Not a substitute for individualized medical care."
            ),
        }
