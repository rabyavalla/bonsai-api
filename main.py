"""
B.O.N.S.A.I. Health Intelligence API
======================================
Agent economy layer — 4 paid endpoints via x402 (USDC on Base) + MCP.

Endpoints:
    GET  /v1/query          $0.02  Tier 1: Knowledge query (vector search)
    POST /v1/protocol       $0.20  Tier 2: Condition protocol (standalone nutrition agent)
    POST /v1/lab-protocol   $0.45  Tier 3: Lab-informed protocol (labs + nutrition pipeline)
    POST /v1/assess         $0.85  Tier 4: Full lifestyle assessment (6 agents + synthesis)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

MCP Hive target: March 8, 2026
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import get_settings
from retrieval import BonsaiRetrieval
from agents import BonsaiAgents
from payment import X402PaymentMiddleware

# ── Structured logging ────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_level_from_name(get_settings().log_level)
    ),
)
logger = structlog.get_logger()

# ── Metrics (in-memory; swap for Prometheus/Datadog in production) ────────────

metrics = {
    "requests_total": 0,
    "requests_by_tier": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0},
    "errors_total": 0,
    "latency_sum_ms": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0},
    "latency_count": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0},
    "payments_verified": 0,
    "payments_failed": 0,
    "revenue_usdc": 0.0,
}

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Lifespan ──────────────────────────────────────────────────────────────────

retrieval: Optional[BonsaiRetrieval] = None
agents: Optional[BonsaiAgents] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval, agents
    logger.info("startup", msg="Initializing B.O.N.S.A.I. API...")
    retrieval = BonsaiRetrieval()
    agents = BonsaiAgents(retrieval)
    logger.info("startup_complete", msg="Retrieval and agents ready.")
    yield
    logger.info("shutdown", msg="B.O.N.S.A.I. API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="B.O.N.S.A.I. Health Intelligence API",
    version="3.0.0",
    description=(
        "Evidence-based health intelligence for AI agents. "
        "WFPB lifestyle medicine backed by ACLM, AHA, ADA, WHO, and 637+ peer-reviewed references. "
        "Payments via x402 protocol (USDC on Base)."
    ),
    lifespan=lifespan,
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(X402PaymentMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────────────


class ProtocolRequest(BaseModel):
    condition: str = Field(
        ...,
        description="Health condition",
        json_schema_extra={
            "enum": [
                "hypertension",
                "type_2_diabetes",
                "hyperlipidemia",
                "autoimmune",
                "gi_health",
                "weight_management",
            ]
        },
    )
    user_context: Optional[dict] = Field(default=None, description="Optional user context")
    duration_weeks: int = Field(default=4, ge=1, le=12)


class LabProtocolRequest(BaseModel):
    lab_values: dict = Field(..., description="Biomarker key-value pairs (e.g. {'LDL': '145 mg/dL'})")
    health_goals: Optional[list[str]] = None
    current_medications: Optional[list[str]] = None
    condition: Optional[str] = None


class AssessmentRequest(BaseModel):
    current_habits: dict = Field(..., description="User habits across lifestyle pillars")
    health_goals: Optional[list[str]] = None
    lab_values: Optional[dict] = None


# ── Utility ───────────────────────────────────────────────────────────────────


def track_request(tier: str, latency_ms: float, price: float):
    """Update in-memory metrics."""
    metrics["requests_total"] += 1
    metrics["requests_by_tier"][tier] += 1
    metrics["latency_sum_ms"][tier] += latency_ms
    metrics["latency_count"][tier] += 1
    metrics["payments_verified"] += 1
    metrics["revenue_usdc"] += price


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/v1/health")
async def health_check():
    """Health check — no payment required."""
    return {
        "status": "healthy",
        "service": "B.O.N.S.A.I. Health Intelligence API",
        "version": "3.0.0",
        "framework": "B.O.N.S.A.I.",
        "dietary_philosophy": "WFPB",
    }


@app.get("/v1/query")
@limiter.limit("100/minute")
async def knowledge_query(
    request: Request,
    q: str = Query(..., description="Natural language health question"),
    pillar: Optional[str] = Query(
        default=None,
        description="Filter by lifestyle medicine pillar",
    ),
    limit: int = Query(default=3, ge=1, le=10),
    include_citations: bool = Query(default=True),
):
    """
    Tier 1 — Knowledge Query ($0.02)

    Semantic search across the B.O.N.S.A.I. WFPB health intelligence knowledge base.
    Returns evidence-based answers with clinical citations.
    """
    start = time.time()

    logger.info("tier1_query", query=q, pillar=pillar, limit=limit)

    results = await retrieval.query(question=q, top_k=limit, pillar=pillar)
    response = retrieval.format_tier1_response(results, q, include_citations)

    latency = (time.time() - start) * 1000
    track_request("tier_1", latency, 0.02)
    logger.info("tier1_complete", results=len(results), latency_ms=round(latency, 1))

    return response


@app.post("/v1/protocol")
@limiter.limit("100/minute")
async def generate_protocol(request: Request, body: ProtocolRequest):
    """
    Tier 2 — Condition Protocol ($0.20)

    Generate a B.O.N.S.A.I.-scored food-as-medicine protocol for a specific health condition.
    Uses the nutrition specialist agent in standalone mode.
    """
    start = time.time()

    valid_conditions = [
        "hypertension", "type_2_diabetes", "hyperlipidemia",
        "autoimmune", "gi_health", "weight_management",
    ]
    if body.condition not in valid_conditions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid condition. Must be one of: {valid_conditions}",
        )

    logger.info("tier2_protocol", condition=body.condition, duration=body.duration_weeks)

    result = await agents.generate_protocol(
        condition=body.condition,
        user_context=body.user_context,
        duration_weeks=body.duration_weeks,
    )

    latency = (time.time() - start) * 1000
    track_request("tier_2", latency, 0.20)
    logger.info("tier2_complete", condition=body.condition, latency_ms=round(latency, 1))

    return result


@app.post("/v1/lab-protocol")
@limiter.limit("100/minute")
async def lab_informed_protocol(request: Request, body: LabProtocolRequest):
    """
    Tier 3 — Lab-Informed Protocol ($0.45)

    Generate a biomarker-calibrated dietary protocol. Sends lab values through
    the B.O.N.S.A.I. labs engine, then generates a personalized nutrition protocol.
    """
    start = time.time()

    if not body.lab_values:
        raise HTTPException(status_code=400, detail="lab_values is required and must not be empty")

    logger.info("tier3_lab_protocol", num_markers=len(body.lab_values))

    result = await agents.lab_informed_protocol(
        lab_values=body.lab_values,
        health_goals=body.health_goals,
        current_medications=body.current_medications,
        condition=body.condition,
    )

    latency = (time.time() - start) * 1000
    track_request("tier_3", latency, 0.45)
    logger.info("tier3_complete", latency_ms=round(latency, 1))

    return result


@app.post("/v1/assess")
@limiter.limit("100/minute")
async def lifestyle_assessment(request: Request, body: AssessmentRequest):
    """
    Tier 4 — Full Lifestyle Assessment ($0.85)

    Complete six-pillar B.O.N.S.A.I. lifestyle assessment. Runs 6 specialist agents
    (nutrition, exercise, sleep, stress, supplements, labs) plus a synthesis agent.
    """
    start = time.time()

    if not body.current_habits:
        raise HTTPException(status_code=400, detail="current_habits is required")

    logger.info("tier4_assessment", has_labs=bool(body.lab_values))

    result = await agents.lifestyle_assessment(
        current_habits=body.current_habits,
        health_goals=body.health_goals,
        lab_values=body.lab_values,
    )

    latency = (time.time() - start) * 1000
    track_request("tier_4", latency, 0.85)
    logger.info("tier4_complete", latency_ms=round(latency, 1))

    return result


# ── Monitoring ────────────────────────────────────────────────────────────────


@app.get("/v1/metrics")
async def get_metrics():
    """Internal metrics endpoint (protect with API key in production)."""
    avg_latency = {}
    for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
        count = metrics["latency_count"][tier]
        avg_latency[tier] = round(metrics["latency_sum_ms"][tier] / count, 1) if count > 0 else 0

    return {
        "requests_total": metrics["requests_total"],
        "requests_by_tier": metrics["requests_by_tier"],
        "average_latency_ms": avg_latency,
        "payments_verified": metrics["payments_verified"],
        "payments_failed": metrics["payments_failed"],
        "revenue_usdc": round(metrics["revenue_usdc"], 2),
    }


# ── MCP Server Info ──────────────────────────────────────────────────────────


@app.get("/.well-known/mcp.json")
async def mcp_manifest():
    """MCP tool discovery endpoint."""
    return {
        "schema_version": "2024-11-05",
        "server": {
            "name": "bonsai-health-intelligence",
            "version": "3.0.0",
            "description": (
                "B.O.N.S.A.I. — Evidence-based WFPB lifestyle medicine intelligence. "
                "Backed by ACLM, AHA, ADA, WHO, and 637+ peer-reviewed references."
            ),
        },
        "tools": [
            {
                "name": "health_query",
                "description": "Ask a health, nutrition, or lifestyle medicine question. Returns evidence-based answers with clinical citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language health question"},
                        "pillar": {
                            "type": "string",
                            "enum": ["nutrition", "physical_activity", "sleep", "stress_management", "substance_avoidance", "social_connection", "any"],
                            "default": "any",
                        },
                        "max_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
                "annotations": {"pricing": {"amount": "0.02", "currency": "USDC", "protocol": "x402"}},
            },
            {
                "name": "generate_protocol",
                "description": "Generate a B.O.N.S.A.I.-scored food-as-medicine protocol for a specific health condition.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "enum": ["hypertension", "type_2_diabetes", "hyperlipidemia", "autoimmune", "gi_health", "weight_management"],
                        },
                        "user_context": {"type": "object"},
                        "duration_weeks": {"type": "integer", "default": 4},
                    },
                    "required": ["condition"],
                },
                "annotations": {"pricing": {"amount": "0.20", "currency": "USDC", "protocol": "x402"}},
            },
            {
                "name": "lab_informed_protocol",
                "description": "Generate a biomarker-calibrated dietary protocol using the B.O.N.S.A.I. labs engine.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lab_values": {"type": "object", "description": "Key-value pairs of biomarker names and values"},
                        "health_goals": {"type": "array", "items": {"type": "string"}},
                        "current_medications": {"type": "array", "items": {"type": "string"}},
                        "condition": {"type": "string"},
                    },
                    "required": ["lab_values"],
                },
                "annotations": {"pricing": {"amount": "0.45", "currency": "USDC", "protocol": "x402"}},
            },
            {
                "name": "lifestyle_assessment",
                "description": "Full six-pillar B.O.N.S.A.I. lifestyle assessment with personalized recommendations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "current_habits": {"type": "object"},
                        "health_goals": {"type": "array", "items": {"type": "string"}},
                        "lab_values": {"type": "object"},
                    },
                    "required": ["current_habits"],
                },
                "annotations": {"pricing": {"amount": "0.85", "currency": "USDC", "protocol": "x402"}},
            },
        ],
    }
