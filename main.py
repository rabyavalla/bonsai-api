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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
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
        20
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

# ── Uptime tracking ──────────────────────────────────────────────────────────

STARTED_AT = datetime.now(timezone.utc)


# ── Request logging middleware ───────────────────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, latency, and client IP."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            latency_ms = round((time.time() - start) * 1000, 1)
            logger.info(
                "request",
                method=method,
                path=path,
                status=response.status_code,
                latency_ms=latency_ms,
                client_ip=client_ip,
            )
            return response
        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 1)
            metrics["errors_total"] += 1
            logger.error(
                "request_error",
                method=method,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=latency_ms,
                client_ip=client_ip,
            )
            raise


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Lifespan ──────────────────────────────────────────────────────────────────

retrieval: Optional[BonsaiRetrieval] = None
agents: Optional[BonsaiAgents] = None


def _get_retrieval_and_agents() -> tuple[BonsaiRetrieval, BonsaiAgents]:
    """Lazy-initialize retrieval and agents on first request."""
    global retrieval, agents
    if retrieval is None:
        logger.info("lazy_init", msg="Initializing retrieval and agents on first request...")
        retrieval = BonsaiRetrieval()
        agents = BonsaiAgents(retrieval)
        logger.info("lazy_init_complete", msg="Retrieval and agents ready.")
    return retrieval, agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", msg="B.O.N.S.A.I. API starting (lazy init mode)...")
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

# Middleware (outermost runs first: CORS → Payment → Logging)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestLoggingMiddleware)
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    """Serve developer docs at root URL."""
    docs_path = Path(__file__).parent / "bonsai_developer_docs.html"
    return HTMLResponse(content=docs_path.read_text())


@app.get("/v1/health")
async def health_check():
    """Health check — no payment required. Verifies dependency config."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    uptime_seconds = int((now - STARTED_AT).total_seconds())

    checks = {
        "pinecone_key": bool(settings.pinecone_api_key),
        "anthropic_key": bool(settings.anthropic_api_key),
        "openai_key": bool(settings.openai_api_key),
        "wallet_configured": bool(settings.bonsai_wallet_address),
    }
    all_ok = all(checks.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "service": "B.O.N.S.A.I. Health Intelligence API",
        "version": "3.0.0",
        "started_at": STARTED_AT.isoformat(),
        "uptime_seconds": uptime_seconds,
        "checks": checks,
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
    _retrieval, _agents = _get_retrieval_and_agents()

    logger.info("tier1_query", query=q, pillar=pillar, limit=limit)

    try:
        results = await _retrieval.query(question=q, top_k=limit, pillar=pillar)
    except Exception as e:
        logger.error("tier1_query_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {type(e).__name__}")

    response = _retrieval.format_tier1_response(results, q, include_citations)

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

    _retrieval, _agents = _get_retrieval_and_agents()
    logger.info("tier2_protocol", condition=body.condition, duration=body.duration_weeks)

    result = await _agents.generate_protocol(
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

    _retrieval, _agents = _get_retrieval_and_agents()
    logger.info("tier3_lab_protocol", num_markers=len(body.lab_values))

    result = await _agents.lab_informed_protocol(
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

    _retrieval, _agents = _get_retrieval_and_agents()
    logger.info("tier4_assessment", has_labs=bool(body.lab_values))

    result = await _agents.lifestyle_assessment(
        current_habits=body.current_habits,
        health_goals=body.health_goals,
        lab_values=body.lab_values,
    )

    latency = (time.time() - start) * 1000
    track_request("tier_4", latency, 0.85)
    logger.info("tier4_complete", latency_ms=round(latency, 1))

    return result


# ── Terms of Service & Legal ──────────────────────────────────────────────────


@app.get("/v1/terms")
async def terms_of_service():
    """Terms of Service and legal disclaimers."""
    return {
        "service": "B.O.N.S.A.I. Health Intelligence API",
        "version": "3.0.0",
        "effective_date": "2026-03-01",
        "terms_of_service": {
            "acceptance": (
                "By accessing any B.O.N.S.A.I. API endpoint or submitting payment, "
                "you agree to these terms. If you do not agree, do not use this API."
            ),
            "service_description": (
                "B.O.N.S.A.I. provides evidence-based whole-food plant-based (WFPB) "
                "lifestyle medicine intelligence via 4 paid API tiers. Content is derived "
                "from published clinical guidelines (ACLM, AHA, ADA, WHO) and 637+ "
                "peer-reviewed references."
            ),
            "not_medical_advice": (
                "ALL CONTENT RETURNED BY THIS API IS EDUCATIONAL AND INFORMATIONAL ONLY. "
                "It does NOT constitute medical advice, diagnosis, or treatment. It is NOT "
                "a substitute for professional medical advice, diagnosis, or treatment from "
                "a qualified healthcare provider. Always seek the advice of your physician "
                "or other qualified health provider with any questions you may have regarding "
                "a medical condition. Never disregard professional medical advice or delay in "
                "seeking it because of information returned by this API."
            ),
            "no_fda_evaluation": (
                "Statements and recommendations returned by this API have not been evaluated "
                "by the U.S. Food and Drug Administration (FDA). This API is not intended to "
                "diagnose, treat, cure, or prevent any disease."
            ),
            "limitation_of_liability": (
                "B.O.N.S.A.I. and its operators shall not be held liable for any damages, "
                "injuries, or adverse health outcomes arising from the use of information "
                "provided by this API. Users assume all risk associated with acting on "
                "API-generated content. In no event shall liability exceed the amount paid "
                "for the specific API call that gave rise to the claim."
            ),
            "payment_terms": (
                "Payments are processed via the x402 protocol using USDC on the Base network. "
                "All payments are final and non-refundable. Pricing is per-request as listed "
                "in the API documentation and MCP manifest."
            ),
            "data_handling": (
                "B.O.N.S.A.I. does not store, log, or retain any personally identifiable "
                "health information (PHI) submitted via API requests. Query content is processed "
                "in memory and discarded after the response is returned. No user accounts are "
                "required. We do not sell or share any data."
            ),
            "ai_disclosure": (
                "Tiers 2-4 use AI language models (Anthropic Claude) to generate responses. "
                "AI-generated content may contain errors, omissions, or inaccuracies. All "
                "outputs should be reviewed by qualified professionals before clinical application."
            ),
            "modifications": (
                "We reserve the right to modify these terms, pricing, or API functionality "
                "at any time. Continued use after changes constitutes acceptance."
            ),
        },
        "pricing": {
            "tier_1_query": "$0.02 USDC",
            "tier_2_protocol": "$0.20 USDC",
            "tier_3_lab_protocol": "$0.45 USDC",
            "tier_4_assessment": "$0.85 USDC",
            "payment_network": "Base (Ethereum L2)",
            "payment_token": "USDC",
            "payment_protocol": "x402",
        },
        "contact": "rabya.valla@gmail.com",
    }


# ── Monitoring ────────────────────────────────────────────────────────────────


@app.get("/v1/metrics")
async def get_metrics(request: Request):
    """Internal metrics endpoint — requires admin API key."""
    settings = get_settings()
    admin_key = request.headers.get(settings.api_key_header)
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API key not configured")
    if not admin_key or admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Unauthorized — admin API key required")

    avg_latency = {}
    for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
        count = metrics["latency_count"][tier]
        avg_latency[tier] = round(metrics["latency_sum_ms"][tier] / count, 1) if count > 0 else 0

    now = datetime.now(timezone.utc)
    uptime_seconds = int((now - STARTED_AT).total_seconds())
    total = metrics["requests_total"]
    error_rate = round(metrics["errors_total"] / total, 4) if total > 0 else 0

    return {
        "started_at": STARTED_AT.isoformat(),
        "uptime_seconds": uptime_seconds,
        "requests_total": total,
        "errors_total": metrics["errors_total"],
        "error_rate": error_rate,
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
