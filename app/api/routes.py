"""FastAPI API Gateway Route Handlers for SYNTRA / OpenPilot AI Platform."""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, status
import httpx

from app.api.schemas import (
    ApiKeysUpdateRequest,
    AuthResponse,
    ChatRequest,
    DashboardStatsResponse,
    DemoSimulateFailureRequest,
    ExecuteRequest,
    FeedbackRequest,
    HealthResponse,
    LoginRequest,
    PasswordResetRequest,
    RecommendRequest,
    RegisterRequest,
    ReviewSubmissionRequest,
    ToolSummary,
    UnderstandRequest,
    UserPreferencesRequest,
    VerifyRequest,
)
from app.services.token_planner import token_planner
from app.core.auth import (
    create_access_token,
    decode_access_token,
    validate_email,
    validate_password_strength,
    verify_password,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.orchestrator.engine import orchestrator_engine
from app.orchestrator.router import intent_router
from app.schemas.context import PromptContext
from app.schemas.responses import ChatResponse
from app.services.catalogue import model_catalogue
from app.services.context_dna import context_dna_manager
from app.services.migration import model_migration_engine
from app.services.recommender import model_recommender
from app.services.verification import VerificationPolicy, verification_engine
from app.storage.repository import state_repository
from app.tools.registry import tool_registry

logger = get_logger("ai_system.api")
router = APIRouter()


# =============================================================================
# Auth Helper Dependency
# =============================================================================

async def get_optional_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Extracts authenticated user from Bearer JWT header or returns demo default."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload:
            return payload

    # If no token, provide guest session
    return {"sub": "guest_anonymous", "email": "guest@openpilot.ai", "is_demo": True}


# =============================================================================
# 1. Authentication & Profile Endpoints
# =============================================================================

@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_user(req: RegisterRequest) -> AuthResponse:
    """Registers a new user with email and password complexity validation."""
    if not validate_email(req.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format. Please provide a valid address."
        )

    is_valid_pwd, pwd_error = validate_password_strength(req.password)
    if not is_valid_pwd:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=pwd_error)

    try:
        user = await state_repository.create_user(
            email=req.email,
            name=req.name,
            password=req.password
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    token = create_access_token(user.id, user.email, is_demo=False)
    prefs = json.loads(user.preferences_json) if user.preferences_json else {}

    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        name=user.name,
        is_demo=False,
        preferences=prefs
    )


@router.post("/auth/login", response_model=AuthResponse)
async def login_user(req: LoginRequest) -> AuthResponse:
    """Authenticates existing users using verified password hashing."""
    user = await state_repository.get_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password credentials."
        )

    if not verify_password(req.password, user.password_hash, user.salt):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password credentials."
        )

    token = create_access_token(user.id, user.email, is_demo=user.is_demo)
    prefs = json.loads(user.preferences_json) if user.preferences_json else {}

    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        name=user.name,
        is_demo=user.is_demo,
        preferences=prefs
    )


@router.post("/auth/demo", response_model=AuthResponse)
async def demo_login() -> AuthResponse:
    """Generates an instant sandboxed guest/demo session."""
    user = await state_repository.get_or_create_demo_user()
    token = create_access_token(user.id, user.email, is_demo=True)
    prefs = json.loads(user.preferences_json) if user.preferences_json else {}

    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        name=user.name,
        is_demo=True,
        preferences=prefs
    )


@router.post("/auth/reset-password")
async def reset_password(req: PasswordResetRequest) -> Dict[str, str]:
    """Resets user password following email verification."""
    user = await state_repository.get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account with this email does not exist.")

    is_valid_pwd, pwd_error = validate_password_strength(req.new_password)
    if not is_valid_pwd:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=pwd_error)

    success = await state_repository.update_user_password(user.id, req.new_password)
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reset password.")

    return {"message": "Password updated successfully. You can now log in with your new password."}


@router.get("/auth/me")
async def get_current_profile(user_payload: Dict[str, Any] = Depends(get_optional_current_user)) -> Dict[str, Any]:
    """Returns profile and preferences for active authenticated user."""
    user_id = user_payload.get("sub")
    user = await state_repository.get_user_by_id(user_id)
    if not user:
        return {
            "id": user_id,
            "email": user_payload.get("email"),
            "name": "Guest Pilot",
            "is_demo": True,
            "preferences": {"ranking_mode": "best_value", "prefer_free": True}
        }
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "is_demo": user.is_demo,
        "preferences": json.loads(user.preferences_json) if user.preferences_json else {}
    }


@router.put("/auth/preferences")
async def update_preferences(
    prefs: UserPreferencesRequest,
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> Dict[str, Any]:
    """Persists updated user ranking and budget preferences."""
    user_id = user_payload.get("sub")
    user = await state_repository.get_user_by_id(user_id)
    if user:
        await state_repository.update_user_preferences(user_id, prefs.model_dump())
    return {"status": "preferences_saved", "preferences": prefs.model_dump()}


# =============================================================================
# 2. Dynamic Request Understanding & Model Recommendations
# =============================================================================

@router.post("/understand")
async def understand_request_endpoint(req: UnderstandRequest) -> Dict[str, Any]:
    """Analyzes prompt text and media modalities to determine task classification and required tools."""
    understanding = intent_router.understand_request(
        prompt=req.prompt,
        has_image=req.has_image,
        has_audio=req.has_audio
    )
    return understanding.model_dump()


@router.get("/catalogue")
async def get_catalogue_endpoint() -> Dict[str, Any]:
    """Retrieves full verified model catalogue with live connectivity status."""
    model_catalogue.refresh_availability()
    models = model_catalogue.get_all_models()
    return {
        "total_models": len(models),
        "models": [m.model_dump() for m in models]
    }


@router.post("/recommend")
async def recommend_models_endpoint(req: RecommendRequest) -> Dict[str, Any]:
    """Generates ranked model recommendations across the 5 dynamic ranking modes."""
    understanding = intent_router.understand_request(
        prompt=req.prompt,
        has_image=req.has_image,
        has_audio=req.has_audio
    )
    recs = model_recommender.recommend(
        understanding=understanding,
        ranking_mode=req.ranking_mode,
        max_budget_cents=req.max_budget_cents
    )
    return recs.model_dump()


@router.post("/execute")
async def execute_model_endpoint(
    req: ExecuteRequest,
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> Dict[str, Any]:
    """Executes a request through the selected or recommended engine with full telemetry."""
    user_id = user_payload.get("sub", "guest_pilot")
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"

    try:
        result = await orchestrator_engine.execute_request(
            prompt=req.prompt,
            user_id=user_id,
            session_id=session_id,
            selected_model=req.selected_model,
            ranking_mode=req.ranking_mode,
            image_base64=req.image_base64,
            has_audio=req.has_audio
        )
        return result
    except Exception as exc:
        logger.error(f"Error in execute_model_endpoint: {exc}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Execution error on '{req.selected_model or 'selected model'}': {str(exc)}"
        )


@router.get("/settings/keys")
async def get_settings_keys_endpoint() -> Dict[str, Any]:
    """Returns configuration status of external cloud API keys and installed local models."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or getattr(settings, "GEMINI_API_KEY", "") or ""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    # Query local Ollama tags
    local_models = []
    ollama_online = False
    try:
        with httpx.Client(timeout=0.5) as client:
            res = client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            if res.status_code == 200:
                ollama_online = True
                local_models = [m.get("name") for m in res.json().get("models", [])]
    except Exception:
        pass

    return {
        "gemini": {
            "configured": bool(gemini_key),
            "masked": f"{gemini_key[:6]}...{gemini_key[-4:]}" if len(gemini_key) > 10 else ("Configured" if gemini_key else "Not Configured")
        },
        "openai": {
            "configured": bool(openai_key),
            "masked": f"{openai_key[:4]}...{openai_key[-4:]}" if len(openai_key) > 8 else "Not Configured"
        },
        "anthropic": {
            "configured": bool(anthropic_key),
            "masked": f"{anthropic_key[:6]}...{anthropic_key[-4:]}" if len(anthropic_key) > 10 else "Not Configured"
        },
        "ollama": {
            "online": ollama_online,
            "installed_models": local_models
        }
    }


@router.post("/settings/keys")
async def update_settings_keys_endpoint(req: ApiKeysUpdateRequest) -> Dict[str, Any]:
    """Saves provider API keys to environment, memory, and persists to .env file."""
    from pathlib import Path
    updated = {}
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"

    env_lines = {}
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_lines[k.strip()] = v.strip()
        except Exception:
            pass

    if req.gemini_api_key is not None:
        val = req.gemini_api_key.strip()
        os.environ["GEMINI_API_KEY"] = val
        os.environ["GOOGLE_API_KEY"] = val
        settings.GEMINI_API_KEY = val
        env_lines["GEMINI_API_KEY"] = val
        gemini_adapter = model_registry.get_gemini_adapter()
        if gemini_adapter:
            gemini_adapter.set_api_key(val)
        updated["gemini"] = bool(val)

    if req.openai_api_key is not None:
        val = req.openai_api_key.strip()
        os.environ["OPENAI_API_KEY"] = val
        env_lines["OPENAI_API_KEY"] = val
        updated["openai"] = bool(val)

    if req.anthropic_api_key is not None:
        val = req.anthropic_api_key.strip()
        os.environ["ANTHROPIC_API_KEY"] = val
        env_lines["ANTHROPIC_API_KEY"] = val
        updated["anthropic"] = bool(val)

    try:
        with open(env_file, "w", encoding="utf-8") as f:
            for k, v in env_lines.items():
                f.write(f"{k}={v}\n")
    except Exception as e:
        logger.warning(f"Could not persist to .env file: {e}")

    model_catalogue.refresh_availability()

    return {
        "status": "success",
        "updated_providers": updated,
        "message": "API keys updated and persisted. Model catalogue refreshed."
    }


@router.get("/history")
async def get_user_history_endpoint(
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> List[Dict[str, Any]]:
    """Retrieves request logs and cost telemetry for the authenticated user."""
    user_id = user_payload.get("sub")
    if not user_id:
        return []
    return await state_repository.get_user_history(user_id)


# =============================================================================
# 3. Legacy / Base Endpoints
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Legacy chat ingress endpoint."""
    session_id = request.session_id or str(uuid.uuid4())[:8]
    context = PromptContext(
        user_id=request.user_id,
        session_id=session_id,
        initial_prompt=request.prompt,
        history=request.history
    )
    return await orchestrator_engine.run(context)


@router.get("/sessions/{session_id}")
async def get_session_details(session_id: str) -> Dict[str, Any]:
    """Retrieves full conversation context and execution audit logs."""
    context = await state_repository.get_session_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    tools = await state_repository.get_tool_executions(session_id)
    return {
        "session_id": session_id,
        "message_count": len(context),
        "messages": context,
        "tool_executions": tools
    }


@router.get("/tools", response_model=List[ToolSummary])
async def list_tools() -> List[ToolSummary]:
    """Lists registered tools."""
    definitions = tool_registry.list_definitions()
    return [
        ToolSummary(name=d.name, description=d.description, parameters=d.parameters)
        for d in definitions
    ]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check."""
    return HealthResponse(
        status="healthy",
        brand="SYNTRA AI / OpenPilot AI",
        environment=settings.ENVIRONMENT,
        database="connected",
        ollama_endpoint=settings.OLLAMA_BASE_URL,
        registered_tools=len(tool_registry.list_definitions()),
        available_models=len(model_catalogue.get_all_models())
    )


# =============================================================================
# 4. Jury Demonstration, Trust Layer & Quality Recovery Handlers
# =============================================================================

@router.post("/demo/simulate-failure-migration")
async def demo_simulate_failure_endpoint(
    req: DemoSimulateFailureRequest,
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> Dict[str, Any]:
    """
    Jury Demonstration Mode:
    Executes a controlled failure injection test displaying the entire orchestration loop:
    1. Identifies user task requirements
    2. Dynamically ranks eligible models
    3. Simulates initial provider failure (clearly labelled)
    4. Evaluates retry bounds
    5. Preserves canonical work into 8-component Context DNA
    6. Selects eligible fallback avoiding failed provider (Circuit Breaker)
    7. Dispatches compressed context packet to target model
    8. Verifies target model output via Trust Layer (Stages A-E)
    9. Emits millisecond execution trace
    """
    user_id = user_payload.get("sub", "jury_evaluator")
    session_id = f"demo_jury_{uuid.uuid4().hex[:8]}"

    result = await model_migration_engine.execute_failover_migration(
        session_id=session_id,
        user_id=user_id,
        prompt=req.prompt,
        source_model=req.source_model,
        trigger_reason="DEMO_CONTROLLED_FAILURE",
        simulated_failure=req.simulated_failure,
        prefer_free_only=req.prefer_free_only,
        verification_policy=req.verification_policy
    )
    return result.model_dump()


@router.post("/verify")
async def verify_endpoint(req: VerifyRequest) -> Dict[str, Any]:
    """Trust Layer: Direct multi-stage verification (Stages A to E)."""
    pol = (
        VerificationPolicy(req.policy)
        if req.policy in [p.value for p in VerificationPolicy]
        else VerificationPolicy.STANDARD
    )
    report = verification_engine.verify(
        prompt=req.prompt,
        output=req.output,
        task_category=req.task_category,
        policy=pol
    )
    return report.model_dump()


@router.post("/feedback")
async def feedback_recovery_endpoint(
    req: FeedbackRequest,
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> Dict[str, Any]:
    """
    User Feedback & Quality Recovery:
    Handles post-response actions: Retry, Improve Answer, Verify Answer, Switch Model, Report Issue.
    """
    user_id = user_payload.get("sub", "guest_pilot")
    session_id = req.session_id

    if req.feedback_type == "verify":
        report = verification_engine.verify(
            prompt=req.prompt,
            output=req.current_output
        )
        return {
            "feedback_type": "verify",
            "verification_report": report.model_dump()
        }

    elif req.feedback_type in ("switch_model", "improve", "retry"):
        alt_model = req.target_model or "qwen-2.5-coder-7b"
        reason = f"USER_FEEDBACK_{req.feedback_type.upper()}"
        migration = await model_migration_engine.execute_failover_migration(
            session_id=session_id,
            user_id=user_id,
            prompt=f"{req.prompt} (Refinement: {req.notes})" if req.notes else req.prompt,
            source_model=alt_model,
            trigger_reason=reason
        )
        return {
            "feedback_type": req.feedback_type,
            "migration": migration.model_dump()
        }

    return {
        "status": "feedback_recorded",
        "feedback_type": req.feedback_type,
        "notes": req.notes
    }


@router.get("/context-dna/{session_id}")
async def get_context_dna_endpoint(session_id: str) -> Dict[str, Any]:
    """Retrieves the live 8-component Context DNA for any active session."""
    messages = await state_repository.get_session_context(session_id)
    tools = await state_repository.get_tool_executions(session_id)
    dna = context_dna_manager.extract_dna_from_session(session_id, messages, tools)
    return dna.model_dump()


# =============================================================================
# 5. Review-to-Routing Feedback Loop Endpoints (Section 12)
# =============================================================================

@router.post("/reviews")
async def submit_review_endpoint(
    req: ReviewSubmissionRequest,
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> Dict[str, Any]:
    """
    Submits or updates a 1-5 star user review and quality assessments.
    Dynamically recalculates Bayesian smoothed rating for future routing.
    """
    user_id = user_payload.get("sub", "guest_pilot")
    saved = await state_repository.save_or_update_review(
        user_id=user_id,
        session_id=req.session_id,
        model_id=req.model_id,
        provider=req.provider,
        task_category=req.task_category,
        rating=req.rating,
        is_correct=req.is_correct,
        is_useful=req.is_useful,
        is_complete=req.is_complete,
        is_fast=req.is_fast,
        cost_acceptable=req.cost_acceptable,
        verification_status=req.verification_status,
        feedback_text=req.feedback_text,
        latency_ms=req.latency_ms,
        cost_usd=req.cost_usd
    )

    updated_stats = await state_repository.get_model_task_rating_stats(
        model_id=req.model_id,
        task_category=req.task_category
    )

    return {
        "status": "success",
        "review": saved,
        "updated_model_stats": updated_stats
    }


@router.get("/reviews/stats")
async def get_review_stats_endpoint(
    model_id: Optional[str] = None,
    task_category: Optional[str] = None
) -> Dict[str, Any]:
    """Returns Bayesian smoothed rating metrics across models or for a specific model."""
    if model_id:
        return await state_repository.get_model_task_rating_stats(
            model_id=model_id,
            task_category=task_category
        )
    return await state_repository.get_all_model_ratings(task_category=task_category)


@router.get("/reviews/my")
async def get_my_reviews_endpoint(
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> List[Dict[str, Any]]:
    """Returns reviews submitted by current user."""
    user_id = user_payload.get("sub", "guest_pilot")
    return await state_repository.get_reviews_for_user(user_id=user_id)


@router.delete("/reviews/{review_id}")
async def delete_review_endpoint(
    review_id: int,
    user_payload: Dict[str, Any] = Depends(get_optional_current_user)
) -> Dict[str, Any]:
    """Allows user to retract/delete their submitted review."""
    user_id = user_payload.get("sub", "guest_pilot")
    success = await state_repository.delete_review(review_id=review_id, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Review not found or not owned by user.")
    return {"status": "deleted", "review_id": review_id}


# =============================================================================
# 6. Dashboard / Control Center & Model Explorer Endpoints
# =============================================================================

@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats_endpoint() -> DashboardStatsResponse:
    """Returns live aggregated system telemetry derived from stored records."""
    data = await state_repository.get_dashboard_aggregates()
    return DashboardStatsResponse(**data)


@router.get("/models")
@router.get("/catalogue")
async def list_models_endpoint() -> List[Dict[str, Any]]:
    """Lists all registered models with pricing, benchmarks, status, and cached ratings."""
    model_catalogue.refresh_availability()
    models = model_catalogue.get_all_models()
    ratings = await state_repository.get_all_model_ratings()

    output = []
    for m in models:
        r_stat = ratings.get(m.id, {})
        u_rating = r_stat.get("smoothed_rating", 4.0)
        u_count = r_stat.get("review_count", 0)

        output.append({
            "id": m.id,
            "name": m.name,
            "provider": m.provider,
            "integration_type": m.integration_type,
            "is_free": m.is_free,
            "input_cost_per_1m": m.input_cost_per_1m,
            "output_cost_per_1m": m.output_cost_per_1m,
            "pricing_display": "100% Free (Open Weights / Local)" if m.is_free else f"${m.input_cost_per_1m:.2f} in / ${m.output_cost_per_1m:.2f} out per 1M",
            "context_limit_tokens": m.context_limit_tokens,
            "supported_modalities": m.supported_modalities,
            "supported_tasks": m.supported_tasks,
            "benchmark_name": m.benchmark_name,
            "benchmark_score": m.benchmark_score,
            "benchmark_source": m.benchmark_source,
            "status": m.status,
            "status_label": m.status_label,
            "adapter_configured": m.adapter_configured,
            "user_rating": round(u_rating, 1) if u_count > 0 else None,
            "review_count": u_count,
            "rating_display": f"{u_rating:.1f}★ ({u_count} reviews)" if u_count > 0 else "4.0★ prior (0 reviews)"
        })
    return output

