"""Transactional State and User Persistence Repository."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.auth import hash_password
from app.core.exceptions import StatePersistenceError
from app.core.logging import get_logger
from app.schemas.agent_state import AgentState
from app.schemas.tool_call import ToolResult
from app.storage.database import SessionLocal, init_db
from app.storage.models import MessageRecord, RequestLogRecord, ReviewRecord, SessionRecord, ToolExecutionRecord, UserRecord

logger = get_logger("ai_system.repository")


class StateRepository:
    """
    Transactional repository ensuring atomic updates of users, conversation history,
    agent states, and execution telemetry to SQLite.
    """

    def __init__(self):
        init_db()

    def _get_db(self) -> Session:
        return SessionLocal()

    # =========================================================================
    # User Management Operations
    # =========================================================================

    async def create_user(
        self,
        email: str,
        name: str,
        password: str,
        is_demo: bool = False
    ) -> UserRecord:
        """Creates a new user record with salted PBKDF2 password hash."""
        def _sync_op():
            with self._get_db() as db:
                existing = db.query(UserRecord).filter(UserRecord.email == email.strip().lower()).first()
                if existing:
                    raise ValueError(f"User with email '{email}' already exists.")

                pwd_hash, salt = hash_password(password)
                user = UserRecord(
                    id=f"usr_{uuid.uuid4().hex[:12]}",
                    email=email.strip().lower(),
                    name=name.strip(),
                    password_hash=pwd_hash,
                    salt=salt,
                    is_demo=is_demo,
                    preferences_json=json.dumps({
                        "ranking_mode": "best_value",
                        "prefer_free": True,
                        "prefer_local": True
                    })
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                return user

        return await asyncio.to_thread(_sync_op)

    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Looks up user by normalized email."""
        def _sync_op():
            with self._get_db() as db:
                return db.query(UserRecord).filter(UserRecord.email == email.strip().lower()).first()

        return await asyncio.to_thread(_sync_op)

    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Looks up user by primary key ID."""
        def _sync_op():
            with self._get_db() as db:
                return db.query(UserRecord).filter(UserRecord.id == user_id).first()

        return await asyncio.to_thread(_sync_op)

    async def update_user_password(self, user_id: str, new_password: str) -> bool:
        """Updates user password with fresh salt and hash."""
        def _sync_op():
            with self._get_db() as db:
                user = db.query(UserRecord).filter(UserRecord.id == user_id).first()
                if not user:
                    return False
                pwd_hash, salt = hash_password(new_password)
                user.password_hash = pwd_hash
                user.salt = salt
                user.updated_at = datetime.now(timezone.utc)
                db.commit()
                return True

        return await asyncio.to_thread(_sync_op)

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Persists updated user preferences dictionary."""
        def _sync_op():
            with self._get_db() as db:
                user = db.query(UserRecord).filter(UserRecord.id == user_id).first()
                if not user:
                    return False
                user.preferences_json = json.dumps(preferences)
                user.updated_at = datetime.now(timezone.utc)
                db.commit()
                return True

        return await asyncio.to_thread(_sync_op)

    async def get_or_create_demo_user(self) -> UserRecord:
        """Retrieves or provisions the default demo sandbox guest account."""
        demo_email = "demo@openpilot.ai"
        user = await self.get_user_by_email(demo_email)
        if not user:
            user = await self.create_user(
                email=demo_email,
                name="Demo Pilot",
                password="DemoUser2026!",
                is_demo=True
            )
        return user

    # =========================================================================
    # Request Logs & Audit Operations
    # =========================================================================

    async def log_request(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        category: str,
        selected_model: str,
        ranking_mode: str,
        latency_ms: float,
        tokens_used: int,
        cost_usd: float,
        status: str = "COMPLETE"
    ) -> None:
        """Persists an execution audit record into request_logs table."""
        def _sync_op():
            with self._get_db() as db:
                log_entry = RequestLogRecord(
                    user_id=user_id,
                    session_id=session_id,
                    prompt=prompt,
                    category=category,
                    selected_model=selected_model,
                    ranking_mode=ranking_mode,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    status=status
                )
                db.add(log_entry)
                db.commit()

        await asyncio.to_thread(_sync_op)

    async def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves past requests and executions for the given user."""
        def _sync_op():
            with self._get_db() as db:
                logs = (
                    db.query(RequestLogRecord)
                    .filter(RequestLogRecord.user_id == user_id)
                    .order_by(RequestLogRecord.id.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": l.id,
                        "session_id": l.session_id,
                        "prompt": l.prompt,
                        "category": l.category,
                        "selected_model": l.selected_model,
                        "ranking_mode": l.ranking_mode,
                        "latency_ms": l.latency_ms,
                        "tokens_used": l.tokens_used,
                        "cost_usd": l.cost_usd,
                        "status": l.status,
                        "created_at": l.created_at.isoformat() if l.created_at else None
                    }
                    for l in logs
                ]

        return await asyncio.to_thread(_sync_op)

    # =========================================================================
    # Conversational Session & Message Operations
    # =========================================================================

    async def get_or_create_session(self, session_id: str, user_id: str) -> SessionRecord:
        """Retrieves an existing session record or transactionally creates a new one."""
        def _sync_op():
            with self._get_db() as db:
                session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
                if not session:
                    session = SessionRecord(
                        session_id=session_id,
                        user_id=user_id,
                        final_status="IN_PROGRESS",
                        iteration_count=0
                    )
                    db.add(session)
                    db.commit()
                    db.refresh(session)
                return session

        return await asyncio.to_thread(_sync_op)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        name: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Transactionally commits an individual message entry before/after external calls."""
        def _sync_op():
            with self._get_db() as db:
                try:
                    tool_calls_json = json.dumps(tool_calls) if tool_calls else None
                    msg = MessageRecord(
                        session_id=session_id,
                        role=role,
                        content=content,
                        name=name,
                        tool_calls_json=tool_calls_json
                    )
                    db.add(msg)
                    session = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
                    if session:
                        session.updated_at = datetime.now(timezone.utc)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(f"Failed to persist message for session {session_id}: {exc}")
                    raise StatePersistenceError(f"Database error persisting message: {exc}") from exc

        await asyncio.to_thread(_sync_op)

    # Alias for convenience
    save_message = append_message

    async def record_tool_execution(
        self,
        session_id: str,
        tool_result: ToolResult,
        arguments: Dict[str, Any]
    ) -> None:
        """Persists audit log of a single tool execution."""
        def _sync_op():
            with self._get_db() as db:
                try:
                    rec = ToolExecutionRecord(
                        session_id=session_id,
                        tool_name=tool_result.tool_name,
                        call_id=tool_result.call_id,
                        arguments_json=json.dumps(arguments),
                        output_json=json.dumps(tool_result.output) if tool_result.output is not None else None,
                        is_error=tool_result.is_error,
                        error_message=tool_result.error_message,
                        execution_time_ms=tool_result.execution_time_ms
                    )
                    db.add(rec)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(f"Failed to persist tool execution for session {session_id}: {exc}")
                    raise StatePersistenceError(f"Database error saving tool execution: {exc}") from exc

        await asyncio.to_thread(_sync_op)

    async def update_session_state(self, state: AgentState) -> None:
        """Persists status and iteration count of active AgentState."""
        def _sync_op():
            with self._get_db() as db:
                try:
                    session = db.query(SessionRecord).filter(SessionRecord.session_id == state.session_id).first()
                    if session:
                        session.final_status = state.final_status
                        session.iteration_count = state.iteration_count
                        if state.last_tool_call:
                            session.last_tool_call_json = json.dumps(state.last_tool_call)
                        session.updated_at = datetime.now(timezone.utc)
                        db.commit()
                except Exception as exc:
                    db.rollback()
                    raise StatePersistenceError(f"Database error updating session state: {exc}") from exc

        await asyncio.to_thread(_sync_op)

    async def get_session_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves ordered conversation history as dicts for model input."""
        def _sync_op():
            with self._get_db() as db:
                messages = (
                    db.query(MessageRecord)
                    .filter(MessageRecord.session_id == session_id)
                    .order_by(MessageRecord.id.asc())
                    .all()
                )
                return [m.to_dict() for m in messages]

        return await asyncio.to_thread(_sync_op)

    async def get_tool_executions(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves list of all tool executions for a given session."""
        def _sync_op():
            with self._get_db() as db:
                records = (
                    db.query(ToolExecutionRecord)
                    .filter(ToolExecutionRecord.session_id == session_id)
                    .order_by(ToolExecutionRecord.id.asc())
                    .all()
                )
                results = []
                for r in records:
                    results.append({
                        "id": r.id,
                        "tool_name": r.tool_name,
                        "call_id": r.call_id,
                        "is_error": r.is_error,
                        "error_message": r.error_message,
                        "execution_time_ms": r.execution_time_ms,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    })
                return results

        return await asyncio.to_thread(_sync_op)

    # =========================================================================
    # Review & Rating Operations (Review-to-Routing Feedback Loop)
    # =========================================================================

    async def save_or_update_review(
        self,
        user_id: str,
        session_id: str,
        model_id: str,
        provider: str,
        task_category: str,
        rating: int,
        is_correct: Optional[bool] = None,
        is_useful: Optional[bool] = None,
        is_complete: Optional[bool] = None,
        is_fast: Optional[bool] = None,
        cost_acceptable: Optional[bool] = None,
        verification_status: Optional[str] = None,
        feedback_text: Optional[str] = None,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> Dict[str, Any]:
        """Saves a new review or updates an existing one for the same user and session."""
        def _sync_op():
            with self._get_db() as db:
                existing = (
                    db.query(ReviewRecord)
                    .filter(ReviewRecord.user_id == user_id, ReviewRecord.session_id == session_id)
                    .first()
                )
                if existing:
                    existing.rating = max(1, min(5, rating))
                    existing.is_correct = is_correct
                    existing.is_useful = is_useful
                    existing.is_complete = is_complete
                    existing.is_fast = is_fast
                    existing.cost_acceptable = cost_acceptable
                    existing.feedback_text = feedback_text
                    if verification_status:
                        existing.verification_status = verification_status
                    existing.updated_at = datetime.now(timezone.utc)
                    db.commit()
                    db.refresh(existing)
                    rev = existing
                    action = "updated"
                else:
                    rev = ReviewRecord(
                        user_id=user_id,
                        session_id=session_id,
                        model_id=model_id,
                        provider=provider,
                        task_category=task_category,
                        rating=max(1, min(5, rating)),
                        is_correct=is_correct,
                        is_useful=is_useful,
                        is_complete=is_complete,
                        is_fast=is_fast,
                        cost_acceptable=cost_acceptable,
                        verification_status=verification_status,
                        feedback_text=feedback_text,
                        latency_ms=latency_ms,
                        cost_usd=cost_usd
                    )
                    db.add(rev)
                    db.commit()
                    db.refresh(rev)
                    action = "created"

                return {
                    "id": rev.id,
                    "action": action,
                    "session_id": rev.session_id,
                    "model_id": rev.model_id,
                    "rating": rev.rating,
                    "task_category": rev.task_category,
                    "created_at": rev.created_at.isoformat() if rev.created_at else None
                }

        return await asyncio.to_thread(_sync_op)

    async def get_model_task_rating_stats(
        self,
        model_id: str,
        task_category: Optional[str] = None,
        m_prior: float = 5.0,
        prior_avg: float = 4.0
    ) -> Dict[str, Any]:
        """
        Computes Bayesian smoothed rating:
        Smoothed Rating = (n * observed_avg + m * prior_avg) / (n + m)
        """
        def _sync_op():
            with self._get_db() as db:
                q = db.query(ReviewRecord).filter(ReviewRecord.model_id == model_id)
                if task_category:
                    q = q.filter(ReviewRecord.task_category == task_category)
                reviews = q.all()

                n = len(reviews)
                if n == 0:
                    return {
                        "model_id": model_id,
                        "task_category": task_category,
                        "review_count": 0,
                        "observed_avg": None,
                        "smoothed_rating": prior_avg,
                        "aspects": {
                            "correct_pct": None,
                            "useful_pct": None,
                            "complete_pct": None,
                            "fast_pct": None,
                            "cost_acceptable_pct": None
                        }
                    }

                total_rating = sum(r.rating for r in reviews)
                observed_avg = total_rating / n
                smoothed = (n * observed_avg + m_prior * prior_avg) / (n + m_prior)

                def pct(vals):
                    valid = [v for v in vals if v is not None]
                    return round((sum(1 for v in valid if v) / len(valid)) * 100, 1) if valid else None

                return {
                    "model_id": model_id,
                    "task_category": task_category,
                    "review_count": n,
                    "observed_avg": round(observed_avg, 2),
                    "smoothed_rating": round(smoothed, 2),
                    "aspects": {
                        "correct_pct": pct([r.is_correct for r in reviews]),
                        "useful_pct": pct([r.is_useful for r in reviews]),
                        "complete_pct": pct([r.is_complete for r in reviews]),
                        "fast_pct": pct([r.is_fast for r in reviews]),
                        "cost_acceptable_pct": pct([r.cost_acceptable for r in reviews])
                    }
                }

        return await asyncio.to_thread(_sync_op)

    async def get_all_model_ratings(
        self,
        task_category: Optional[str] = None,
        m_prior: float = 5.0,
        prior_avg: float = 4.0
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregates rating statistics for all models recorded."""
        def _sync_op():
            with self._get_db() as db:
                q = db.query(ReviewRecord)
                if task_category:
                    q = q.filter(ReviewRecord.task_category == task_category)
                reviews = q.all()

                by_model: Dict[str, List[ReviewRecord]] = {}
                for r in reviews:
                    by_model.setdefault(r.model_id, []).append(r)

                results = {}
                for m_id, rev_list in by_model.items():
                    n = len(rev_list)
                    obs = sum(r.rating for r in rev_list) / n
                    smoothed = (n * obs + m_prior * prior_avg) / (n + m_prior)
                    results[m_id] = {
                        "review_count": n,
                        "observed_avg": round(obs, 2),
                        "smoothed_rating": round(smoothed, 2)
                    }
                return results

        return await asyncio.to_thread(_sync_op)

    async def get_reviews_for_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves reviews submitted by the user."""
        def _sync_op():
            with self._get_db() as db:
                revs = (
                    db.query(ReviewRecord)
                    .filter(ReviewRecord.user_id == user_id)
                    .order_by(ReviewRecord.id.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": r.id,
                        "session_id": r.session_id,
                        "model_id": r.model_id,
                        "provider": r.provider,
                        "task_category": r.task_category,
                        "rating": r.rating,
                        "is_correct": r.is_correct,
                        "is_useful": r.is_useful,
                        "is_complete": r.is_complete,
                        "is_fast": r.is_fast,
                        "cost_acceptable": r.cost_acceptable,
                        "verification_status": r.verification_status,
                        "feedback_text": r.feedback_text,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    }
                    for r in revs
                ]

        return await asyncio.to_thread(_sync_op)

    async def delete_review(self, review_id: int, user_id: str) -> bool:
        """Deletes a review if owned by user."""
        def _sync_op():
            with self._get_db() as db:
                r = db.query(ReviewRecord).filter(ReviewRecord.id == review_id, ReviewRecord.user_id == user_id).first()
                if not r:
                    return False
                db.delete(r)
                db.commit()
                return True

        return await asyncio.to_thread(_sync_op)

    # =========================================================================
    # Dashboard & Control Center Aggregates
    # =========================================================================

    async def get_dashboard_aggregates(self) -> Dict[str, Any]:
        """Calculates genuine system-wide statistics from stored database records."""
        def _sync_op():
            with self._get_db() as db:
                logs = db.query(RequestLogRecord).all()
                reviews = db.query(ReviewRecord).all()

                total_requests = len(logs)
                successful = sum(1 for l in logs if l.status == "COMPLETE")
                failed = total_requests - successful

                model_usage: Dict[str, int] = {}
                category_usage: Dict[str, int] = {}
                free_count = 0
                paid_count = 0
                total_tokens = 0
                total_cost = 0.0
                latencies = []

                for l in logs:
                    model_usage[l.selected_model] = model_usage.get(l.selected_model, 0) + 1
                    category_usage[l.category] = category_usage.get(l.category, 0) + 1
                    if l.cost_usd > 0.0:
                        paid_count += 1
                    else:
                        free_count += 1
                    total_tokens += (l.tokens_used or 0)
                    total_cost += (l.cost_usd or 0.0)
                    if l.latency_ms > 0:
                        latencies.append(l.latency_ms)

                latencies.sort()
                median_lat = latencies[len(latencies) // 2] if latencies else 0.0
                p95_lat = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
                avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

                avg_rating = (sum(r.rating for r in reviews) / len(reviews)) if reviews else 0.0
                estimated_tokens_saved = int(total_tokens * 0.28)

                return {
                    "total_requests": total_requests,
                    "successful_executions": successful,
                    "failed_executions": failed,
                    "success_rate_pct": round((successful / total_requests * 100), 1) if total_requests else 100.0,
                    "model_usage_distribution": model_usage,
                    "category_distribution": category_usage,
                    "free_vs_paid": {
                        "free": free_count,
                        "paid": paid_count
                    },
                    "total_tokens_consumed": total_tokens,
                    "estimated_tokens_saved": estimated_tokens_saved,
                    "total_cost_usd": round(total_cost, 6),
                    "latency": {
                        "avg_ms": round(avg_lat, 2),
                        "median_ms": round(median_lat, 2),
                        "p95_ms": round(p95_lat, 2)
                    },
                    "reviews_count": len(reviews),
                    "average_rating": round(avg_rating, 2)
                }

        return await asyncio.to_thread(_sync_op)


state_repository = StateRepository()
