"""SQLAlchemy ORM Models for Users, Sessions, Messages, and Audit Records."""

import json
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.storage.database import Base


class UserRecord(Base):
    """Registered and authenticated user record."""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    password_hash = Column(String(256), nullable=False)
    salt = Column(String(64), nullable=False)
    is_demo = Column(Boolean, default=False)
    preferences_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    request_logs = relationship("RequestLogRecord", back_populates="user", cascade="all, delete-orphan", order_by="RequestLogRecord.id.desc()")


class SessionRecord(Base):
    """Persistent representation of a conversational session and its AgentState."""
    __tablename__ = "sessions"

    session_id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    final_status = Column(String(32), default="IN_PROGRESS")
    iteration_count = Column(Integer, default=0)
    last_tool_call_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("MessageRecord", back_populates="session", cascade="all, delete-orphan", order_by="MessageRecord.id")
    tool_executions = relationship("ToolExecutionRecord", back_populates="session", cascade="all, delete-orphan", order_by="ToolExecutionRecord.id")


class MessageRecord(Base):
    """Persistent chronological message entry."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    name = Column(String(64), default="")
    tool_calls_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="messages")

    def to_dict(self):
        item = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            item["name"] = self.name
        if self.tool_calls_json:
            try:
                item["tool_calls"] = json.loads(self.tool_calls_json)
            except Exception:
                pass
        return item


class ToolExecutionRecord(Base):
    """Audit log record for every tool execution during a turn."""
    __tablename__ = "tool_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    tool_name = Column(String(64), nullable=False)
    call_id = Column(String(64), nullable=True)
    arguments_json = Column(Text, nullable=True)
    output_json = Column(Text, nullable=True)
    is_error = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionRecord", back_populates="tool_executions")


class RequestLogRecord(Base):
    """Chronological log of user requests, chosen models, recommendations, and execution costs."""
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    category = Column(String(64), nullable=False)
    selected_model = Column(String(64), nullable=False)
    ranking_mode = Column(String(32), default="best_value")
    latency_ms = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    status = Column(String(32), default="COMPLETE")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("UserRecord", back_populates="request_logs")


class ReviewRecord(Base):
    """Persisted user review and multidimensional feedback on an execution."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    model_id = Column(String(64), nullable=False, index=True)
    provider = Column(String(64), nullable=False)
    task_category = Column(String(64), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 1 to 5 stars
    is_correct = Column(Boolean, nullable=True)
    is_useful = Column(Boolean, nullable=True)
    is_complete = Column(Boolean, nullable=True)
    is_fast = Column(Boolean, nullable=True)
    cost_acceptable = Column(Boolean, nullable=True)
    verification_status = Column(String(32), nullable=True)
    feedback_text = Column(Text, nullable=True)
    latency_ms = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
