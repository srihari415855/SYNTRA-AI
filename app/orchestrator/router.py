"""Dynamic Request Understanding and Multi-Tier Intent Routing Pipeline."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.responses import RouteDecision

logger = get_logger("ai_system.router")


class RequestUnderstanding(BaseModel):
    """Deep structural and semantic classification of an incoming user request."""
    task_category: str = Field(..., description="Detected category e.g. 'coding', 'math', 'vision', etc.")
    category_label: str = Field(..., description="Human-friendly label e.g. 'Coding and Debugging'")
    confidence: float = Field(default=0.95, description="Confidence score of classification")
    required_modalities: List[str] = Field(default_factory=lambda: ["text"])
    required_tools: List[str] = Field(default_factory=list)
    complexity: str = Field(default="moderate", description="'simple', 'moderate', 'complex', 'high-reasoning'")
    estimated_prompt_tokens: int = Field(default=20)
    estimated_output_tokens: int = Field(default=150)
    user_constraints: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(..., description="Detailed explanation of why this routing was determined")


class IntentRouter:
    """
    Orchestration router responsible for dynamic request understanding,
    modality detection, and routing decision generation.
    """

    def __init__(self):
        # Deterministic fast rules
        self._math_regex = re.compile(r"^(?:calculate|calc|\bwhat is\b)\s*([\d\.\s\+\-\*\/\(\)\^]+)\??$", re.IGNORECASE)
        self._pure_math_regex = re.compile(r"^\s*[\d\.\s\+\-\*\/\(\)]+\s*$")
        self._sys_info_regex = re.compile(r"^(?:system\s*info|diagnostics|specs|os\s*info|sysinfo)\??$", re.IGNORECASE)
        self._weather_regex = re.compile(r"^(?:weather in|what is the weather in)\s+([a-zA-Z\s]+)\??$", re.IGNORECASE)

    def understand_request(
        self,
        prompt: str,
        has_image: bool = False,
        has_audio: bool = False,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> RequestUnderstanding:
        """
        Dynamically analyzes user input text and attached modalities to determine
        exact task category, required capabilities, token bounds, and reasoning.
        """
        cleaned = prompt.strip()
        lower = cleaned.lower()
        word_count = len(cleaned.split())
        est_prompt_tokens = max(10, int(word_count * 1.3))

        modalities = ["text"]
        if has_image:
            modalities.append("image")
        if has_audio:
            modalities.append("audio")

        # 1. Image Understanding (Multimodal Vision)
        if has_image or any(k in lower for k in ["look at this image", "describe this image", "what is in this photo", "ocr", "read image"]):
            return RequestUnderstanding(
                task_category="image_understanding",
                category_label="Image Understanding & Vision",
                confidence=0.98 if has_image else 0.85,
                required_modalities=list(set(modalities + ["image"])),
                required_tools=[],
                complexity="moderate" if word_count < 50 else "complex",
                estimated_prompt_tokens=est_prompt_tokens + 500,  # Image token overhead
                estimated_output_tokens=250,
                rationale="Visual media attached or requested. Requires vision-capable multimodal engine."
            )

        # 2. Direct Tool Execution / Math
        if self._math_regex.match(cleaned) or (len(cleaned) > 2 and self._pure_math_regex.match(cleaned)):
            return RequestUnderstanding(
                task_category="tool_execution",
                category_label="Tool Execution (Safe Arithmetic)",
                confidence=0.99,
                required_modalities=modalities,
                required_tools=["calculator"],
                complexity="simple",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=30,
                rationale="Detected pure mathematical expression matching AST calculator tool directly."
            )

        if self._sys_info_regex.match(cleaned):
            return RequestUnderstanding(
                task_category="tool_execution",
                category_label="Tool Execution (System Diagnostics)",
                confidence=0.99,
                required_modalities=modalities,
                required_tools=["system_info"],
                complexity="simple",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=50,
                rationale="Explicit system diagnostic command routed to isolated system_info tool."
            )

        if self._weather_regex.match(cleaned):
            return RequestUnderstanding(
                task_category="tool_execution",
                category_label="Tool Execution (Weather Service)",
                confidence=0.99,
                required_modalities=modalities,
                required_tools=["weather_lookup"],
                complexity="simple",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=50,
                rationale="Explicit location weather query routed to weather_lookup service."
            )

        # 3. Speech & Audio Processing
        if has_audio or any(k in lower for k in ["transcribe this", "audio transcript", "listen to this", "speech to text"]):
            return RequestUnderstanding(
                task_category="speech_processing",
                category_label="Speech & Audio Processing",
                confidence=0.96,
                required_modalities=list(set(modalities + ["audio"])),
                required_tools=[],
                complexity="moderate",
                estimated_prompt_tokens=est_prompt_tokens + 200,
                estimated_output_tokens=300,
                rationale="Audio recording or speech stream detected requiring audio recognition."
            )

        # 4. Coding & Debugging
        coding_signals = [
            "def ", "class ", "import ", "function", "javascript", "python",
            "sql", "query", "code", "bug", "refactor", "syntax", "git",
            "api endpoint", "html", "css", "dockerfile", "regex", "traceback",
            "compile", "typescript", "rust", "algorithm", "unittest", "fastapi"
        ]
        if any(s in lower for s in coding_signals):
            return RequestUnderstanding(
                task_category="coding",
                category_label="Coding & Debugging",
                confidence=0.94,
                required_modalities=modalities,
                required_tools=["code_runner"] if "run" in lower or "execute" in lower else [],
                complexity="complex" if word_count > 60 else "moderate",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=400,
                rationale="Contains programming semantics, code constructs, or debugging directives. Requires code-specialized model."
            )

        # 5. Mathematical Reasoning
        math_signals = [
            "solve equation", "calculus", "integral", "derivative", "matrix",
            "probability", "algebra", "theorem", "geometry", "linear algebra",
            "math problem", "calculate the probability", "standard deviation"
        ]
        if any(s in lower for s in math_signals):
            return RequestUnderstanding(
                task_category="mathematical_reasoning",
                category_label="Mathematical Reasoning",
                confidence=0.93,
                required_modalities=modalities,
                required_tools=["calculator"],
                complexity="complex",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=350,
                rationale="Advanced mathematical query requiring step-by-step symbolic or numerical deduction."
            )

        # 6. Structured Data Analysis
        data_signals = ["csv", "json", "pandas", "dataframe", "dataset", "tabular", "columns", "aggregate", "pivot"]
        if any(s in lower for s in data_signals):
            return RequestUnderstanding(
                task_category="structured_data_analysis",
                category_label="Structured Data Analysis",
                confidence=0.91,
                required_modalities=modalities,
                required_tools=[],
                complexity="moderate",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=300,
                rationale="Tabular or structured data format manipulation identified."
            )

        # 7. Summarization
        summarize_signals = ["summarize", "tl;dr", "tldr", "bullet points summary", "key takeaways", "condense", "abstract"]
        if any(s in lower for s in summarize_signals) or (word_count > 120 and "summary" in lower):
            return RequestUnderstanding(
                task_category="summarization",
                category_label="Summarization & Condensing",
                confidence=0.92,
                required_modalities=modalities,
                required_tools=[],
                complexity="moderate",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=150,
                rationale="Explicit condensation and summarization request detected."
            )

        # 8. Research & Q&A
        research_signals = ["what is the history of", "research paper", "explain the concept", "who was", "literature review", "scientific mechanism"]
        if any(s in lower for s in research_signals):
            return RequestUnderstanding(
                task_category="research",
                category_label="Research & Question Answering",
                confidence=0.88,
                required_modalities=modalities,
                required_tools=[],
                complexity="moderate",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=350,
                rationale="In-depth factual inquiry or scientific research topic."
            )

        # 9. General Writing & Creative
        writing_signals = ["write an email", "draft a letter", "blog post", "story", "creative writing", "revise", "rewrite", "tone"]
        if any(s in lower for s in writing_signals):
            return RequestUnderstanding(
                task_category="general_writing",
                category_label="General Writing & Composition",
                confidence=0.89,
                required_modalities=modalities,
                required_tools=[],
                complexity="moderate",
                estimated_prompt_tokens=est_prompt_tokens,
                estimated_output_tokens=300,
                rationale="Creative drafting, correspondence, or text restructuring request."
            )

        # Default: General Knowledge
        return RequestUnderstanding(
            task_category="general",
            category_label="General Knowledge & Dialogue",
            confidence=0.85,
            required_modalities=modalities,
            required_tools=[],
            complexity="simple" if word_count < 30 else "moderate",
            estimated_prompt_tokens=est_prompt_tokens,
            estimated_output_tokens=200,
            rationale="General query handled by primary conversational reasoning engine."
        )

    def route(self, prompt: str) -> RouteDecision:
        """Original fast route method for backward compatibility."""
        understanding = self.understand_request(prompt)

        if understanding.task_category == "tool_execution" and understanding.required_tools:
            return RouteDecision(
                route_type="deterministic",
                category="direct_tool",
                target_model=understanding.required_tools[0],
                rationale=understanding.rationale
            )

        if understanding.task_category == "coding":
            return RouteDecision(
                route_type="model_classified",
                category="coding",
                target_model=settings.CODING_MODEL,
                rationale=understanding.rationale
            )
        elif understanding.task_category in ("mathematical_reasoning", "complex_reasoning"):
            return RouteDecision(
                route_type="model_classified",
                category="complex_reasoning",
                target_model=settings.COMPLEX_REASONING_MODEL,
                rationale=understanding.rationale
            )
        else:
            return RouteDecision(
                route_type="model_classified",
                category="general",
                target_model=settings.DEFAULT_MODEL,
                rationale=understanding.rationale
            )


intent_router = IntentRouter()
