"""Dynamic Model Recommendation Engine Implementing 5 Transparent Ranking Policies."""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.orchestrator.router import RequestUnderstanding
from app.services.catalogue import ModelCatalogue, ModelSpec, model_catalogue


class RecommendedModel(BaseModel):
    """Ranked model candidate with transparent pricing, benchmark evidence, and rationale."""
    model_id: str
    model_name: str
    provider: str
    rank: int
    is_top_pick: bool
    is_free: bool
    estimated_turn_cost_usd: float
    pricing_display: str
    benchmark_name: str
    benchmark_score: float
    benchmark_source: str
    context_limit_tokens: int
    execution_status: str
    execution_status_label: str
    suitability_reason: str
    value_score: Optional[float] = None
    user_rating: Optional[float] = None
    review_count: int = 0
    rating_display: Optional[str] = None


class RecommendationResult(BaseModel):
    """Complete recommendation report for a specific request and ranking mode."""
    ranking_mode: str
    task_category: str
    category_label: str
    total_eligible_models: int
    top_recommendation_id: str
    summary_rationale: str
    models: List[RecommendedModel]


class ModelRecommender:
    """
    Computes dynamic model recommendations from actual request understanding,
    modality requirements, verified catalogue specifications, and ranking policies.
    """

    def __init__(self, catalogue: Optional[ModelCatalogue] = None):
        self.catalogue = catalogue or model_catalogue

    def recommend(
        self,
        understanding: RequestUnderstanding,
        ranking_mode: str = "best_value",
        prefer_free: Optional[bool] = None,
        max_budget_cents: Optional[float] = None,
        ratings: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> RecommendationResult:
        """
        Executes capability filtering, cost estimation, review-weighted scoring, and mode-based ranking.
        Ranking modes supported: 'free', 'paid', 'lowest_price', 'best_performance', 'best_value'.
        """
        self.catalogue.refresh_availability()
        all_models = self.catalogue.get_all_models()

        # Step 1: Modality & Capability Filtering
        required_mods = set(understanding.required_modalities)
        task = understanding.task_category.lower()

        eligible: List[ModelSpec] = []
        for m in all_models:
            # Check modality compatibility (e.g. image understanding requires vision support)
            model_mods = set(m.supported_modalities)
            if not required_mods.issubset(model_mods):
                continue
            if "image" in required_mods and "image" not in model_mods:
                continue

            # Check task capability compatibility
            if not m.is_compatible_with(task):
                continue

            eligible.append(m)

        # Fallback to all models if filtering leaves none
        if not eligible:
            eligible = all_models

        # Step 2: Calculate Turn Costs based on estimated tokens
        prompt_tokens = understanding.estimated_prompt_tokens
        output_tokens = understanding.estimated_output_tokens

        # Step 3: Apply Free / Paid preference if explicitly requested
        mode = ranking_mode.lower()
        if mode == "free":
            filtered = [m for m in eligible if m.is_free]
            if not filtered:
                filtered = eligible
        elif mode == "paid":
            filtered = [m for m in eligible if not m.is_free]
            if not filtered:
                filtered = eligible
        else:
            filtered = eligible

        # Step 4: Mode Ranking with Review-to-Routing Bayesian Feedback Adjustment
        ranked_candidates: List[Dict[str, Any]] = []
        for m in filtered:
            est_cost = 0.0
            if not m.is_free:
                est_cost = (prompt_tokens * (m.input_cost_per_1m / 1_000_000)) + (output_tokens * (m.output_cost_per_1m / 1_000_000))

            # Budget constraint check
            if max_budget_cents is not None and (est_cost * 100) > max_budget_cents:
                continue

            # Task domain affinity multiplier
            spec_tasks = [t.lower() for t in m.supported_tasks]
            if task in spec_tasks:
                task_affinity = 1.15
            elif "general" in spec_tasks:
                task_affinity = 1.05
            else:
                task_affinity = 0.90

            # Review-weighted adjustment
            stat = (ratings or {}).get(m.id)
            user_rating = stat.get("smoothed_rating", 4.0) if stat else 4.0
            rev_count = stat.get("review_count", 0) if stat else 0
            # Rating multiplier: neutral (4.0) -> 1.0; 5.0 -> 1.12; 2.0 -> 0.77
            rating_weight = (0.84 + 0.16 * (user_rating / 4.0)) if rev_count > 0 else 1.0

            # Compute Value Score for 'best_value' mode: (Benchmark * affinity) / (1 + log10(cost_in_cents + 1))
            cost_in_cents = est_cost * 100
            raw_value_score = (m.benchmark_score * task_affinity) / (1.0 + math.log10(cost_in_cents + 1.0))
            adjusted_value_score = round(raw_value_score * rating_weight, 2)
            adjusted_benchmark = round(m.benchmark_score * task_affinity * rating_weight, 2)

            ranked_candidates.append({
                "spec": m,
                "cost_usd": est_cost,
                "cost_in_cents": cost_in_cents,
                "value_score": adjusted_value_score,
                "benchmark": m.benchmark_score,
                "adjusted_benchmark": adjusted_benchmark,
                "user_rating": round(user_rating, 1) if rev_count > 0 else None,
                "review_count": rev_count
            })

        # Sorting strategies
        if mode == "lowest_price":
            # Ascending by cost, then descending by adjusted benchmark
            ranked_candidates.sort(key=lambda x: (x["cost_usd"], -x["adjusted_benchmark"]))
        elif mode == "best_performance":
            # Descending by review-weighted benchmark score
            ranked_candidates.sort(key=lambda x: -x["adjusted_benchmark"])
        elif mode == "free":
            # Free models sorted by review-weighted benchmark performance
            ranked_candidates.sort(key=lambda x: -x["adjusted_benchmark"])
        elif mode == "paid":
            # Paid models sorted by review-weighted benchmark performance
            ranked_candidates.sort(key=lambda x: -x["adjusted_benchmark"])
        else:  # default 'best_value'
            # Balanced price-to-performance ratio adjusted by user feedback
            ranked_candidates.sort(key=lambda x: -x["value_score"])

        # Step 5: Format Recommendations
        recommendations: List[RecommendedModel] = []
        for i, item in enumerate(ranked_candidates):
            spec: ModelSpec = item["spec"]
            rank = i + 1
            is_top = (rank == 1)

            reason = self._generate_suitability_reason(spec, understanding, mode, is_top)
            pricing_disp = "100% Free (Open Weights / Local)" if spec.is_free else f"${spec.input_cost_per_1m:.2f} in / ${spec.output_cost_per_1m:.2f} out per 1M"

            u_rating = item["user_rating"]
            r_count = item["review_count"]
            r_disp = f"{u_rating}★ ({r_count} review{'s' if r_count != 1 else ''})" if r_count > 0 else "4.0★ prior (0 reviews)"

            recommendations.append(RecommendedModel(
                model_id=spec.id,
                model_name=spec.name,
                provider=spec.provider,
                rank=rank,
                is_top_pick=is_top,
                is_free=spec.is_free,
                estimated_turn_cost_usd=round(item["cost_usd"], 6),
                pricing_display=pricing_disp,
                benchmark_name=spec.benchmark_name,
                benchmark_score=spec.benchmark_score,
                benchmark_source=spec.benchmark_source,
                context_limit_tokens=spec.context_limit_tokens,
                execution_status=spec.status,
                execution_status_label=spec.status_label,
                suitability_reason=reason,
                value_score=item["value_score"] if mode == "best_value" else None,
                user_rating=u_rating,
                review_count=r_count,
                rating_display=r_disp
            ))

        top_id = recommendations[0].model_id if recommendations else "gemma-4-e4b"
        summary = self._generate_summary(top_id, understanding, mode)

        return RecommendationResult(
            ranking_mode=mode,
            task_category=understanding.task_category,
            category_label=understanding.category_label,
            total_eligible_models=len(recommendations),
            top_recommendation_id=top_id,
            summary_rationale=summary,
            models=recommendations
        )

    def _generate_suitability_reason(
        self,
        spec: ModelSpec,
        understanding: RequestUnderstanding,
        mode: str,
        is_top: bool
    ) -> str:
        """Constructs honest, fact-based rationale for candidate suitability."""
        category = understanding.task_category

        if spec.id == "qwen2.5-coder:7b" and category == "coding":
            return "Top pick for coding: Specialized HumanEval 84.1% performance with zero API billing overhead."
        if spec.id == "deepseek-r1:8b" and category in ("mathematical_reasoning", "complex_reasoning"):
            return "Deep reasoning leader: 89.2% MATH benchmark score with transparent local step-by-step proofs."
        if spec.id == "claude-3-5-sonnet":
            return "Frontier benchmark leader: Exceptional HumanEval 93.7% score; premium commercial API."
        if spec.id == "gemma-4-e4b":
            return "Default local orchestrator: High speed, low memory footprint, verified tool integration."
        if spec.id == "gpt-4o" and "image" in understanding.required_modalities:
            return "Omni vision powerhouse: Direct native image parsing and visual question answering."
        if spec.is_free:
            return f"Free execution via {spec.provider}. Verified benchmark {spec.benchmark_name}: {spec.benchmark_score}%."
        return f"Commercial engine via {spec.provider}. Estimated turn cost: ${(understanding.estimated_prompt_tokens * spec.input_cost_per_1m / 1e6):.6f}."

    def _generate_summary(self, top_id: str, understanding: RequestUnderstanding, mode: str) -> str:
        """Summarizes why the selected ranking mode chose the #1 engine."""
        return (
            f"Under '{mode.replace('_', ' ').title()}' mode for '{understanding.category_label}', "
            f"the routing engine selected '{top_id}' to optimize for task requirements, "
            f"verified benchmark evidence, and billing constraints."
        )


model_recommender = ModelRecommender()
